"""Two-slot local staging; only release a batch after its remote copy succeeds."""
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import mkdtemp

from .io import read_jsonl, write_json, write_jsonl


class BatchCache:
    def __init__(self, root, reserve_bytes=1024 ** 3):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.root = Path(mkdtemp(prefix="g-", dir=root))
        self.reserve_bytes = reserve_bytes

    def stage(self, number, jobs, manifest, estimated_bytes):
        free = shutil.disk_usage(self.root).free
        if free < self.reserve_bytes + estimated_bytes:
            raise OSError(f"Local cache space insufficient: free={free}, "
                          f"required={self.reserve_bytes + estimated_bytes}; cache={self.root}")
        batch = self.root / f"b{number}"
        batch.mkdir()
        write_json(batch / "manifest.json", manifest)
        write_jsonl(batch / "jobs.jsonl", jobs)
        (batch / "trace").mkdir()
        return batch

    def batches(self, jobs, size, manifest, bytes_per_job):
        if type(size) is not int or size < 1:
            raise ValueError("Batch size must be a positive integer.")
        if not jobs:
            return
        # A single producer can stage only the next batch. The caller must commit
        # and release the current batch before advancing this iterator.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.stage, 0, jobs[:size], manifest,
                                 min(2 * size, len(jobs)) * bytes_per_job)
            for start in range(0, len(jobs), size):
                batch = future.result()
                following = start + size
                if following < len(jobs):
                    part = jobs[following:following + size]
                    future = pool.submit(self.stage, following // size, part, manifest,
                                         (len(part) + len(jobs[start:start + size])) * bytes_per_job)
                yield start, batch, read_jsonl(batch / "jobs.jsonl")

    def commit(self, batch, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        partial = destination.with_name(destination.name + ".part")
        # On failure keep both the local batch and partial remote copy for inspection.
        shutil.copytree(batch, partial)
        partial.rename(destination)
        shutil.rmtree(batch)

    def finish(self):
        self.root.rmdir()  # Never remove uncommitted data, including prefetched batches.
