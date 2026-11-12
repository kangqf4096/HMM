"""Two-slot local staging; only release a batch after its remote copy succeeds."""
import shutil
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import mkdtemp

from .io import read_jsonl, write_json, write_jsonl
from .memory import cache_lock, cgroup_path, memory_usage


class BatchCache:
    def __init__(self, root, reserve_bytes=1024 ** 3, *, fallback_root=None,
                 spill_bytes=90_000_000_000):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.root = Path(mkdtemp(prefix="g-", dir=root))
        self.reserve_bytes = reserve_bytes
        self.fallback = None
        self.spilled = False
        self.lock_path = None
        self.stop = threading.Event()
        self.monitor = None
        self.error = None
        if fallback_root is not None:
            self.fallback = Path(fallback_root) / self.root.name
            self.fallback.mkdir(parents=True, exist_ok=False)
            self.lock_path = str(self.root / '.lock')
            self.group = cgroup_path()
            self.spill_bytes = spill_bytes
            self.monitor = threading.Thread(target=self._watch, daemon=True)
            self.monitor.start()

    def locked(self):
        return cache_lock(self.lock_path)

    def _watch(self):
        while not self.stop.wait(.5):
            try:
                if memory_usage(self.group) >= self.spill_bytes:
                    self.spill()
            except Exception as error:
                self.error = error
                print(f'[cache] pressure spill failed; original cache retained: {error}', flush=True)
                return

    def spill(self):
        """Pause all trace writers; publish a disk copy before freeing RAM."""
        with self.locked():
            self._spill_locked()

    def _spill_locked(self):
        if self.spilled or self.fallback is None:
            return
        for batch in self.root.glob('b*'):
            if batch.is_symlink():
                continue
            target = self.fallback / batch.name
            partial = target.with_name(target.name + '.part')
            shutil.copytree(batch, partial, copy_function=self._spill_file)
            partial.rename(target)
            # Every writer/reader holds .lock, including during this brief gap.
            shutil.rmtree(batch)
            batch.symlink_to(target, target_is_directory=True)
        self.spilled = True
        print(f'[cache] cgroup pressure: spilled to {self.fallback}; continuing on disk', flush=True)

    @staticmethod
    def _spill_file(source, target):
        # Network file pages also count against cgroup memory. Flush the copy and
        # ask Linux to evict its pages before releasing the corresponding tmpfs.
        with open(source, 'rb') as src, open(target, 'wb') as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
            if hasattr(os, 'posix_fadvise'):
                try:
                    os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass  # Some network filesystems do not implement this hint.
        shutil.copystat(source, target)
        return target

    def stage(self, number, jobs, manifest, estimated_bytes):
        with self.locked():
            return self._stage(number, jobs, manifest, estimated_bytes)

    def _stage(self, number, jobs, manifest, estimated_bytes):
        if self.error:
            raise RuntimeError('Cache pressure monitor failed') from self.error
        if self.fallback is not None and (memory_usage(self.group) + estimated_bytes >= self.spill_bytes
                                         or shutil.disk_usage(self.root).free < self.reserve_bytes + estimated_bytes):
            self._spill_locked()
        staging = self.fallback if self.spilled else self.root
        free = shutil.disk_usage(staging).free
        if free < self.reserve_bytes + estimated_bytes:
            raise OSError(f"Local cache space insufficient: free={free}, "
                          f"required={self.reserve_bytes + estimated_bytes}; cache={self.root}")
        batch = self.root / f"b{number}"
        if self.spilled:
            target = staging / batch.name
            target.mkdir()
            batch.symlink_to(target, target_is_directory=True)
        else:
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
                with self.locked():
                    staged_jobs = read_jsonl(batch / "jobs.jsonl")
                yield start, batch, staged_jobs

    def commit(self, batch, destination):
        with self.locked():
            self._commit(batch, destination)

    def _commit(self, batch, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        partial = destination.with_name(destination.name + ".part")
        # On failure keep both the local batch and partial remote copy for inspection.
        shutil.copytree(batch, partial)
        partial.rename(destination)
        if batch.is_symlink():
            target = batch.resolve()
            batch.unlink()
            shutil.rmtree(target)
        else:
            shutil.rmtree(batch)

    def finish(self):
        self.close()
        if self.lock_path:
            Path(self.lock_path).unlink(missing_ok=True)
        self.root.rmdir()  # Never remove uncommitted data, including prefetched batches.
        if self.fallback is not None:
            self.fallback.rmdir()

    def close(self):
        self.stop.set()
        if self.monitor:
            self.monitor.join()
        if self.error:
            raise RuntimeError('Cache pressure monitor failed; retained cache for inspection') from self.error
