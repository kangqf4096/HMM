"""Inspect pinned arXiv Parquet shards and refresh only unreadable cached files."""
from pathlib import Path

from .data import DATASET, SUBSET, REVISION


def inspect_parquet(path, expected_size=None):
    """Read file boundaries and Parquet metadata, without hashing the file body."""
    path = Path(path)
    try:
        size = path.stat().st_size
        if size < 12:
            return {"ok": False, "size": size, "reason": "file is too short for a Parquet header/footer"}
        with path.open("rb") as handle:
            head = handle.read(64)
            handle.seek(-8, 2)
            footer = handle.read(8)
        if not head.startswith(b"PAR1"):
            kind = "Git LFS pointer" if head.startswith(b"version https://git-lfs") else "non-Parquet content"
            return {"ok": False, "size": size, "reason": f"missing PAR1 header ({kind})"}
        if footer[-4:] != b"PAR1":
            return {"ok": False, "size": size, "reason": "missing PAR1 footer"}
        metadata_size = int.from_bytes(footer[:4], "little")
        if metadata_size > size - 12:
            return {"ok": False, "size": size, "reason": "Parquet footer length exceeds file size"}
        if expected_size is not None and size != expected_size:
            return {"ok": False, "size": size, "reason": f"file size differs: {size} != {expected_size}"}
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(path)
        columns = metadata.schema.to_arrow_schema().names
        if "article" not in columns:
            return {"ok": False, "size": size, "reason": "missing article column"}
        return {"ok": True, "size": size, "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups}
    except Exception as error:
        # ArrowInvalid/ArrowIOError are not consistently subclasses of OSError.
        return {"ok": False, "reason": f"{type(error).__name__}: {error}"}


def repair_arxiv(config, report_path="tmp/arxiv-cache.json"):
    """Recover link-backed caches into ordinary persistent and working files."""
    from .arxiv_files import ensure_arxiv_files

    _, report = ensure_arxiv_files(
        config.get("dataset", DATASET), config.get("subset", SUBSET),
        config.get("dataset_revision", REVISION), report_path=report_path)
    print(f"arXiv files ready; newly downloaded {report['refreshed']} files. Report: {report_path}", flush=True)
    return report
