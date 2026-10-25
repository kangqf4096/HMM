"""Inspect pinned arXiv Parquet shards and refresh only unreadable cached files."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re

from .data import DATASET, SUBSET, REVISION, SPLITS
from .io import write_json
from .progress import progress


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
    dataset = config.get("dataset", DATASET)
    subset = config.get("subset", SUBSET)
    revision = config.get("dataset_revision", REVISION)
    if (dataset, subset) != (DATASET, SUBSET):
        raise ValueError(f"This repair command only handles {DATASET}, subset={SUBSET}.")
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

    cache = str(Path(".cache/hf/hub").resolve())
    pattern = re.compile(r"document/(train|validation|test)-\d+-of-\d+\.parquet")
    progress("读取 arXiv 分片清单")
    files = sorted((item for item in HfApi().list_repo_tree(
        dataset, repo_type="dataset", revision=revision, path_in_repo=subset)
        if pattern.fullmatch(item.path)), key=lambda item: item.path)
    if not files or {pattern.fullmatch(item.path)[1] for item in files} != set(SPLITS):
        raise RuntimeError("Pinned arXiv release does not contain all three expected Parquet splits.")
    packages = {}
    for name in ("datasets", "pyarrow", "huggingface_hub", "fsspec"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    report = {"dataset": dataset, "subset": subset, "revision": revision,
              "cache": cache, "packages": packages, "files": []}
    for index, item in enumerate(files, 1):
        kwargs = dict(repo_id=dataset, filename=item.path, repo_type="dataset",
                      revision=revision, cache_dir=cache)
        path = try_to_load_from_cache(**kwargs)
        existing = isinstance(path, str) and Path(path).is_file()
        before = inspect_parquet(path, item.size) if existing else {"ok": False, "reason": "not cached"}
        row = {"file": item.path, "expected_size": item.size, "before": before,
               "action": "reused" if before["ok"] else ("redownloaded" if existing else "downloaded")}
        if existing:
            row["path"] = str(path)
        report["files"].append(row)
        write_json(report_path, report)
        print(f"{index}/{len(files)} {item.path}: {row['action']} · {before.get('reason', 'metadata readable')}", flush=True)
        if not before["ok"]:
            progress("下载异常或缺失的 arXiv 分片", index - 1, len(files), item.path)
            try:
                path = hf_hub_download(**kwargs, force_download=existing)
            except Exception as error:
                row["download_error"] = f"{type(error).__name__}: {error}"
                write_json(report_path, report)
                raise
        row["path"] = str(path)
        row["after"] = inspect_parquet(path, item.size)
        write_json(report_path, report)
        if not row["after"]["ok"]:
            raise RuntimeError(f"arXiv shard is still unreadable after download: {item.path}; "
                               f"{row['after']['reason']}. Report: {report_path}")
        progress("检查 arXiv 分片", index, len(files), item.path)
    report["refreshed"] = sum(row["action"] != "reused" for row in report["files"])
    report["rows_by_split"] = {split: sum(row["after"]["rows"] for row in report["files"]
                                         if pattern.fullmatch(row["file"])[1] == split) for split in SPLITS}
    write_json(report_path, report)
    print(f"检查完成：{len(files)} 个分片，下载/重新下载 {report['refreshed']} 个。报告：{report_path}", flush=True)
    progress("arXiv 缓存检查完成", len(files), len(files))
    return report
