"""Read-only dataset checks: enumerate all shards, decode data, report link targets."""
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic

from .arxiv_cache import inspect_parquet
from .arxiv_files import PATTERN, cached_parquet_source
from .data import DATASET, REVISION, SPLITS, SUBSET
from .io import write_json
from .paths import project_path, project_root, runtime_root
from .progress import progress

ROW_COUNTS = {"train": 203037, "validation": 6436, "test": 6440}


def shard_manifest(revision, offline):
    names = ([f"document/train-{index:05d}-of-00015.parquet" for index in range(15)]
             + [f"document/{split}-00000-of-00001.parquet" for split in ("validation", "test")])
    if offline:
        if revision != REVISION:
            raise ValueError("Offline shard names are only recorded for the pinned arXiv revision.")
        return [{"file": name, "size": None} for name in sorted(names)]
    from huggingface_hub import HfApi
    items = [{"file": item.path, "size": item.size} for item in HfApi().list_repo_tree(
        DATASET, repo_type="dataset", revision=revision, path_in_repo=SUBSET)
        if PATTERN.fullmatch(item.path)]
    if not items or {PATTERN.fullmatch(item["file"])[1] for item in items} != set(SPLITS):
        raise RuntimeError("Remote manifest does not contain all three arXiv splits.")
    if revision == REVISION and {item["file"] for item in items} != set(names):
        raise RuntimeError("Remote manifest differs from the 17 pinned arXiv shard names.")
    return sorted(items, key=lambda item: item["file"])


def inspect_contents(path, expected_size=None, *, full=True):
    """Check metadata and optionally decode every column in bounded batches."""
    path = Path(path)
    result = {"path": str(path), "exists": path.exists(), "symlink": path.is_symlink()}
    if result["symlink"]:
        try:
            result["link_target"] = str(path.readlink())
        except OSError as error:
            result["link_error"] = str(error)
    if not result["exists"]:
        return {**result, "ok": False, "reason": "missing or dangling link"}
    result.update(inspect_parquet(path, expected_size))
    if not result["ok"]:
        try:
            if path.stat().st_size <= 4096:
                result["small_file_text"] = path.read_bytes()[:256].decode("utf-8", errors="replace")
        except OSError:
            pass
        return result
    if not full:
        return {**result, "scan": "metadata_only"}
    import pyarrow.parquet as pq

    decoded = 0
    last_progress = 0
    try:
        reader = pq.ParquetFile(str(path), memory_map=False, pre_buffer=False)
        try:
            result["columns"] = reader.schema_arrow.names
            for batch in reader.iter_batches(batch_size=128, use_threads=False):
                batch.validate(full=True)
                decoded += batch.num_rows
                if monotonic() - last_progress >= .5:
                    progress("逐批读取数据分片", decoded, result["rows"], path.name)
                    last_progress = monotonic()
        finally:
            reader.close()
        if decoded != result["rows"]:
            raise ValueError(f"Decoded row count differs: {decoded} != {result['rows']}")
        result.update(scan="all_columns", decoded_rows=decoded)
    except Exception as error:
        result.update(ok=False, decoded_rows=decoded, reason=f"{type(error).__name__}: {error}")
    return result


def check_arxiv(config, *, project=None, report_path="tmp/arxiv-check.json", full=True, offline=False):
    """Inspect every shard, continuing on bad files. Only diagnostic reports are written."""
    if (config.get("dataset", DATASET), config.get("subset", SUBSET)) != (DATASET, SUBSET):
        raise ValueError(f"This checker handles only {DATASET}/{SUBSET}.")
    # Fail clearly on missing dependencies instead of classifying data as invalid.
    import pyarrow.parquet

    project = project_root(project)
    report_path = project_path(report_path, project)
    revision = config.get("dataset_revision", REVISION)
    archive = project / "data/arxiv" / revision
    repo_cache = project / ".cache/hf/hub/datasets--ccdv--arxiv-summarization"
    runtime = runtime_root(project)
    if any(report_path.resolve().is_relative_to(path.resolve()) for path in (archive, repo_cache)):
        raise ValueError("Write diagnostic reports outside the dataset and Hub cache directories.")
    progress("读取分片清单", detail="仅查询元数据，不下载数据文件" if not offline else "使用固定版本的离线清单")
    items = shard_manifest(revision, offline)
    packages = {}
    for package in ("pyarrow", "datasets", "huggingface_hub"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    report = {"dataset": DATASET, "subset": SUBSET, "revision": revision,
              "project": str(project), "archive": str(archive), "packages": packages,
              "mode": "all_columns" if full else "metadata_only", "offline": offline,
              "expected_files": len(items), "files": []}
    checks = {}

    def inspect(path, size):
        key = str(path)
        if key not in checks:
            checks[key] = inspect_contents(path, size, full=full)
        return checks[key]

    for ordinal, item in enumerate(items, 1):
        name, size = item["file"], item["size"]
        path = archive / name
        primary = inspect(path, size)
        row = {"file": name, "split": PATTERN.fullmatch(name)[1], "expected_size": size,
               "primary": primary, "cache_attempts": []}
        report["files"].append(row)
        if primary["ok"] and not primary["symlink"]:
            row.update(status="ready", readable_source=primary)
        else:
            # Check the current file too: a 79-byte placeholder may point into old blobs.
            origins = [path, repo_cache / "snapshots" / revision / name]
            if runtime != project:
                origins.append(runtime / ".cache/arxiv" / revision / name)
            for origin in origins:
                source, attempts = cached_parquet_source(origin, repo_cache, size)
                row["cache_attempts"].extend(attempts)
                if source is None:
                    continue
                candidate = inspect(source, size)
                if candidate["ok"]:
                    row.update(status="recoverable", readable_source=candidate)
                    break
            else:
                row["status"] = "missing" if not primary["exists"] else "unreadable"
        message = f"{ordinal}/{len(items)} [{row['status']}] {name} · {primary.get('size', 'missing')} bytes"
        if not primary["ok"]:
            message += f" · {primary['reason']}"
        if row["status"] == "recoverable":
            message += f"\n  已有可读数据：{row['readable_source']['path']}"
        if primary.get("small_file_text"):
            message += "\n  小文件内容：" + repr(primary["small_file_text"])
        print(message, flush=True)
        progress("检查全部 arXiv 分片", ordinal, len(items), f"{row['status']} · {name}")
        write_json(report_path, report)
    counts = Counter(row["status"] for row in report["files"])
    available_rows = {split: sum(row.get("readable_source", {}).get("rows", 0)
                                 for row in report["files"] if row["split"] == split) for split in SPLITS}
    expected_rows = ROW_COUNTS if revision == REVISION else None
    row_counts_ok = expected_rows is None or available_rows == expected_rows
    report["summary"] = {
        "counts": {name: counts[name] for name in ("ready", "recoverable", "missing", "unreadable")},
        "archive_ready": counts["ready"] == len(items) and row_counts_ok,
        "all_bytes_available": counts["ready"] + counts["recoverable"] == len(items) and row_counts_ok,
        "available_rows_by_split": available_rows, "expected_rows_by_split": expected_rows,
        "row_counts_match": row_counts_ok,
        "recoverable_files": [row["file"] for row in report["files"] if row["status"] == "recoverable"],
        "no_readable_source": [row["file"] for row in report["files"] if row["status"] in ("missing", "unreadable")],
    }
    write_json(report_path, report)
    lines = ["# arXiv dataset check", "", f"Project: `{project}`", f"Mode: `{report['mode']}`", "",
             "No dataset files were downloaded, copied, removed, or modified. No file hashes were computed.", "",
             "| File | Status | Primary bytes | Readable source |", "|---|---|---:|---|"]
    for row in report["files"]:
        source = row.get("readable_source", {}).get("path", "")
        lines.append(f"| {row['file']} | {row['status']} | {row['primary'].get('size', '')} | {source} |")
    lines.extend(["", f"Archive ready: {report['summary']['archive_ready']}",
                  f"All source bytes available: {report['summary']['all_bytes_available']}",
                  f"Rows by split: {available_rows}", f"Expected rows: {expected_rows}", ""])
    report_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n检查完成：{dict(counts)}\n持久目录可直接使用：{report['summary']['archive_ready']}"
          f"\n全部分片已有可读数据：{report['summary']['all_bytes_available']}"
          f"\n行数：{available_rows}\n报告：{report_path}", flush=True)
    progress("检查完成", len(items), len(items), "只生成报告，未修改数据集")
    return report
