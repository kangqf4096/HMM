"""Keep ordinary Parquet files on persistent storage and read local working copies."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import re

from .arxiv_cache import inspect_parquet
from .io import write_json
from .progress import progress

DATASET = "ccdv/arxiv-summarization"
PATTERN = re.compile(r"document/(train|validation|test)-\d+-of-\d+\.parquet")
SPLITS = ("train", "validation", "test")


def cached_parquet_source(path, repo_cache, expected_size):
    """Find actual bytes behind cached links or small text link placeholders."""
    repo_cache = Path(repo_cache).resolve()
    queue = [Path(path)] if isinstance(path, (str, Path)) else []
    seen, attempts = set(), []
    while queue and len(seen) < 16:
        candidate = queue.pop(0)
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        check = inspect_parquet(candidate, expected_size)
        attempts.append({"path": key, "symlink": candidate.is_symlink(), **check})
        if check["ok"]:
            return candidate, attempts
        try:
            # Drive can expose a Hub link target as a tiny text file.
            if not candidate.is_file() or candidate.stat().st_size > 4096:
                continue
            text = candidate.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            continue
        match = re.fullmatch(r"(?:\.\./)*blobs/([a-fA-F0-9/]+)", text)
        if not match:
            continue
        tail = match[1]
        target = repo_cache / "blobs" / tail
        if target.resolve().is_relative_to(repo_cache):
            queue.append(target)
        # Hub versions use both blobs/<id> and blobs/<prefix>/<id> layouts.
        basename = tail.rsplit("/", 1)[-1]
        for alternative in (repo_cache / "blobs" / basename,
                            repo_cache / "blobs" / basename[:2] / basename):
            if alternative.resolve().is_relative_to(repo_cache):
                queue.append(alternative)
    return None, attempts


def copy_parquet(source, destination, expected_size, stage):
    """Copy bytes to a regular file, reporting byte progress and keeping the source."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    size = source.stat().st_size
    copied = 0
    progress(stage, 0, size, destination.name)
    with source.open("rb") as src, partial.open("wb") as dst:
        while chunk := src.read(8 * 1024**2):
            dst.write(chunk)
            copied += len(chunk)
            progress(stage, copied, size, destination.name)
    check = inspect_parquet(partial, expected_size)
    if not check["ok"]:
        raise RuntimeError(f"Copied arXiv file is unreadable: {partial}; {check['reason']}")
    partial.replace(destination)
    if destination.is_symlink():
        raise RuntimeError(f"Expected an ordinary Parquet file, found a link: {destination}")
    return check


def ensure_arxiv_files(dataset, subset, revision, *, project=None, runtime=None,
                       report_path="tmp/arxiv-files.json"):
    """Recover existing blobs first; download only when no usable file is available."""
    if (dataset, subset) != (DATASET, "document"):
        raise ValueError("Ordinary Parquet storage requires ccdv/arxiv-summarization/document.")
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

    project = Path(project or os.environ.get("OPT_HMM_PROJECT", Path.cwd())).resolve()
    runtime = Path(runtime or os.environ.get("OPT_HMM_RUNTIME", project)).resolve()
    archive = project / "data" / "arxiv" / revision
    work = runtime / ".cache" / "arxiv" / revision if runtime != project else archive
    hub_cache = project / ".cache/hf/hub"
    repo_cache = hub_cache / "datasets--ccdv--arxiv-summarization"
    report_path = Path(report_path)
    if not report_path.is_absolute():
        report_path = project / report_path
    progress("读取 arXiv 分片清单")
    items = sorted((item for item in HfApi().list_repo_tree(
        dataset, repo_type="dataset", revision=revision, path_in_repo=subset)
        if PATTERN.fullmatch(item.path)), key=lambda item: item.path)
    if not items or {PATTERN.fullmatch(item.path)[1] for item in items} != set(SPLITS):
        raise RuntimeError("Pinned arXiv release does not contain all three expected splits.")
    packages = {}
    for name in ("pyarrow", "datasets", "huggingface_hub", "fsspec"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    report = {"dataset": dataset, "subset": subset, "revision": revision,
              "archive": str(archive), "work": str(work), "packages": packages, "files": []}
    paths = {split: [] for split in SPLITS}
    for ordinal, item in enumerate(items, 1):
        local, persistent = work / item.path, archive / item.path
        row = {"file": item.path, "expected_size": item.size,
               "path": str(local), "persistent_path": str(persistent)}
        report["files"].append(row)
        write_json(report_path, report)
        progress("准备普通 Parquet 文件", ordinal - 1, len(items), item.path)
        local_check = inspect_parquet(local, item.size) if local.is_file() and not local.is_symlink() else {"ok": False}
        if local_check["ok"]:
            row["action"] = "reused_local"
        else:
            source = None
            if persistent.is_file() and not persistent.is_symlink():
                persistent_check = inspect_parquet(persistent, item.size)
                if persistent_check["ok"]:
                    source = persistent
            if source is None:
                cached = try_to_load_from_cache(repo_id=dataset, filename=item.path, revision=revision,
                                                repo_type="dataset", cache_dir=str(hub_cache))
                source, row["cache_attempts"] = cached_parquet_source(cached, repo_cache, item.size)
            try:
                if source is not None:
                    row["source"] = str(source)
                    copy_parquet(source, local, item.size, "复制 arXiv 到本地工作目录")
                    row["action"] = "copied_existing"
                else:
                    print(f"No readable existing bytes for {item.path}; downloading a regular local file.", flush=True)
                    downloaded = hf_hub_download(repo_id=dataset, filename=item.path, revision=revision,
                                                 repo_type="dataset", local_dir=str(work),
                                                 force_download=local.exists())
                    if Path(downloaded).resolve() != local.resolve():
                        copy_parquet(downloaded, local, item.size, "保存 arXiv 本地普通文件")
                    row["action"] = "downloaded"
                local_check = inspect_parquet(local, item.size)
                if not local_check["ok"] or local.is_symlink():
                    raise RuntimeError(f"Local arXiv Parquet is unreadable: {local}; {local_check}")
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                write_json(report_path, report)
                raise
        row["after"] = local_check
        if persistent != local:
            persistent_check = inspect_parquet(persistent, item.size) if persistent.is_file() and not persistent.is_symlink() else {"ok": False}
            if not persistent_check["ok"]:
                copy_parquet(local, persistent, item.size, "保存普通 arXiv 分片到 Drive")
        paths[PATTERN.fullmatch(item.path)[1]].append(local)
        write_json(report_path, report)
        print(f"{ordinal}/{len(items)} {item.path}: {row['action']} · {local_check['rows']} rows · {local}", flush=True)
        progress("普通 Parquet 分片已就绪", ordinal, len(items), item.path)
    report["refreshed"] = sum(row["action"] == "downloaded" for row in report["files"])
    report["rows_by_split"] = {split: sum(row["after"]["rows"] for row in report["files"]
                                         if PATTERN.fullmatch(row["file"])[1] == split) for split in SPLITS}
    write_json(report_path, report)
    return paths, report
