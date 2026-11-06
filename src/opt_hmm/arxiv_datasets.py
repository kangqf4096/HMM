"""Reuse the pinned datasets Arrow cache before resolving anything on the Hub."""
import inspect
from pathlib import Path

from .paths import project_path
from .progress import progress

SPLITS = ("train", "validation", "test")


def load_cached_arxiv(dataset, subset, revision, cache_dir):
    """Open the exact release with datasets' cache reader, without Hub queries."""
    config_dir = Path(cache_dir) / dataset.replace("/", "___") / subset
    for version_dir in sorted(config_dir.glob("*")):
        cached_dir = version_dir / revision
        if not (cached_dir / "dataset_info.json").is_file():
            continue
        # This is the reader used by datasets 3/4 for its offline fallback.
        # Supply the release explicitly: its automatic lookup picks the newest
        # cached release, which need not be the revision requested here.
        from datasets.packaged_modules.cache.cache import Cache

        builder = Cache(cache_dir=str(cache_dir), repo_id=dataset,
                        dataset_name=dataset.rsplit("/", 1)[-1],
                        config_name=subset, version=version_dir.name, hash=revision)
        if not builder.info.splits or any(split not in builder.info.splits for split in SPLITS):
            continue
        # Newer datasets releases dropped run_post_process from as_dataset().
        kwargs = ({"run_post_process": False}
                  if "run_post_process" in inspect.signature(builder.as_dataset).parameters else {})
        try:
            return builder.as_dataset(**kwargs)
        except FileNotFoundError:
            # A missing Arrow shard is a cache miss, not a completed release.
            continue
    return None


def load_arxiv_datasets(dataset, subset, revision):
    """Use local data first; resolve/download only when the release is absent."""
    from datasets import config as datasets_config, load_dataset

    cache_dir = project_path(".cache/hf/datasets")
    progress("查找 arXiv 本地缓存", detail=f"版本 {revision}")
    raw = load_cached_arxiv(dataset, subset, revision, cache_dir)
    if raw is not None:
        progress("复用 arXiv 本地缓存", detail="已跳过远程查询")
        return raw
    if datasets_config.HF_HUB_OFFLINE:
        raise FileNotFoundError(
            f"No complete local cache for {dataset}/{subset} at revision {revision} "
            f"in {cache_dir}; offline mode is enabled.")
    progress("本地缓存未命中，加载 arXiv", detail="允许联网；下载和 Arrow 处理进度由 datasets 报告")
    return load_dataset(dataset, subset, revision=revision, cache_dir=str(cache_dir))
