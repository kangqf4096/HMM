"""Read complete source rows with ParquetFile, without the datasets fragment scan."""
from bisect import bisect_right
from collections import OrderedDict
from operator import index as integer_index
from pathlib import Path
import re

from .progress import progress


class ParquetSplit:
    """An article-only random-access view preserving shard and row order."""

    def __init__(self, paths, cache_bytes=64 * 1024**2):
        import pyarrow.parquet as pq

        self.groups, self.ends = [], []
        self.cache, self.cache_bytes = OrderedDict(), 0
        self.cache_limit = cache_bytes
        total = 0
        for path in paths:
            path = Path(path)
            reader = pq.ParquetFile(str(path), memory_map=False, pre_buffer=False)
            try:
                if "article" not in reader.schema_arrow.names:
                    raise ValueError(f"Missing article column: {path}")
                for group in range(reader.num_row_groups):
                    total += reader.metadata.row_group(group).num_rows
                    self.groups.append((path, group))
                    self.ends.append(total)
            finally:
                reader.close()
        self.rows = total

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        import pyarrow.parquet as pq

        index = integer_index(index)
        if index < 0:
            index += self.rows
        if not 0 <= index < self.rows:
            raise IndexError(index)
        ordinal = bisect_right(self.ends, index)
        start = self.ends[ordinal - 1] if ordinal else 0
        key = self.groups[ordinal]
        if key in self.cache:
            table = self.cache.pop(key)
            self.cache[key] = table
        else:
            path, group = key
            reader = pq.ParquetFile(str(path), memory_map=False, pre_buffer=False)
            try:
                table = reader.read_row_group(group, columns=["article"], use_threads=False)
            except Exception as error:
                raise RuntimeError(f"Cannot read arXiv article column: {path}, row group {group}: {error}") from error
            finally:
                reader.close()
            if table.num_rows != self.ends[ordinal] - start:
                raise ValueError(f"Parquet row count changed while reading: {key}")
            if table.nbytes <= self.cache_limit:
                while self.cache and (len(self.cache) >= 2 or self.cache_bytes + table.nbytes > self.cache_limit):
                    _, removed = self.cache.popitem(last=False)
                    self.cache_bytes -= removed.nbytes
                self.cache[key] = table
                self.cache_bytes += table.nbytes
        return {"article": table.column("article")[index - start].as_py()}


def load_arxiv_parquet(dataset, subset, revision):
    """Resolve the pinned Hub files and expose all official rows for sampling."""
    from huggingface_hub import HfApi, hf_hub_download

    if (dataset, subset) != ("ccdv/arxiv-summarization", "document"):
        raise ValueError("ParquetFile reader requires ccdv/arxiv-summarization/document.")
    splits = ("train", "validation", "test")
    pattern = re.compile(r"document/(train|validation|test)-\d+-of-\d+\.parquet")
    progress("直接读取 arXiv Parquet 元数据", detail="复用已有分片，不生成全语料 Arrow 缓存")
    names = sorted(name for name in HfApi().list_repo_files(dataset, repo_type="dataset", revision=revision)
                   if pattern.fullmatch(name))
    if {pattern.fullmatch(name)[1] for name in names} != set(splits):
        raise ValueError("Pinned arXiv release is missing an official split.")
    paths = {split: [] for split in splits}
    for ordinal, name in enumerate(names, 1):
        path = hf_hub_download(repo_id=dataset, filename=name, revision=revision,
                               repo_type="dataset", cache_dir=str(Path(".cache/hf/hub").resolve()))
        paths[pattern.fullmatch(name)[1]].append(path)
        progress("准备 arXiv Parquet 分片", ordinal, len(names), name)
    raw = {}
    progress("读取 arXiv 行组元数据", 0, len(splits))
    for ordinal, split in enumerate(splits, 1):
        raw[split] = ParquetSplit(paths[split])
        progress("读取 arXiv 行组元数据", ordinal, len(splits), f"{split}: {len(raw[split])} 篇")
    return raw
