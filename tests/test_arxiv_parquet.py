import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

from opt_hmm.arxiv_parquet import ParquetSplit, load_arxiv_parquet
from opt_hmm.data import ROLES, select_articles
from opt_hmm.io import read_json


class WordTokenizer:
    def encode(self, text, **kwargs):
        return [int(value) for value in text.split()][:kwargs["max_length"]]


def article(index):
    return {"article": " ".join([str(index)] * 150), "abstract": "never used"}


@unittest.skipUnless(pa is not None, "real Parquet reader tests require pyarrow")
class ParquetReaderTests(unittest.TestCase):
    def folder(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="pq")

    def write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
        return path

    def test_indexing_across_shards_and_row_groups_with_bounded_cache(self):
        with self.folder() as folder:
            root = Path(folder)
            rows = [article(i) for i in range(13)]
            paths = [self.write(root / "a.parquet", rows[:5]), self.write(root / "b.parquet", rows[5:])]
            split = ParquetSplit(paths, cache_bytes=1)
            self.assertEqual(len(split), len(rows))
            for index in (12, 0, 4, 5, 8, 3, 11):
                self.assertEqual(split[index]["article"], rows[index]["article"])
            self.assertEqual(split[-1]["article"], rows[-1]["article"])
            self.assertFalse(split.cache)
            with self.assertRaises(IndexError):
                split[13]

    def test_document_sampling_matches_identical_in_memory_source_rows(self):
        with self.folder() as folder:
            root = Path(folder)
            raw = {"train": [article(i) for i in range(10)],
                   "validation": [article(i) for i in range(10, 14)],
                   "test": [article(0), article(14), article(15), article(16)]}
            direct = {}
            for split, rows in raw.items():
                paths = [self.write(root / f"{split}-0.parquet", rows[:2]),
                         self.write(root / f"{split}-1.parquet", rows[2:])]
                direct[split] = ParquetSplit(paths)
            config = read_json(ROOT / "configs/colab_arxiv.json")
            config["roles"] = dict.fromkeys(ROLES, 1)
            reference = select_articles(raw, WordTokenizer(), config)
            self.assertEqual(select_articles(direct, WordTokenizer(), config), reference)

    def test_load_uses_staged_files_for_the_pinned_release(self):
        with self.folder() as folder:
            root = Path(folder)
            paths = {f"document/{split}-00000-of-00001.parquet":
                     self.write(root / f"{split}.parquet", [article(index)])
                     for index, split in enumerate(("train", "validation", "test"))}
            staged = {split: [paths[f"document/{split}-00000-of-00001.parquet"]]
                      for split in ("train", "validation", "test")}
            with patch("opt_hmm.arxiv_files.ensure_arxiv_files", return_value=(staged, {})) as ensure:
                loaded = load_arxiv_parquet("ccdv/arxiv-summarization", "document", "pinned")
            self.assertEqual({key: len(value) for key, value in loaded.items()},
                             {"train": 1, "validation": 1, "test": 1})
            ensure.assert_called_once_with("ccdv/arxiv-summarization", "document", "pinned")

    def test_missing_article_column_fails_before_sampling(self):
        with self.folder() as folder:
            path = Path(folder) / "a.parquet"
            pq.write_table(pa.table({"abstract": ["text"]}), path)
            with self.assertRaisesRegex(ValueError, "Missing article"):
                ParquetSplit([path])


if __name__ == "__main__":
    unittest.main()
