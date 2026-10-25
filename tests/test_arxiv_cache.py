import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_hmm.arxiv_cache import inspect_parquet, repair_arxiv
from opt_hmm.data import DATASET, REVISION, SUBSET
from opt_hmm.io import read_json


class ArxivCacheTests(unittest.TestCase):
    def folder(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="ac")

    def arrow_modules(self):
        arrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        metadata = SimpleNamespace(num_rows=5, num_row_groups=1,
            schema=SimpleNamespace(to_arrow_schema=lambda: SimpleNamespace(names=["article", "abstract"])))
        parquet.read_metadata = Mock(return_value=metadata)
        arrow.parquet = parquet
        return {"pyarrow": arrow, "pyarrow.parquet": parquet}

    def valid_bytes(self):
        # The boundary layout is real; metadata parsing is replaced in these CPU tests.
        return b"PAR1" + b"body" + (0).to_bytes(4, "little") + b"PAR1"

    def config(self):
        return {"dataset": DATASET, "subset": SUBSET, "dataset_revision": REVISION}

    def test_reports_truncated_html_lfs_and_mismatched_files_without_hashing(self):
        with self.folder() as folder:
            path = Path(folder) / "a.parquet"
            for value, reason in ((b"PAR1", "too short"),
                                  (b"<html>failure response</html>", "non-Parquet"),
                                  (b"version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
                                  (b"PAR1" + b"data" * 5 + b"FAIL", "footer")):
                path.write_bytes(value)
                result = inspect_parquet(path)
                self.assertFalse(result["ok"])
                self.assertIn(reason, result["reason"])
            path.write_bytes(self.valid_bytes())
            self.assertIn("size differs", inspect_parquet(path, 999)["reason"])
            with patch.dict(sys.modules, self.arrow_modules()):
                self.assertEqual(inspect_parquet(path)["rows"], 5)

    def test_refreshes_bad_shard_and_downloads_missing_shard_reuses_good_shard(self):
        with self.folder() as folder:
            root = Path(folder)
            paths = {split: root / f"{split}.parquet" for split in ("train", "validation", "test")}
            paths["train"].write_bytes(b"PAR1" + b"bad footer data")
            paths["validation"].write_bytes(self.valid_bytes())
            model = root / "model.bin"
            model.write_bytes(b"model stays")
            items = [SimpleNamespace(path=f"document/{split}-00000-of-00001.parquet", size=len(self.valid_bytes()))
                     for split in paths]
            hub = types.ModuleType("huggingface_hub")
            hub.HfApi = Mock()
            hub.HfApi.return_value.list_repo_tree.return_value = items

            def split_of(filename):
                return filename.split("/")[-1].split("-")[0]

            hub.try_to_load_from_cache = lambda **kwargs: str(paths[split_of(kwargs["filename"])])

            def download(**kwargs):
                path = paths[split_of(kwargs["filename"])]
                path.write_bytes(self.valid_bytes())
                return str(path)

            hub.hf_hub_download = Mock(side_effect=download)
            with patch.dict(sys.modules, {"huggingface_hub": hub, **self.arrow_modules()}), \
                    redirect_stdout(io.StringIO()):
                report = repair_arxiv(self.config(), root / "report.json")
            calls = {split_of(call.kwargs["filename"]): call.kwargs for call in hub.hf_hub_download.call_args_list}
            self.assertEqual(set(calls), {"train", "test"})
            self.assertTrue(calls["train"]["force_download"])
            self.assertFalse(calls["test"]["force_download"])
            self.assertEqual(report["rows_by_split"], {"train": 5, "validation": 5, "test": 5})
            self.assertEqual(report["refreshed"], 2)
            self.assertEqual(model.read_bytes(), b"model stays")
            self.assertEqual(read_json(root / "report.json")["refreshed"], 2)

    def test_failed_refresh_preserves_file_diagnostic_report(self):
        with self.folder() as folder:
            root = Path(folder)
            path = root / "bad.parquet"
            path.write_bytes(b"not parquet content")
            hub = types.ModuleType("huggingface_hub")
            hub.HfApi = Mock()
            hub.HfApi.return_value.list_repo_tree.return_value = [
                SimpleNamespace(path=f"document/{split}-00000-of-00001.parquet", size=100)
                for split in ("train", "validation", "test")]
            hub.try_to_load_from_cache = Mock(return_value=str(path))
            hub.hf_hub_download = Mock(return_value=str(path))
            with patch.dict(sys.modules, {"huggingface_hub": hub}), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "still unreadable"):
                    repair_arxiv(self.config(), root / "report.json")
            self.assertFalse(read_json(root / "report.json")["files"][0]["after"]["ok"])


if __name__ == "__main__":
    unittest.main()
