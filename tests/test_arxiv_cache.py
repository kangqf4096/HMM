import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_hmm.arxiv_cache import inspect_parquet


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




if __name__ == "__main__":
    unittest.main()
