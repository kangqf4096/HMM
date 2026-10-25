import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

from opt_hmm.arxiv_check import check_arxiv, inspect_contents, shard_manifest
from opt_hmm.data import DATASET, REVISION


@unittest.skipUnless(pa is not None, "dataset integrity tests require pyarrow")
class ArxivCheckTests(unittest.TestCase):
    def folder(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="ck")

    def write_parquet(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"article": ["some body " * 100, "another body " * 100],
                                 "abstract": ["summary", "summary"]}), path, use_dictionary=False)
        return path.stat().st_size

    def config(self):
        return {"dataset": DATASET, "subset": "document", "dataset_revision": "fixture"}

    def test_offline_manifest_has_all_17_pinned_shards(self):
        manifest = shard_manifest(REVISION, offline=True)
        self.assertEqual(len(manifest), 17)
        self.assertEqual(sum(row["file"].startswith("document/train-") for row in manifest), 15)
        self.assertTrue(all(row["size"] is None for row in manifest))

    def test_classifies_all_files_finds_blob_and_never_modifies_dataset(self):
        with self.folder() as folder:
            project = Path(folder)
            archive = project / "data/arxiv/fixture"
            repo = project / ".cache/hf/hub/datasets--ccdv--arxiv-summarization"
            names = {split: f"document/{split}-00000-of-00001.parquet"
                     for split in ("train", "validation", "test")}
            train_size = self.write_parquet(archive / names["train"])
            identifier = "b" * 64
            blob = repo / "blobs/bb" / identifier
            validation_size = self.write_parquet(blob)
            placeholder = archive / names["validation"]
            placeholder.write_text(f"../../blobs/bb/{identifier}", encoding="ascii")
            manifest = [{"file": names[split], "size": train_size if split != "validation" else validation_size}
                        for split in ("train", "validation", "test")]
            before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
            with patch("opt_hmm.arxiv_check.shard_manifest", return_value=manifest), \
                    patch.dict(os.environ, {"OPT_HMM_RUNTIME": str(project / "runtime")}), \
                    redirect_stdout(io.StringIO()):
                report = check_arxiv(self.config(), project=project)
            self.assertEqual([row["status"] for row in report["files"]], ["ready", "recoverable", "missing"])
            self.assertEqual(report["summary"]["recoverable_files"], [names["validation"]])
            self.assertEqual(report["summary"]["no_readable_source"], [names["test"]])
            self.assertFalse(report["summary"]["archive_ready"])
            source = report["files"][1]["readable_source"]
            self.assertEqual(source["path"], str(blob))
            self.assertEqual(source["decoded_rows"], 2)
            self.assertIn("blobs/bb/", report["files"][1]["primary"]["small_file_text"])
            self.assertEqual({path: path.read_bytes() for path in before}, before)
            new_files = {path for path in project.rglob("*") if path.is_file()} - set(before)
            self.assertEqual(new_files, {project / "tmp/arxiv-check.json", project / "tmp/arxiv-check.md"})

    def test_full_read_finds_bad_data_page_even_when_footer_is_readable(self):
        with self.folder() as folder:
            path = Path(folder) / "bad.parquet"
            self.write_parquet(path)
            offset = pq.read_metadata(path).row_group(0).column(0).data_page_offset
            with path.open("r+b") as handle:
                handle.seek(offset)
                handle.write(b"\xff" * 16)
            quick = inspect_contents(path, full=False)
            self.assertTrue(quick["ok"])
            full = inspect_contents(path, full=True)
            self.assertFalse(full["ok"])
            self.assertIn("reason", full)

    def test_all_plain_files_and_correct_rows_mark_archive_ready(self):
        with self.folder() as folder:
            project = Path(folder)
            manifest = []
            for split in ("train", "validation", "test"):
                name = f"document/{split}-00000-of-00001.parquet"
                size = self.write_parquet(project / "data/arxiv/fixture" / name)
                manifest.append({"file": name, "size": size})
            with patch("opt_hmm.arxiv_check.shard_manifest", return_value=manifest), redirect_stdout(io.StringIO()):
                report = check_arxiv(self.config(), project=project)
            self.assertTrue(report["summary"]["archive_ready"])
            self.assertTrue(report["summary"]["all_bytes_available"])
            self.assertEqual(report["summary"]["available_rows_by_split"], {"train": 2, "validation": 2, "test": 2})

    def test_pinned_split_row_count_mismatch_is_reported(self):
        with self.folder() as folder:
            project = Path(folder)
            config = self.config()
            config["dataset_revision"] = REVISION
            manifest = []
            for split in ("train", "validation", "test"):
                name = f"document/{split}-00000-of-00001.parquet"
                size = self.write_parquet(project / "data/arxiv" / REVISION / name)
                manifest.append({"file": name, "size": size})
            with patch("opt_hmm.arxiv_check.shard_manifest", return_value=manifest), redirect_stdout(io.StringIO()):
                report = check_arxiv(config, project=project, full=False)
            self.assertFalse(report["summary"]["row_counts_match"])
            self.assertFalse(report["summary"]["archive_ready"])
            self.assertEqual(report["summary"]["expected_rows_by_split"]["train"], 203037)


if __name__ == "__main__":
    unittest.main()
