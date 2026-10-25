import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

from opt_hmm.arxiv_files import cached_parquet_source, ensure_arxiv_files
from opt_hmm.arxiv_cache import inspect_parquet


@unittest.skipUnless(pa is not None, "ordinary-file staging tests require pyarrow")
class ArxivFilesTests(unittest.TestCase):
    def folder(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="af")

    def fixture(self, root):
        project, runtime = root / "drive", root / "runtime"
        repo_cache = project / ".cache/hf/hub/datasets--ccdv--arxiv-summarization"
        paths, payloads, items = {}, {}, []
        for ordinal, split in enumerate(("train", "validation", "test")):
            name = f"document/{split}-00000-of-00001.parquet"
            identifier = str(ordinal + 1) * 64
            blob = repo_cache / "blobs" / identifier[:2] / identifier
            blob.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"article": [f"body {split}"], "abstract": ["summary"]}), blob)
            snapshot = repo_cache / "snapshots/pinned" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text(f"../../blobs/{identifier[:2]}/{identifier}", encoding="ascii")
            paths[name], payloads[name] = snapshot, blob.read_bytes()
            items.append(SimpleNamespace(path=name, size=blob.stat().st_size))
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = Mock()
        hub.HfApi.return_value.list_repo_tree.return_value = [*items, SimpleNamespace(path="section/train.parquet")]
        hub.try_to_load_from_cache = Mock(side_effect=lambda **kw: str(paths[kw["filename"]]))
        hub.hf_hub_download = Mock(side_effect=AssertionError("existing real blobs must be reused"))
        return project, runtime, repo_cache, paths, payloads, hub

    def test_recovers_drive_text_links_creates_regular_files_and_never_redownloads(self):
        with self.folder() as folder:
            project, runtime, repo_cache, paths, payloads, hub = self.fixture(Path(folder))
            original = {name: path.read_bytes() for name, path in paths.items()}
            with patch.dict(sys.modules, {"huggingface_hub": hub}), redirect_stdout(io.StringIO()):
                staged, report = ensure_arxiv_files("ccdv/arxiv-summarization", "document", "pinned",
                                                   project=project, runtime=runtime)
                # A second invocation uses ordinary files and does not touch old Hub links.
                hub.try_to_load_from_cache.side_effect = AssertionError("local files must be reused")
                _, second = ensure_arxiv_files("ccdv/arxiv-summarization", "document", "pinned",
                                               project=project, runtime=runtime)
            hub.hf_hub_download.assert_not_called()
            self.assertEqual(report["refreshed"], 0)
            self.assertEqual({row["action"] for row in second["files"]}, {"reused_local"})
            for row in report["files"]:
                for key in ("path", "persistent_path"):
                    path = Path(row[key])
                    self.assertFalse(path.is_symlink())
                    self.assertEqual(path.read_bytes(), payloads[row["file"]])
                    self.assertTrue(inspect_parquet(path)["ok"])
                self.assertEqual(paths[row["file"]].read_bytes(), original[row["file"]])
            self.assertEqual({split: len(files) for split, files in staged.items()},
                             {"train": 1, "validation": 1, "test": 1})

    def test_missing_real_blob_downloads_to_local_dir_then_persists_plain_files(self):
        with self.folder() as folder:
            project, runtime, repo_cache, paths, payloads, hub = self.fixture(Path(folder))
            hub.try_to_load_from_cache = Mock(return_value=None)

            def download(**kwargs):
                path = Path(kwargs["local_dir"]) / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payloads[kwargs["filename"]])
                return str(path)

            hub.hf_hub_download = Mock(side_effect=download)
            with patch.dict(sys.modules, {"huggingface_hub": hub}), redirect_stdout(io.StringIO()):
                _, report = ensure_arxiv_files("ccdv/arxiv-summarization", "document", "pinned",
                                               project=project, runtime=runtime)
            self.assertEqual(report["refreshed"], 3)
            for call in hub.hf_hub_download.call_args_list:
                self.assertTrue(Path(call.kwargs["local_dir"]).is_relative_to(runtime))
                self.assertNotIn("cache_dir", call.kwargs)
            for row in report["files"]:
                self.assertTrue(Path(row["persistent_path"]).is_file())

    def test_missing_runtime_copy_restores_from_persistent_regular_files(self):
        with self.folder() as folder:
            project, runtime, repo_cache, paths, payloads, hub = self.fixture(Path(folder))
            for name, payload in payloads.items():
                path = project / "data/arxiv/pinned" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            hub.try_to_load_from_cache = Mock(side_effect=AssertionError("persistent file should be used"))
            with patch.dict(sys.modules, {"huggingface_hub": hub}), redirect_stdout(io.StringIO()):
                _, report = ensure_arxiv_files("ccdv/arxiv-summarization", "document", "pinned",
                                               project=project, runtime=runtime)
            hub.hf_hub_download.assert_not_called()
            self.assertEqual(report["refreshed"], 0)

    def test_rejected_cached_placeholder_is_not_copied_back_by_downloader(self):
        for destination_exists in (False, True):
            with self.subTest(destination_exists=destination_exists), self.folder() as folder:
                project, _, repo_cache, paths, payloads, hub = self.fixture(Path(folder))
                # Only the 79-byte snapshot placeholders remain in the Hub cache.
                for blob in (repo_cache / "blobs").rglob("*"):
                    if blob.is_file():
                        blob.unlink()
                archive = project / "data/arxiv/pinned"
                ready_name = "document/test-00000-of-00001.parquet"
                ready_path = archive / ready_name
                ready_path.parent.mkdir(parents=True, exist_ok=True)
                ready_path.write_bytes(payloads[ready_name])
                if destination_exists:
                    for name, snapshot in paths.items():
                        if name != ready_name:
                            (archive / name).write_bytes(snapshot.read_bytes())

                def download(**kwargs):
                    path = Path(kwargs["local_dir"]) / kwargs["filename"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Hub local_dir mode reuses the rejected snapshot unless forced.
                    payload = (payloads[kwargs["filename"]] if kwargs["force_download"]
                               else paths[kwargs["filename"]].read_bytes())
                    path.write_bytes(payload)
                    return str(path)

                hub.hf_hub_download = Mock(side_effect=download)
                with patch.dict(sys.modules, {"huggingface_hub": hub}), redirect_stdout(io.StringIO()):
                    _, report = ensure_arxiv_files("ccdv/arxiv-summarization", "document", "pinned",
                                                   project=project)
                self.assertEqual(report["refreshed"], 2)
                self.assertEqual(hub.hf_hub_download.call_count, 2)
                self.assertTrue(all(call.kwargs["force_download"]
                                    for call in hub.hf_hub_download.call_args_list))
                for row in report["files"]:
                    self.assertEqual(Path(row["path"]).read_bytes(), payloads[row["file"]])
                    self.assertTrue(row["after"]["ok"])
                self.assertEqual(ready_path.read_bytes(), payloads[ready_name])

    def test_default_reads_persistent_files_even_with_runtime_environment_and_other_cwd(self):
        with self.folder() as folder:
            project, runtime, repo_cache, paths, payloads, hub = self.fixture(Path(folder))
            for name, payload in payloads.items():
                path = project / "data/arxiv/pinned" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            hub.try_to_load_from_cache.side_effect = AssertionError("persistent file must be used")
            with chdir(ROOT.parent), patch.dict(sys.modules, {"huggingface_hub": hub}), \
                    patch.dict(os.environ, {"OPT_HMM_PROJECT": str(project), "OPT_HMM_RUNTIME": str(runtime)}), \
                    redirect_stdout(io.StringIO()):
                loaded, report = ensure_arxiv_files("ccdv/arxiv-summarization", "document", "pinned")
            self.assertEqual(report["work"], report["archive"])
            self.assertTrue(all(path.is_relative_to(project) for split in loaded.values() for path in split))
            self.assertFalse((runtime / ".cache/arxiv").exists())
            hub.hf_hub_download.assert_not_called()

    def test_cyclic_text_links_stop_with_diagnostic_attempts(self):
        with self.folder() as folder:
            root = Path(folder)
            path = root / "blobs/aa"
            path.parent.mkdir()
            path.write_text("../../blobs/aa", encoding="ascii")
            source, attempts = cached_parquet_source(path, root, None)
            self.assertIsNone(source)
            self.assertLessEqual(len(attempts), 16)
            self.assertTrue(all(not row["ok"] for row in attempts))


if __name__ == "__main__":
    unittest.main()
