import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clean_wiki import clean_wiki


class CleanupTests(unittest.TestCase):
    def make_file(self, root, relative, value="data"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    def test_cleanup_removes_identified_data_preserves_other_sources_and_is_repeatable(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="wc") as folder:
            root = Path(folder).resolve()
            removed_files = [
                ".cache/hf/datasets/Salesforce___wikitext/raw/data.arrow",
                ".cache/hf/datasets/_wikitext_raw.lock",
                ".cache/hf/hub/datasets--Salesforce--wikitext/blobs/p0",
                ".cache/hf/hub/.locks/datasets--Salesforce--wikitext/p0.lock",
                ".cache/hf/datasets/downloads/a0",
                ".cache/hf/datasets/downloads/a0.lock",
                ".cache/hf/datasets/downloads/a0.incomplete",
                ".cache/hf/datasets/downloads/extracted/e0/wikitext-2-raw/wiki.train.raw",
                "data/text0/prompts.jsonl",
            ]
            preserved_files = [
                ".cache/hf/datasets/ccdv___arxiv-summarization/data.arrow",
                ".cache/hf/hub/datasets--ccdv--arxiv-summarization/blobs/p1",
                ".cache/hf/hub/models--example--wikitext/model.safetensors",
                ".cache/hf/datasets/downloads/a1",
                "data/arxiv0/prompts.jsonl",
                "runs/text0-gen/observations.jsonl",
            ]
            for relative in removed_files + preserved_files:
                self.make_file(root, relative)
            wiki_metadata = self.make_file(root, ".cache/hf/datasets/downloads/a0.json",
                json.dumps({"url": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/data.parquet"}))
            other_metadata = self.make_file(root, ".cache/hf/datasets/downloads/a1.json",
                json.dumps({"url": "https://huggingface.co/datasets/ccdv/arxiv-summarization/resolve/main/data.parquet"}))
            self.make_file(root, "data/text0/manifest.json",
                           json.dumps({"config": {"dataset": "Salesforce/wikitext"}}))
            self.make_file(root, "data/arxiv0/manifest.json",
                           json.dumps({"dataset": "ccdv/arxiv-summarization"}))
            self.assertTrue(clean_wiki(root))
            for relative in removed_files:
                self.assertFalse((root / relative).exists(), relative)
            self.assertFalse(wiki_metadata.exists())
            for relative in preserved_files:
                self.assertEqual((root / relative).read_text(encoding="utf-8"), "data", relative)
            self.assertTrue(other_metadata.exists())
            self.assertEqual(clean_wiki(root), [])

    def test_outside_project_target_rejected_before_any_deletion(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="wc") as folder:
            root = Path(folder).resolve()
            first = self.make_file(root, ".cache/hf/datasets/Salesforce___wikitext/data.arrow")
            other = self.make_file(root, ".cache/hf/datasets/wikitext/data.arrow")
            resolve = Path.resolve

            def resolved(path, *args, **kwargs):
                if path == other.parent:
                    return root.parent / "external"
                return resolve(path, *args, **kwargs)

            with patch.object(Path, "resolve", resolved):
                with self.assertRaisesRegex(ValueError, "outside the project"):
                    clean_wiki(root)
            self.assertTrue(first.exists())
            self.assertTrue(other.exists())


if __name__ == "__main__":
    unittest.main()
