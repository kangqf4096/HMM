import copy
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_hmm.data import DATASET, REVISION, ROLES, SUBSET, prepare, select_articles
from opt_hmm.generate import generate
from opt_hmm.io import read_json, read_jsonl, write_json
from opt_hmm.paths import project_path


class WordTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, *, add_special_tokens, truncation, max_length):
        self.calls.append(text)
        # Distinct paper prefixes make accidental abstract use observable.
        tokens = [int(word) for word in text.split()]
        return tokens[:max_length]


def paper(index, length=150):
    return {"article": " ".join([str(index)] * length), "abstract": "not the body"}


class DataTests(unittest.TestCase):
    def config(self):
        config = read_json(ROOT / "configs" / "text_smoke.json")
        config["roles"] = dict.fromkeys(ROLES, 1)
        return config

    def raw(self, count=30):
        return {"train": [paper(i) for i in range(count)],
                "validation": [], "test": []}

    def test_paper_bodies_have_disjoint_roles_and_separate_edit_pool(self):
        raw = {"train": [paper(i) for i in range(2)],
               "validation": [paper(i) for i in range(2, 4)],
               "test": [paper(i) for i in range(4, 6)]}
        records, stats = select_articles(raw, WordTokenizer(), self.config())
        self.assertEqual(Counter(r["role"] for r in records), dict.fromkeys(ROLES, 1))
        self.assertEqual(len({r["group"] for r in records}), 6)
        self.assertEqual({r["source_split"] for r in records}, set(raw))
        for record in records:
            index = int(raw[record["source_split"]][record["source_row"]]["article"].split()[0])
            self.assertEqual(record["prompt_ids"], [index] * 50)
            self.assertEqual("pool_ids" in record, record["role"] == "pool")
            if record["role"] == "pool":
                self.assertEqual(record["pool_ids"], [index] * 100)
        self.assertEqual(stats["selected_articles"], 6)

    def test_short_empty_and_duplicate_bodies_cannot_fill_roles(self):
        raw = self.raw(5)
        duplicate = copy.deepcopy(raw["train"][0])
        duplicate["article"] = duplicate["article"].replace(" ", "\n  ")
        raw["validation"] = [duplicate, paper(90, 3), {"article": "", "abstract": "long"},
                             {"article": None, "abstract": "long"}]
        raw["test"] = [copy.deepcopy(raw["train"][1])]
        with self.assertRaisesRegex(ValueError, "only 5 qualify"):
            select_articles(raw, WordTokenizer(), self.config())
        raw["test"].append(paper(5))
        records, _ = select_articles(raw, WordTokenizer(), self.config())
        self.assertEqual(len({tuple(r["prompt_ids"]) for r in records}), 6)

    def test_sampling_is_reproducible_and_stops_without_tokenizing_corpus(self):
        raw = self.raw(100)
        raw["train"] = [paper(i, 3000) for i in range(100)]
        tokenizer = WordTokenizer()
        first, stats = select_articles(raw, tokenizer, self.config())
        second, _ = select_articles(raw, WordTokenizer(), self.config())
        self.assertEqual(first, second)
        self.assertEqual(stats, {"source_articles": 100, "examined_articles": 6, "selected_articles": 6})
        self.assertEqual(len(tokenizer.calls), 6)
        pool = next(r for r in first if r["role"] == "pool")
        self.assertEqual(len(pool["pool_ids"]), 2048)
        config = self.config()
        config["seed"] += 1
        third, _ = select_articles(raw, WordTokenizer(), config)
        self.assertNotEqual([r["group"] for r in first], [r["group"] for r in third])

    def test_invalid_role_counts_and_insufficient_capacity_fail(self):
        for value in (0, -1, 1.5, True):
            config = self.config()
            config["roles"]["test"] = value
            with self.assertRaisesRegex(ValueError, "positive integer counts"):
                select_articles(self.raw(), WordTokenizer(), config)
        with self.assertRaisesRegex(ValueError, "only 5 source rows"):
            select_articles(self.raw(5), WordTokenizer(), self.config())

    def test_prepare_loads_pinned_document_release_and_saves_source(self):
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = Mock(return_value=self.raw())
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = WordTokenizer()
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="ax") as folder:
            model = Path(folder) / "model"
            model.mkdir()
            output = Path(folder) / "out"
            with patch.dict(sys.modules, {"datasets": datasets, "transformers": transformers}):
                stats = prepare(self.config(), model, output)
            datasets.load_dataset.assert_called_once_with(
                DATASET, SUBSET, revision=REVISION, cache_dir=str(project_path(".cache/hf/datasets")))
            manifest = read_json(output / "manifest.json")
            self.assertEqual(manifest["dataset"], DATASET)
            self.assertEqual(manifest["dataset_revision"], REVISION)
            self.assertEqual(manifest["text_field"], "article")
            self.assertEqual(manifest["selected_articles"], stats["selected_articles"])
            self.assertEqual(len(read_jsonl(output / "prompts.jsonl")), 6)

    def test_direct_reader_does_not_invoke_datasets_generation(self):
        config = self.config()
        config["dataset_loader"] = "parquet_file"
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = Mock(side_effect=AssertionError("fragment path must not run"))
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = WordTokenizer()
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="ax") as folder:
            model, output = Path(folder) / "model", Path(folder) / "out"
            model.mkdir()
            with patch.dict(sys.modules, {"datasets": datasets, "transformers": transformers}), \
                    patch("opt_hmm.arxiv_parquet.load_arxiv_parquet", return_value=self.raw()) as direct:
                prepare(config, model, output)
            datasets.load_dataset.assert_not_called()
            direct.assert_called_once_with(DATASET, SUBSET, REVISION)
            self.assertEqual(read_json(output / "manifest.json")["dataset_loader"], "parquet_file")

    def test_generation_rejects_stale_corpus_before_gpu_startup(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="ax") as folder:
            model = Path(folder) / "model"
            model.mkdir()
            for name, value in (("dataset", "other/corpus"),
                                ("subset", "section"), ("dataset_revision", "older")):
                config = self.config()
                stale = {"config": config, "model": str(model),
                         "dataset": DATASET, "subset": SUBSET, "dataset_revision": REVISION}
                stale[name] = value
                write_json(Path(folder) / "manifest.json", stale)
                with patch("opt_hmm.generate.check_runtime") as runtime:
                    with self.assertRaisesRegex(ValueError, f"differ on {name}"):
                        generate(config, model, folder, Path(folder) / "out")
                    runtime.assert_not_called()
            # A manifest without explicit source metadata must be prepared again.
            config = self.config()
            for name in ("dataset", "subset", "dataset_revision"):
                config.pop(name)
            write_json(Path(folder) / "manifest.json", {"config": config, "model": str(model)})
            with patch("opt_hmm.generate.check_runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "differ on dataset"):
                    generate(config, model, folder, Path(folder) / "out")
                runtime.assert_not_called()

    def test_relocated_prompts_require_the_same_recorded_model_release(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="ax") as folder:
            root = Path(folder)
            model = root / "new-model"
            model.mkdir()
            config = self.config()
            source = {"repo_id": "facebook/opt-1.3b", "revision": "pinned-release"}
            manifest = {"model": str(root / "old-model"), "model_source": source, "config": config,
                        "dataset": DATASET, "subset": SUBSET, "dataset_revision": REVISION}
            write_json(root / "manifest.json", manifest)
            write_json(model / "opt_hmm_source.json", source)
            with patch("opt_hmm.generate.check_runtime", side_effect=RuntimeError("GPU reached")) as runtime:
                with self.assertRaisesRegex(RuntimeError, "GPU reached"):
                    generate(config, model, root, root / "out")
                runtime.assert_called_once()
            for replacement in ({"repo_id": "facebook/opt-1.3b", "revision": "another-release"}, {}):
                write_json(model / "opt_hmm_source.json", replacement)
                with patch("opt_hmm.generate.check_runtime") as runtime:
                    with self.assertRaisesRegex(ValueError, "different model"):
                        generate(config, model, root, root / "out")
                    runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
