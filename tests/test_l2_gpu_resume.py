"""Stage recovery with interrupted commands and legacy L2 output layouts; no GPU."""
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import l2_gpu
from opt_hmm.l2.stages import stage_table
from opt_hmm.progress import Overall


class StageResumeTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="rs-", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.source = {"repo_id": "Qwen/test", "revision": "fixed"}
        self.write(self.model / "opt_hmm_source.json", self.source)
        self.config = json.loads((ROOT / "configs/text_l2_smoke.json").read_text(encoding="utf-8"))
        self.config["dtype"] = "bfloat16"
        self.cfg = self.root / "config.json"
        self.write(self.cfg, self.config)
        self.progress = self.root / "progress.json"
        self.calls, self.fail = [], None
        self.patch_root = patch.object(l2_gpu, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def runner(self, command, **kwargs):
        args = command[2:]
        def option(key):
            return args[args.index(key) + 1]
        kind, output = args[0], Path(option("--output"))
        side = option("--side") if kind == "l2-generate" else ("train" if output.name.endswith("train-pos") else "test")
        name = kind if kind == "prepare" else f"{kind}-{side}"
        self.calls.append(name)
        output.mkdir(parents=True)
        if self.fail == name:
            (output / "partial").write_text("keep this", encoding="utf-8")
            raise subprocess.CalledProcessError(9, command)
        saved = {"config": copy.deepcopy(self.config)}
        if kind == "prepare":
            saved.update(model=str(self.model), model_source=self.source)
            files, metadata = ["prompts.jsonl"], "manifest.json"
        elif kind == "l2-generate":
            saved.update(side=side, model=str(self.model), dtype=option("--dtype"), tensor_parallel=1,
                         prompt_manifest={"model_source": self.source})
            files, metadata = ["sources.jsonl"], "metadata.json"
        else:
            saved.update(side=side, sources=option("--input"))
            if side == "test":
                self.assertEqual(Path(option("--train")), self.root / "runs/x-train-pos")
            files, metadata = ["positions.jsonl", "articles.jsonl"], "manifest.json"
        for file in files:
            self.write(output / file, {"result": name})
        self.write(output / metadata, saved)
        return subprocess.CompletedProcess(command, 0)

    def run_pipeline(self, dtype=None):
        with patch.object(l2_gpu.subprocess, "run", side_effect=self.runner), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            l2_gpu.run_pipeline(self.model, "x", self.cfg, self.progress, dtype)

    def test_fresh_run_then_all_legacy_outputs_are_skipped(self):
        self.run_pipeline()
        self.assertEqual(self.calls, ["prepare", "l2-generate-train", "l2-build-train", "l2-generate-test", "l2-build-test"])
        self.calls.clear()
        self.run_pipeline()
        self.assertEqual(self.calls, [])
        saved = l2_gpu.read_json(self.progress)
        for stage in ("G1", "B1", "G2", "B2"):
            self.assertEqual(saved["stages"][stage]["done"], saved["stages"][stage]["total"])

    def test_interrupted_train_build_reuses_generation_and_preserves_partial_output(self):
        self.fail = "l2-build-train"
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline()
        self.fail, self.calls = None, []
        self.run_pipeline()
        self.assertEqual(self.calls, ["l2-build-train", "l2-generate-test", "l2-build-test"])
        self.assertEqual((self.root / "runs/x-train-pos.old1/partial").read_text(), "keep this")

    def test_interrupted_test_generation_skips_both_training_steps(self):
        self.fail = "l2-generate-test"
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline()
        self.fail, self.calls = None, []
        self.run_pipeline()
        self.assertEqual(self.calls, ["l2-generate-test", "l2-build-test"])
        self.assertTrue((self.root / "runs/x-test-gen.old1/partial").is_file())

    def test_truncated_manifest_and_multiple_retries_preserve_each_attempt(self):
        self.run_pipeline()
        output = self.root / "runs/x-test-pos"
        (output / "manifest.json").write_text('{"config":', encoding="utf-8")
        self.fail, self.calls = "l2-build-test", []
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline()
        self.fail, self.calls = None, []
        self.run_pipeline()
        self.assertEqual(self.calls, ["l2-build-test"])
        self.assertTrue((self.root / "runs/x-test-pos.old1/manifest.json").is_file())
        self.assertTrue((self.root / "runs/x-test-pos.old2/partial").is_file())

    def test_changed_generation_dtype_is_rejected_without_moving_valid_results(self):
        self.run_pipeline()
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "new run-name"):
            self.run_pipeline("float16")
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "runs/x-train-gen.old1").exists())

    def test_changed_experiment_or_model_source_is_rejected(self):
        self.run_pipeline()
        original = copy.deepcopy(self.config)
        self.config["seed"] += 1
        self.write(self.cfg, self.config)
        with self.assertRaisesRegex(ValueError, "new run-name"):
            self.run_pipeline()
        self.config = original
        self.write(self.cfg, self.config)
        self.write(self.model / "opt_hmm_source.json", {"repo_id": "another/model", "revision": "other"})
        with self.assertRaisesRegex(ValueError, "new run-name"):
            self.run_pipeline()

    def test_same_model_can_move_and_local_cache_settings_can_change(self):
        self.run_pipeline()
        moved = self.root / "moved"
        self.model.rename(moved)
        self.model = moved
        self.config.update(model=str(moved), generation_storage={"cache_root": "another-cache"})
        self.write(self.cfg, self.config)
        self.calls.clear()
        self.run_pipeline()
        self.assertEqual(self.calls, [])

    def test_skipping_preserves_recorded_progress_timing(self):
        self.run_pipeline()
        overall = Overall(stage_table(self.config), path=str(self.progress), stream=io.StringIO())
        for stage in ("G1", "B1", "G2", "B2"):
            overall.done[stage] = overall.stages[stage]["total"]
            overall.seconds[stage] = 123.
        overall.emit()
        self.run_pipeline()
        saved = l2_gpu.read_json(self.progress)
        self.assertEqual(saved["stages"]["G1"]["seconds"], 123.)
        self.assertEqual(saved["stages"]["G1"]["cached"], 0)

    def test_restarted_upstream_rebuilds_dependent_positions(self):
        self.run_pipeline()
        (self.root / "runs/x-train-gen/metadata.json").unlink()
        self.calls.clear()
        self.run_pipeline()
        self.assertEqual(self.calls, ["l2-generate-train", "l2-build-train", "l2-build-test"])
        for name in ("x-train-gen", "x-train-pos", "x-test-pos"):
            self.assertTrue((self.root / "runs" / f"{name}.old1").is_dir())

    def test_missing_result_file_restarts_the_build(self):
        self.run_pipeline()
        (self.root / "runs/x-test-pos/articles.jsonl").unlink()
        self.calls.clear()
        self.run_pipeline()
        self.assertEqual(self.calls, ["l2-build-test"])

    def test_run_name_cannot_move_output_outside_the_project(self):
        for name in ("../x", "/x", "..", "a/b", "a\\b"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "single directory"):
                l2_gpu.run_pipeline(self.model, name, self.cfg, self.progress)


if __name__ == "__main__":
    unittest.main()
