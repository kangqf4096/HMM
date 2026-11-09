"""CPU output recovery must preserve GPU inputs and reuse old/new SeedBS caches."""
import contextlib
import copy
import io
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import l2_cpu
from opt_hmm.io import new_output


class CpuResumeTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="cpu-", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "configs/text_l2_smoke.json").read_text(encoding="utf-8"))
        self.progress, self.weights = self.root / "bar.json", self.root / "weights.json"
        self.manifests = {}
        for side in ("train", "test"):
            directory = self.root / "runs" / f"x-{side}-pos"
            manifest = {"config": self.config, "side": side, "key_id": self.config["l2"]["sides"][side]["key_id"],
                        "articles": 2, "positions": 20, "l_top": {"L_top": 12}}
            self.write(directory / "manifest.json", manifest)
            for file in ("positions.jsonl", "articles.jsonl"):
                self.write(directory / file, {"input": side})
            self.manifests[side] = copy.deepcopy(manifest)
        self.calls, self.fail, self.caches = [], None, {}
        root_patch = patch.object(l2_cpu, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def frozen(self):
        return {"config": {k: self.config[k] for k in ("tokens", "context_width", "bins", "alpha", "seed")},
                "l2": self.config["l2"], "train_key_id": self.manifests["train"]["key_id"],
                "l_top": self.manifests["train"]["l_top"],
                "methods": {"pd_m100": {"cap": 10}, "pe_m100": {"cap": 25}},
                "seedbs": {"seedbs": {"B": 6}}}

    def runner(self, command, **kwargs):
        args = command[2:]
        def option(key):
            return Path(args[args.index(key) + 1])
        name = args[0].removeprefix("l2-")
        self.calls.append(name)
        output = option("--output")
        if name == "weights":
            self.write(output, {"written": True})
            return subprocess.CompletedProcess(command, 0)
        # Exercise the same empty-output rule as the actual commands.
        new_output(output)
        if name in ("fit", "test"):
            cache = option("--cache")
            cache.mkdir(parents=True, exist_ok=True)
            self.caches[name] = cache
            with (cache / "B6_fixture.pkl").open("wb") as handle:
                pickle.dump({"cached": True}, handle)
        if self.fail == name:
            self.write(output / "partial.json", {"keep": True})
            raise subprocess.CalledProcessError(9, command)
        for file in l2_cpu.FILES[name]:
            self.write(output / file, {"complete": name})
        if name == "mechanism":
            self.write(output / "mechanism.json", {"groups": {}, "M1": {}, "M2": {}})
        elif name == "fit":
            self.write(output / "frozen.json", self.frozen())
        elif name == "test":
            self.write(output / "report.json", {"methods": ["pd_m100", "pe_m100"]})
            (output / "metrics.jsonl").write_text('{}\n' * 4, encoding="utf-8")
            self.assertTrue((option("--frozen") / "frozen.json").is_file())
        elif name == "cap":
            frozen = l2_cpu.read_json(option("--frozen") / "frozen.json")
            self.write(output / "cap_sweep.json", {"rows": [{}] * (2 * len(self.config["l2"]["cap_sweep"])),
                       "frozen_cap": {name: value["cap"] for name, value in frozen["methods"].items()}})
        return subprocess.CompletedProcess(command, 0)

    def run_pipeline(self, workers=10):
        with patch.object(l2_cpu.subprocess, "run", side_effect=self.runner), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            l2_cpu.run_pipeline("x", workers, self.progress, self.weights)

    def test_first_run_then_rerun_keeps_all_inputs_and_completed_results(self):
        inputs = {p: p.read_bytes() for p in (self.root / "runs").glob("*-pos/*")}
        self.run_pipeline()
        self.assertEqual(self.calls, ["mechanism", "fit", "test", "cap", "weights"])
        self.calls.clear()
        self.run_pipeline(workers=2)
        self.assertEqual(self.calls, ["weights"])
        self.assertEqual(inputs, {p: p.read_bytes() for p in inputs})
        self.assertEqual(self.caches["fit"], self.root / "runs/x-cache/fit-seedbs")
        self.assertEqual(self.caches["test"], self.root / "runs/x-cache/test-seedbs")
        self.assertFalse(list((self.root / "runs").glob("*.old*")))

    def test_fit_interruption_keeps_cache_and_skips_mechanism_on_retry(self):
        self.fail = "fit"
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline()
        cache = self.caches["fit"]
        cached_bytes = (cache / "B6_fixture.pkl").read_bytes()
        self.fail, self.calls = None, []
        self.run_pipeline()
        self.assertEqual(self.calls, ["fit", "test", "cap", "weights"])
        self.assertEqual((cache / "B6_fixture.pkl").read_bytes(), cached_bytes)
        self.assertTrue((self.root / "runs/x-fit.old1/partial.json").is_file())

    def test_legacy_cache_is_imported_before_output_is_archived_and_is_readable(self):
        from opt_hmm.l2.seedbs_run import _job
        output = self.root / "runs/x-fit"
        legacy = output / "cache/seedbs/B6_old.pkl"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(pickle.dumps({"old_result": 123}))
        self.write(output / "coverage.json", {"partial": True})
        self.run_pipeline()
        imported = self.root / "runs/x-cache/fit-seedbs/B6_old.pkl"
        with patch("opt_hmm.l2.seedbs_run.sb.seedbs_precompute", side_effect=AssertionError("recomputed cached work")):
            result, hit = _job(([], 6, 0, {}, str(imported)))
        self.assertTrue(hit)
        self.assertEqual(result, {"old_result": 123})
        self.assertTrue((self.root / "runs/x-fit.old1/cache/seedbs/B6_old.pkl").is_file())

    def test_complete_legacy_outputs_are_adopted_without_recomputation(self):
        self.run_pipeline()
        for name in l2_cpu.FILES:
            (self.root / "runs" / f"x-{name}" / "resume.json").unlink()
        self.calls.clear()
        self.run_pipeline()
        self.assertEqual(self.calls, ["weights"])
        self.assertTrue((self.root / "runs/x-fit/resume.json").is_file())

    def test_frozen_file_alone_is_not_a_completed_fit(self):
        self.run_pipeline()
        (self.root / "runs/x-fit/fit_report.md").unlink()
        self.calls.clear()
        self.run_pipeline()
        self.assertEqual(self.calls, ["fit", "test", "cap", "weights"])
        self.assertTrue((self.root / "runs/x-fit.old1/frozen.json").is_file())

    def test_partial_final_metrics_are_restarted_without_repeating_fit(self):
        self.run_pipeline()
        (self.root / "runs/x-test/metrics.jsonl").write_text('{}\n{}\n{}\n{"partial":', encoding="utf-8")
        self.calls.clear()
        self.run_pipeline()
        self.assertEqual(self.calls, ["test", "weights"])
        self.assertTrue((self.root / "runs/x-test.old1/metrics.jsonl").is_file())

    def test_changed_input_data_stops_before_overwriting_results(self):
        self.run_pipeline()
        manifest = copy.deepcopy(self.manifests["train"])
        manifest["positions"] += 1
        self.write(self.root / "runs/x-train-pos/manifest.json", manifest)
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "CPU inputs changed"):
            self.run_pipeline()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "runs/x-mechanism.old1").exists())

    def test_changed_frozen_parameters_invalidate_test_and_cap(self):
        self.run_pipeline()
        frozen = l2_cpu.read_json(self.root / "runs/x-fit/frozen.json")
        frozen["methods"]["pd_m100"]["cap"] = 50
        self.write(self.root / "runs/x-fit/frozen.json", frozen)
        self.calls.clear()
        self.run_pipeline()
        self.assertEqual(self.calls, ["test", "cap", "weights"])

    def test_missing_gpu_input_stops_without_creating_cpu_outputs(self):
        (self.root / "runs/x-test-pos/articles.jsonl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "GPU input"):
            self.run_pipeline()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "runs/x-mechanism").exists())

    def test_mismatched_training_test_config_is_rejected(self):
        manifest = copy.deepcopy(self.manifests["test"])
        manifest["config"]["tokens"] += 1
        self.write(self.root / "runs/x-test-pos/manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "different configurations"):
            self.run_pipeline()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
