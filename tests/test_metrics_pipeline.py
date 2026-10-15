import copy
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_hmm.experiment import run_evaluation
from opt_hmm.io import new_output, read_json, read_jsonl
from opt_hmm.metrics import evaluate, paired_bootstrap, segment_metrics
from opt_hmm.simulate import run


class MetricsTests(unittest.TestCase):
    def test_null_f1_not_rewarded(self):
        result = evaluate([0, 0], [0, 0], [.1, .2], 0)
        self.assertIsNone(result["f1"])
        self.assertIsNone(result["iou"])
        self.assertAlmostEqual(result["brier"], .025)

    def test_empty_prediction_loses_positive_iou(self):
        result = evaluate([0, 1, 1], [0, 0, 0], [0, 0, 0], 0)
        self.assertEqual(result["iou"], 0)
        self.assertEqual(result["f1"], 0)

    def test_half_open_interval_matching(self):
        truth, prediction = np.zeros(30, bool), np.zeros(30, bool)
        truth[10:21], prediction[12:23] = True, True
        metrics = segment_metrics(truth, prediction)
        self.assertEqual(metrics["segment_tp"], 1)
        self.assertEqual(metrics["boundary_mae"], 2.)

    def test_segment_dp_matches_brute_force(self):
        rng = np.random.default_rng(65)
        from opt_hmm.detector import intervals
        for _ in range(20):
            truth, pred = rng.random(12) < .5, rng.random(12) < .5
            a, b = intervals(truth), intervals(pred)
            eligible = []
            for i, left in enumerate(a):
                for j, right in enumerate(b):
                    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
                    union = left[1] - left[0] + right[1] - right[0] - overlap
                    if overlap / union >= .5:
                        eligible.append((i, j))
            maximum = 0
            for count in range(len(eligible) + 1):
                for pairs in itertools.combinations(eligible, count):
                    if len({x[0] for x in pairs}) == len(pairs) == len({x[1] for x in pairs}):
                        maximum = max(maximum, len(pairs))
            self.assertEqual(segment_metrics(truth, pred)["segment_tp"], maximum)

    def test_bootstrap_groups_all_variants(self):
        rows = []
        for group in ("a", "b"):
            for variant in range(3):
                for method, iou in (("independent", .2), ("opt_hmm", .4)):
                    rows.append({"group": group, "method": method, "metrics": {"iou": iou}})
        result = paired_bootstrap(rows, "opt_hmm", draws=100)
        self.assertEqual(result["groups"], 2)
        self.assertAlmostEqual(result["mean"], .2)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def config(self):
        config = read_json(ROOT / "configs" / "sim_smoke.json")
        config.update(tokens=24, test_documents=3, null_cal=20, null_test=8,
                      cal_pivots=500, bins=10, bootstrap=50)
        return config

    def test_full_pipeline_and_group_leak_rejection(self):
        with tempfile.TemporaryDirectory(prefix="t", dir=ROOT / "tmp") as folder:
            output = Path(folder) / "a"
            report = run(self.config(), output)
            self.assertIn("opt_hmm", report["summary"])
            self.assertTrue((output / "predictions.jsonl").is_file())
            thresholds = read_json(output / "thresholds.json")
            self.assertEqual(next(iter(thresholds["opt_hmm"].values()))["count"], 20)
            records = read_jsonl(output / "observations.jsonl")
            leaked = copy.deepcopy(records)
            test = next(r for r in leaked if r["role"] == "test")
            test["group"] = "cal"
            with self.assertRaisesRegex(ValueError, "leaks"):
                run_evaluation(leaked, Path(folder) / "b", self.config())
            with self.assertRaises(FileExistsError):
                new_output(output)

    def test_missing_null_length_rejected(self):
        with tempfile.TemporaryDirectory(prefix="t", dir=ROOT / "tmp") as folder:
            output = Path(folder) / "a"
            run(self.config(), output)
            records = read_jsonl(output / "observations.jsonl")
            test = next(r for r in records if r["role"] == "test")
            for name in ("y", "mask", "z"):
                test[name] = test[name][:-1]
            with self.assertRaisesRegex(ValueError, "No independent null"):
                run_evaluation(records, Path(folder) / "b", self.config())


if __name__ == "__main__":
    unittest.main()
