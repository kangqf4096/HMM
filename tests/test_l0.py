import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from opt_hmm.io import read_json, read_jsonl
from opt_hmm.l0.build import build
from opt_hmm.l0.config import validate_l0
from opt_hmm.l0.decide import build_rules, decide, lookup
from opt_hmm.l0.emission import fit_emission, integral_mass
from opt_hmm.l0.evaluate import evaluate_positions
from opt_hmm.l0.generate import plan_jobs, source_record
from opt_hmm.l0.metrics import paired_difference, position_metrics, segment_report, bootstrap_indices, auc
from opt_hmm.l0.resample import entropy_nats, resample
from opt_hmm.l0.schema import (breakage_mask, build_article, distance_to_edit, scenario_for, truth_mask,
                               validate_article)
from opt_hmm.l0.synth import ToyLM, generate_sources
from opt_hmm.protocol import PUBLIC_KEY, generation_contexts, operational_labels

SMOKE = read_json(root / "configs/text_l0_smoke.json")
FULL = read_json(root / "configs/text_l0_full.json")


def scratch():
    (root / "tmp").mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root / "tmp", prefix="l0-")


class TruthTests(unittest.TestCase):
    def test_breakage_window_covers_the_next_four_seeds_and_is_clipped(self):
        np.testing.assert_array_equal(np.flatnonzero(breakage_mask(20, [5], 4)), [5, 6, 7, 8, 9])
        np.testing.assert_array_equal(np.flatnonzero(breakage_mask(20, [18], 4)), [18, 19])
        np.testing.assert_array_equal(np.flatnonzero(breakage_mask(20, [1], 4)), [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(np.flatnonzero(breakage_mask(20, [2, 4], 4)), np.arange(2, 9))

    def test_truth_is_source_minus_breakage(self):
        truth = truth_mask(20, [[5, 15]], [8], 4)
        np.testing.assert_array_equal(np.flatnonzero(truth), [5, 6, 7, 13, 14])
        self.assertFalse(truth_mask(20, [], [], 4).any())
        self.assertTrue(truth_mask(20, [[0, 20]], [], 4).all())

    def test_distance_to_edit(self):
        self.assertEqual(distance_to_edit(5, []), [None] * 5)
        self.assertEqual(distance_to_edit(6, [1, 4]), [1, 0, 1, 1, 0, 1])

    def test_rows_cover_exactly_n_eff_positions(self):
        source = {"article_id": "a", "split": "test", "scenario": "switch", "ranges": [[10, 30]]}
        tokens = list(range(40))
        rows, article = build_article(source, tokens, [], PUBLIC_KEY, 4)
        self.assertEqual([r["t"] for r in rows], list(range(4, 40)))
        self.assertEqual(article["n_eff"], 36)
        self.assertEqual(sum(r["truth"] for r in rows), 20)
        self.assertAlmostEqual(article["eps_true"], 20 / 36)
        validate_article(rows, article, PUBLIC_KEY, 4)

    def test_validation_rejects_tampered_truth_and_pivots(self):
        source = {"article_id": "a", "split": "test", "scenario": "full_wm", "ranges": [[0, 30]]}
        rows, article = build_article(source, list(range(30)), [], PUBLIC_KEY, 4)
        bad = [dict(r) for r in rows]
        bad[3]["truth"] = False
        with self.assertRaisesRegex(ValueError, "truth"):
            validate_article(bad, article, PUBLIC_KEY, 4)
        bad = [dict(r) for r in rows]
        bad[3]["Y"] = 0.5
        with self.assertRaisesRegex(ValueError, "Y does not match"):
            validate_article(bad, article, PUBLIC_KEY, 4)

    def test_truth_agrees_with_context_matching_labels_of_the_existing_protocol(self):
        # operational_labels reasons from tokens and contexts; S \ B reasons from edit positions only.
        with scratch() as folder:
            lm, _ = generate_sources(SMOKE, Path(folder) / "src", vocab=500)
            sources = read_jsonl(Path(folder) / "src/sources.jsonl")
        source = next(s for s in sources if s["scenario"] == "switch")
        tokens, m, n = source["token_ids"], 4, len(source["token_ids"])
        probs = lm.replay(source["prompt_ids"], tokens)
        for seed in range(5):
            final, edits, _ = resample(tokens, probs, .3, 1.0, np.random.default_rng(seed))
            edited = {e["t"] for e in edits}
            origin = [None if t in edited else t for t in range(n)]
            contexts = generation_contexts(source["prompt_ids"], tokens, m)
            enabled = [any(a <= t < b for a, b in source["ranges"]) for t in range(n)]
            _, signal = operational_labels(final, origin, tokens, contexts, enabled, m)
            np.testing.assert_array_equal(signal[m:], truth_mask(n, source["ranges"], sorted(edited), m)[m:])


class ResampleTests(unittest.TestCase):
    def distributions(self, n=40, vocab=30, seed=0):
        rng = np.random.default_rng(seed)
        probs = rng.dirichlet(np.full(vocab, .3), size=n)
        probs[::2] = rng.dirichlet(np.full(vocab, 5.), size=n // 2)
        return probs

    def test_edits_only_at_high_entropy_positions_with_a_different_token(self):
        probs = self.distributions()
        tokens = [int(np.argmax(p)) for p in probs]
        final, edits, stats = resample(tokens, probs, .3, 2.0, np.random.default_rng(1))
        self.assertEqual(stats["requested"], 12)
        self.assertEqual(stats["edited"] + stats["shortfall"], 12)
        for e in edits:
            self.assertGreaterEqual(entropy_nats(probs[e["t"]]), 2.0)
            self.assertNotEqual(e["new"], e["old"])
            self.assertEqual(final[e["t"]], e["new"])
            self.assertGreater(probs[e["t"]][e["new"]], 0)
            self.assertGreaterEqual(e["new_rank"], 1)
        edited = {e["t"] for e in edits}
        self.assertTrue(all(final[t] == tokens[t] for t in range(len(tokens)) if t not in edited))

    def test_shortfall_is_recorded_when_too_few_positions_are_eligible(self):
        probs = self.distributions()
        tokens = [int(np.argmax(p)) for p in probs]
        _, edits, stats = resample(tokens, probs, .3, 100.0, np.random.default_rng(1))
        self.assertEqual((edits, stats["eligible"], stats["shortfall"]), ([], 0, 12))
        _, edits, stats = resample(tokens, probs, .9, 2.0, np.random.default_rng(1))
        self.assertEqual(len(edits), min(stats["requested"], stats["eligible"]))
        _, edits, stats = resample(tokens, probs, .9, 4.0, np.random.default_rng(1))
        self.assertEqual((len(edits), stats["shortfall"] > 0), (stats["eligible"], stats["eligible"] < 36))

    def test_deterministic_for_a_seed(self):
        probs = self.distributions()
        tokens = [int(np.argmax(p)) for p in probs]
        first = resample(tokens, probs, .2, 1.0, np.random.default_rng(7))
        self.assertEqual(first, resample(tokens, probs, .2, 1.0, np.random.default_rng(7)))

    def test_replacement_is_drawn_without_replacement_from_the_model_distribution(self):
        probs = np.tile([.5, .3, .2, 0., 0.], (3000, 1))
        final, edits, _ = resample([0] * 3000, probs, .9, 0.0, np.random.default_rng(3))
        counts = np.bincount([e["new"] for e in edits], minlength=5)
        self.assertEqual(counts[0] + counts[3] + counts[4], 0)
        self.assertAlmostEqual(counts[1] / counts.sum(), .6, delta=.03)


class RuleTests(unittest.TestCase):
    def null_items(self, n=400, seed=0, length=124):
        rng = np.random.default_rng(seed)
        return [(length, np.clip(rng.beta(.3, 3., length), 0, 1)) for _ in range(n)]

    def test_fraction_rule_with_the_max_threshold_is_the_max_rule(self):
        config = {"alpha": .05, "fraction_tau": .5, "min_group_size": 100}
        rules = build_rules(self.null_items(), config)
        self.assertEqual(lookup(rules["P_same"], 124).value, 0.)
        rng = np.random.default_rng(9)
        for _ in range(300):
            scores = np.clip(rng.beta(.4, 2., 124), 0, 1)
            _, d = decide(scores, 124, rules)
            self.assertEqual(d["P_same"], d["M"])

    def test_separate_fraction_threshold_can_differ_from_max_rule(self):
        config = {"alpha": .05, "fraction_tau": .5, "min_group_size": 100}
        rules = build_rules(self.null_items(), config)
        spread = np.full(124, .6)
        spiky = np.zeros(124)
        spiky[10] = 1.
        self.assertTrue(decide(spread, 124, rules)[1]["P"])
        self.assertFalse(decide(spiky, 124, rules)[1]["P"])
        self.assertTrue(decide(spiky, 124, rules)[1]["M"])

    def test_null_false_alarm_rate_is_at_most_alpha_on_fresh_null(self):
        config = {"alpha": .05, "fraction_tau": .5, "min_group_size": 100}
        rules = build_rules(self.null_items(seed=1), config)
        fresh = self.null_items(n=2000, seed=2)
        rate = np.mean([decide(s, n, rules)[1]["M"] for n, s in fresh])
        self.assertLessEqual(rate, .05 + .02)

    def test_too_few_null_articles_never_fires(self):
        rules = build_rules(self.null_items(n=3), {"alpha": .05, "fraction_tau": .5, "min_group_size": 100})
        self.assertFalse(lookup(rules["M"], 124).finite_rank)
        self.assertFalse(decide(np.ones(124), 124, rules)[1]["M"])

    def test_small_length_groups_fall_back_to_the_pooled_threshold(self):
        items = self.null_items(n=200, length=124) + self.null_items(n=10, seed=5, length=60)
        rules = build_rules(items, {"alpha": .05, "fraction_tau": "same", "min_group_size": 100})
        self.assertIn("124", rules["M"])
        self.assertNotIn("60", rules["M"])
        self.assertIs(lookup(rules["M"], 60), rules["M"]["pooled"])
        self.assertIs(rules["P"], rules["P_same"])


class MetricTests(unittest.TestCase):
    def test_position_metrics(self):
        truth = np.array([1, 1, 0, 0, 1], bool)
        pred = np.array([1, 0, 1, 0, 1], bool)
        m = position_metrics(truth, pred)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["iou"], 2 / 4)
        self.assertAlmostEqual(m["f1"], 4 / 6)
        empty = position_metrics(np.zeros(4, bool), np.ones(4, bool))
        self.assertIsNone(empty["iou"])
        self.assertIsNone(empty["recall"])

    def test_segment_recall_uses_coverage_and_boundary_offsets(self):
        truth = np.zeros(30, bool)
        truth[2:12] = True
        truth[20:28] = True
        pred = np.zeros(30, bool)
        pred[4:12] = True
        pred[20:22] = True
        r = segment_report(truth, pred)
        self.assertEqual((r["true_segments"], r["recalled_segments"]), (2, 1))
        self.assertEqual(r["boundary_errors"], [(2, 0)])
        self.assertAlmostEqual(r["segment_recall"], .5)

    def test_paired_bootstrap_is_paired(self):
        base = np.random.default_rng(0).uniform(.2, .9, 40)
        indices = bootstrap_indices(40, 500, 1)
        shifted = paired_difference(base + .1, base, indices)
        self.assertAlmostEqual(shifted["mean"], .1)
        self.assertAlmostEqual(shifted["ci95"][0], .1)
        self.assertAlmostEqual(shifted["ci95"][1], .1)

    def test_auc(self):
        self.assertEqual(auc([3, 4], [1, 2]), 1.)
        self.assertEqual(auc([1, 1], [1, 1]), .5)


class EmissionTests(unittest.TestCase):
    def test_cap_bounds_every_family_and_histogram_is_normalized(self):
        y = np.random.default_rng(0).beta(6., .6, 3000)
        y = np.clip(y, 1e-9, 1 - 1e-9)
        grid = np.linspace(1e-6, 1 - 1e-6, 10001)
        for name in ("histogram", "beta", "beta_mixture"):
            density = fit_emission(name, y, {"bins": 25, "pseudocount": .5, "cap": 25})
            self.assertLessEqual(float(density(grid).max()), 25.)
            self.assertGreater(float(density(grid).min()), 0.)
        self.assertGreater(float(fit_emission("beta_nocap", y, {"cap": 25})(grid).max()), 25.)
        hist = fit_emission("histogram", y, {"bins": 25, "pseudocount": .5, "cap": 25})
        self.assertAlmostEqual(integral_mass(hist), 1., places=3)


class ConfigTests(unittest.TestCase):
    def test_shipped_configs_are_valid(self):
        validate_l0(SMOKE)
        validate_l0(FULL)

    def test_rejects_protocol_violations(self):
        for change in ({"roles": {**SMOKE["roles"], "test": 5}}, {"merge_gap": 1}, {"edits": []},
                       {"edits": [{"kind": "replace", "rate": .1}, {"kind": "replace", "rate": .3}]},
                       {"decision": ["max_score"]}, {"emission": []}, {"alpha": 0}, {"fraction_tau": 1.5},
                       {"rho_grid": []}, {"cap": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_l0({**SMOKE, **change})

    def test_scenarios_are_balanced_and_planned_from_role_order(self):
        self.assertEqual([scenario_for("test", i) for i in range(4)], ["full_wm", "switch", "resample_0.1", "resample_0.3"])
        self.assertEqual({scenario_for("test", i) for i in range(160)}, set(scenario_for("test", i) for i in range(4)))
        prompts = [{"role": r, "group": f"{r}{i}", "prompt_ids": [1] * 50}
                   for r, c in SMOKE["roles"].items() for i in range(c)]
        jobs = plan_jobs(prompts, SMOKE)
        by_scenario = {j["scenario"]: j["ranges"] for j in jobs}
        self.assertEqual(by_scenario["switch"], [[25, 89]])
        self.assertEqual(by_scenario["null"], [])
        self.assertEqual(by_scenario["full_wm"], [[0, 128]])
        self.assertEqual(len(jobs), 8)

    def test_source_record_rejects_a_sampler_audit_that_does_not_match_the_tokens(self):
        job = {"doc": {"group": "g", "role": "test", "prompt_ids": [1, 2, 3, 4, 5]}, "scenario": "null", "ranges": []}
        tokens = [7, 8, 9, 10, 11]
        contexts = generation_contexts(job["doc"]["prompt_ids"], tokens, 4)
        from opt_hmm.protocol import context_seed
        audit = {t: {"step": t, "active": False, "context": contexts[t], "seed": str(context_seed(contexts[t], PUBLIC_KEY))}
                 for t in range(5)}
        self.assertEqual(source_record(job, tokens, audit, PUBLIC_KEY, 4)["token_ids"], tokens)
        audit[2] = {**audit[2], "active": True}
        with self.assertRaises(RuntimeError):
            source_record(job, tokens, audit, PUBLIC_KEY, 4)

    def test_source_record_tolerates_only_the_trailing_processor_call(self):
        from opt_hmm.protocol import context_seed
        job = {"doc": {"group": "g", "role": "test", "prompt_ids": [1, 2, 3, 4, 5]}, "scenario": "null", "ranges": []}
        tokens = [7, 8, 9, 10, 11]
        contexts = generation_contexts(job["doc"]["prompt_ids"], tokens, 4)
        audit = {t: {"step": t, "active": False, "context": contexts[t], "seed": str(context_seed(contexts[t], PUBLIC_KEY))}
                 for t in range(5)}
        audit[5] = {"step": 5, "active": False, "context": [8, 9, 10, 11], "seed": "0"}
        self.assertEqual(source_record(job, tokens, dict(audit), PUBLIC_KEY, 4)["token_ids"], tokens)
        audit[6] = dict(audit[5], step=6)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            source_record(job, tokens, dict(audit), PUBLIC_KEY, 4)
        del audit[6], audit[3]
        with self.assertRaisesRegex(RuntimeError, "missing"):
            source_record(job, tokens, dict(audit), PUBLIC_KEY, 4)


class ToyPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = scratch()
        base = Path(cls.folder.name)
        config = {**SMOKE, "bootstrap": 200}
        lm, _ = generate_sources(config, base / "src", vocab=800)
        cls.build = build(base / "src", base / "pos", replay=lm.replay)
        cls.result = evaluate_positions(base / "pos", base / "eval")
        cls.base = base

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_positions_follow_the_contract(self):
        rows = read_jsonl(self.base / "pos/positions.jsonl")
        self.assertEqual(len(rows), 8 * 124)
        self.assertEqual({r["scenario"] for r in rows}, {"full_wm", "switch", "resample_0.1", "resample_0.3", "null"})
        self.assertEqual({r["split"] for r in rows}, {"calibration", "dev", "test", "null_cal", "null_test"})
        edited = [r for r in rows if r["edited"]]
        self.assertEqual({r["scenario"] for r in edited}, {"resample_0.1", "resample_0.3"})
        self.assertTrue(all(r["dist_to_edit"] == 0 for r in edited))
        self.assertTrue(all(not r["truth"] for r in rows if r["scenario"] == "null"))

    def test_smoke_exit_criteria(self):
        report = read_json(self.base / "eval/report.json")
        self.assertTrue(report["smoke_check"]["all_metrics_present"], report["smoke_check"]["missing"])
        self.assertEqual(set(report["summary"]), {"full_wm", "switch", "resample_0.1", "resample_0.3"})
        self.assertEqual(report["rule_equivalence"]["P_same_equals_M"], 1.0)
        self.assertTrue((self.base / "eval/frozen.json").is_file())
        text = (self.base / "eval/report.md").read_text(encoding="utf-8")
        self.assertIn("玩具语言模型", text)

    def test_frozen_thresholds_are_the_ones_used(self):
        frozen = read_json(self.base / "eval/frozen.json")
        report = read_json(self.base / "eval/report.json")
        for key, value in report["thresholds_pooled"].items():
            self.assertEqual(frozen["rules"][key]["M"]["pooled"]["value"], value["tau"])

    def test_evaluation_may_override_emission_but_not_protocol(self):
        with scratch() as other:
            with self.assertRaisesRegex(ValueError, "protocol"):
                evaluate_positions(self.base / "pos", Path(other) / "a", {"tokens": 100})
            evaluate_positions(self.base / "pos", Path(other) / "b", {"emission": ["beta_nocap"], "bootstrap": 20})
            report = read_json(Path(other) / "b/report.json")
            self.assertEqual(set(report["summary"]["switch"]), {"beta_nocap"})

    def test_output_directory_cannot_be_reused(self):
        with self.assertRaises(FileExistsError):
            evaluate_positions(self.base / "pos", self.base / "eval")


class ToyModelTests(unittest.TestCase):
    def test_replay_matches_generation_distributions(self):
        lm = ToyLM(300, seed=2)
        prompt, tokens = [1, 2, 3, 4], [5, 6, 7]
        probs = lm.replay(prompt, tokens)
        np.testing.assert_allclose(probs[1], lm.probs(prompt + tokens[:1]))
        np.testing.assert_allclose(probs.sum(axis=1), 1.)


if __name__ == "__main__":
    unittest.main()
