import math
import sys
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from opt_hmm.l0.seedbs import (article_seed, ks_scan, segment_pvalue, segment_scores, seedbs_intervals,
                               seedbs_precompute, seedbs_scores, seedbs_select, window_pvalues)


def brute_ks(p):
    l = len(p)
    out = []
    for j in range(1, l):
        best = 0.
        for t in np.unique(p):
            f1, f2 = np.mean(p[:j] <= t), np.mean(p[j:] <= t)
            best = max(best, abs(f1 - f2))
        out.append(j * (l - j) / l ** 1.5 * best)
    return np.array(out)


class IntervalTests(unittest.TestCase):
    def test_layers_follow_algorithm_one_for_m124(self):
        a, m = 2 ** -.5, 124
        got = seedbs_intervals(m, 6, a, 2.5)
        expected = []
        for k in range(1, math.ceil(math.log(m, 1 / a)) + 1):
            length = m * a ** (k - 1)
            if length < 15:
                continue
            n = 1 if k == 1 else 2 * math.ceil((1 / a) ** (k - 1) - 1e-12) - 1
            shift = (m - length) / (n - 1) if n > 1 else 0
            expected += [(math.floor(i * shift + 1e-12), math.ceil(i * shift + length - 1e-12)) for i in range(n)]
        self.assertEqual(got, list(dict.fromkeys(expected)))
        self.assertEqual(got[0], (0, 124))
        self.assertEqual(len([s for s in got if s[1] - s[0] in (88, 87)]), 3)
        self.assertTrue(all(s - r >= 15 for r, s in got))

    def test_min_length_scales_with_b(self):
        self.assertEqual(seedbs_intervals(124, 20), [(0, 124), (0, 88), (18, 106), (36, 124), (0, 62), (31, 93), (62, 124)])
        self.assertGreater(len(seedbs_intervals(124, 6)), len(seedbs_intervals(124, 20)))

    def test_small_case_by_hand(self):
        # m=8, a=1/sqrt2: layer 1 (0,8]; layer 2 l=5.657, n=3, shift=1.17: (0,6],(1,7],(2,8]
        self.assertEqual(seedbs_intervals(8, 2, 2 ** -.5, 2.5)[:4], [(0, 8), (0, 6), (1, 7), (2, 8)])


class KSTests(unittest.TestCase):
    def test_counting_identity_matches_definition(self):
        rng = np.random.default_rng(0)
        for l in (5, 12, 31):
            p = np.round(rng.random(l), 1) if l == 12 else rng.random(l)  # ties included
            np.testing.assert_allclose(ks_scan(p)[0], brute_ks(p), atol=1e-12)


class SegmentPValueTests(unittest.TestCase):
    def test_matches_poisson_tail_and_monte_carlo(self):
        for length, s in ((1, .3), (3, 1.7), (5, 9.), (10, 4.)):
            tail = 1 - sum(math.exp(-s) * s ** k / math.factorial(k) for k in range(length))
            self.assertAlmostEqual(segment_pvalue(s, length), tail, places=10)
        rng = np.random.default_rng(1)
        draws = rng.exponential(size=(200000, 20)).sum(axis=1)
        for s in (12., 17., 22.):
            self.assertAlmostEqual(segment_pvalue(s, 20), float((draws <= s).mean()), delta=.004)

    def test_small_values_do_not_cancel(self):
        self.assertGreater(segment_pvalue(1., 60), 0.)
        self.assertLess(segment_pvalue(1., 60), 1e-60)

    def test_scores_in_unit_interval_and_constant_per_segment(self):
        rng = np.random.default_rng(2)
        y = rng.random(50)
        sc = segment_scores(y, [20, 35])
        self.assertTrue(np.all((sc >= 0) & (sc <= 1)))
        self.assertEqual(len(set(sc[:20])), 1)
        self.assertEqual(len(set(sc[35:])), 1)
        strong = segment_scores(np.full(30, .999), [])
        self.assertGreater(strong[0], .999)


class SeedBSTests(unittest.TestCase):
    def test_article_seed_is_stable(self):
        self.assertEqual(article_seed(7, "arxiv-train-3"), article_seed(7, "arxiv-train-3"))
        self.assertNotEqual(article_seed(7, "a"), article_seed(7, "b"))

    def test_window_pvalues_are_valid(self):
        y = np.random.default_rng(3).random(60)
        p = window_pvalues(y, 10, 99, np.random.default_rng(0))
        self.assertTrue(np.all((p > 0) & (p <= 1)))

    def test_single_changepoint_recovered(self):
        hits = 0
        for seed in range(10):
            rng = np.random.default_rng(100 + seed)
            y = rng.random(124)
            y[:60] = rng.beta(4, 1, 60)
            y = np.clip(y, 1e-9, 1 - 1e-9)
            pre = seedbs_precompute(y, 10, seed, T=199, Tp=299)
            cuts = seedbs_select(pre, .01)
            hits += any(abs(c - 60) <= 6 for c in cuts)
        self.assertGreaterEqual(hits, 8)

    def test_null_has_few_false_changepoints(self):
        total = 0
        for seed in range(30):
            y = np.random.default_rng(500 + seed).random(124)
            total += len(seedbs_select(seedbs_precompute(y, 14, seed, T=199, Tp=299), .01))
        self.assertLess(total / 30, .6)  # block bootstrap on B-dependent p-values is mildly liberal; c is calibrated on null_cal

    def test_precompute_is_deterministic_and_independent_of_zeta(self):
        rng = np.random.default_rng(9)
        y = rng.random(80)
        y[:30] = rng.beta(5, 1, 30)
        a = seedbs_precompute(y, 10, 4, T=99, Tp=99)
        b = seedbs_precompute(y, 10, 4, T=99, Tp=99)
        self.assertEqual(a["ptilde"], b["ptilde"])
        self.assertEqual(a["cut"], b["cut"])
        loose, tight = seedbs_select(a, .5), seedbs_select(a, .001)
        self.assertGreaterEqual(len(loose), len(tight))
        scores, cuts = seedbs_scores(y, 10, .5, 4, T=99, Tp=99)
        self.assertEqual(cuts, loose)
        self.assertEqual(len(scores), len(y))
        self.assertIn("seconds", a)

    def test_not_selection_prefers_narrowest_and_drops_containing_intervals(self):
        pre = {"intervals": [(0, 100), (0, 40), (60, 100)], "cut": [50, 20, 80], "ptilde": [.001, .001, .001]}
        # narrowest (0,40) picks 20; (0,100) contains 20 and is removed; (60,100) picks 80.
        self.assertEqual(seedbs_select(pre, .01), [20, 80])


if __name__ == "__main__":
    unittest.main()
