import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_hmm.calibration import calibrate
from opt_hmm.detector import Detector, intervals
from opt_hmm.emissions import Histogram, sample_signal, theoretical_cdf, theoretical_density
from opt_hmm.hmm import forward_backward, forward_log_likelihood, transition, viterbi
from opt_hmm.proportion import estimate_proportion


def enumerate_paths(g, epsilon, rho, mask=None):
    pi, a = transition(epsilon, rho)
    g = np.where(mask, g, 1.) if mask is not None else g
    weights, paths = [], list(itertools.product((0, 1), repeat=len(g)))
    for path in paths:
        weight = pi[path[0]]
        for t, state in enumerate(path):
            if t:
                weight *= a[path[t - 1], state]
            if state:
                weight *= g[t]
        weights.append(weight)
    weights = np.asarray(weights)
    return float(weights.sum()), (weights[:, None] * np.asarray(paths)).sum(0) / weights.sum(), paths[int(weights.argmax())]


class HMMTests(unittest.TestCase):
    def test_forward_backward_and_viterbi_against_exhaustive_paths(self):
        rng = np.random.default_rng(10)
        for n in (1, 2, 5, 12):
            g = np.exp(rng.normal(size=n))
            for e, rho in ((.37, .8), (.5, 0), (0., .95), (1., .95)):
                with self.subTest(n=n, e=e, rho=rho):
                    likelihood, gamma, path = enumerate_paths(g, e, rho)
                    result = forward_backward(g, e, rho)
                    self.assertAlmostEqual(result.log_likelihood, np.log(likelihood), places=11)
                    np.testing.assert_allclose(result.gamma, gamma, atol=1e-11)
                    np.testing.assert_array_equal(viterbi(g, e, rho), path)

    def test_rho_zero_matches_independent_mixture(self):
        g = np.array([.1, .3, 1., 2., 10.])
        e = .24
        np.testing.assert_allclose(forward_backward(g, e, 0).gamma, e * g / (1 - e + e * g))

    def test_missing_observations_are_neutral_and_break_output_segments(self):
        g, mask = np.array([3., 50., .2, 2.]), np.array([1, 0, 1, 1], bool)
        likelihood, gamma, _ = enumerate_paths(g, .4, .8, mask)
        result = forward_backward(g, .4, .8, mask)
        np.testing.assert_allclose(result.gamma, gamma)
        self.assertAlmostEqual(result.log_likelihood, np.log(likelihood))
        self.assertEqual(intervals(np.ones(4, bool) & mask), [(0, 1), (2, 4)])

    def test_stationarity_and_expected_run_lengths(self):
        pi, a = transition(.3, .95)
        np.testing.assert_allclose(pi @ a, pi)
        self.assertAlmostEqual(1 / a[1, 0], 1 / ((1 - .95) * .7))
        self.assertAlmostEqual(1 / a[0, 1], 1 / ((1 - .95) * .3))

    def test_long_sequence_stays_finite(self):
        g = np.tile([.001, 20., .2, 3.], 2500)
        result = forward_backward(g, .3, .98)
        self.assertTrue(np.isfinite(result.log_likelihood))
        self.assertTrue(np.isfinite(result.gamma).all())
        self.assertTrue(np.all((result.gamma >= 0) & (result.gamma <= 1)))

    def test_invalid_inputs(self):
        for g in ([0, 1], [1, np.nan], []):
            with self.assertRaises(ValueError):
                forward_backward(g, .5, .9)


class EmissionAndProportionTests(unittest.TestCase):
    def test_histogram_normalization_and_empty_bins(self):
        h = Histogram.fit([.99] * 10, bins=100)
        self.assertAlmostEqual(h.density.mean(), 1.)
        self.assertTrue((h.density > 0).all())
        restored = Histogram.from_dict(h.to_dict())
        np.testing.assert_array_equal(restored([.1, .99]), h([.1, .99]))

    def test_signal_sampling_matches_cdf(self):
        rng = np.random.default_rng(22)
        p = [.8, .15, .05]
        sample = sample_signal(rng, 50000, p)
        grid = np.linspace(.1, .9, 9)
        empirical = (sample[:, None] <= grid).mean(0)
        np.testing.assert_allclose(empirical, theoretical_cdf(grid, p), atol=.009)

    def test_direct_gumbel_max_matches_pivot_formula(self):
        rng = np.random.default_rng(29)
        p = np.array([.2, .3, .5])
        u = rng.uniform(np.nextafter(0., 1.), 1., size=(50000, 3))
        winners = np.argmax(np.log(u) / p, axis=1)
        y = u[np.arange(len(u)), winners]
        grid = np.array([.25, .5, .75, .9])
        np.testing.assert_allclose((y[:, None] <= grid).mean(0), theoretical_cdf(grid, p), atol=.008)
        np.testing.assert_allclose(np.bincount(winners, minlength=3) / len(winners), p, atol=.008)

    def test_degenerate_distribution_has_no_information(self):
        y = np.array([.01, .2, .8, .99])
        np.testing.assert_allclose(theoretical_density(y, [1, 0]), 1.)
        np.testing.assert_allclose(theoretical_cdf(y, [1, 0]), y)
        self.assertFalse(estimate_proportion(np.ones(10)).identifiable)
        output = Detector(lambda x: np.ones(len(x))).infer(y)
        self.assertFalse(output.identifiable)
        self.assertTrue((output.scores == 0).all())

    def test_proportion_endpoints_and_interior_score(self):
        self.assertEqual(estimate_proportion([.1, .5]).value, 0.)
        self.assertEqual(estimate_proportion([2., 3.]).value, 1.)
        g = np.array([.1, .4, .9, 2., 3., 5.])
        result = estimate_proportion(g)
        derivative = np.sum((g - 1) / (1 - result.value + result.value * g))
        self.assertAlmostEqual(derivative, 0, places=6)
        grid = np.linspace(0, 1, 20001)
        objective = np.log(1 - grid[:, None] + grid[:, None] * g).sum(1)
        self.assertLessEqual(abs(grid[objective.argmax()] - result.value), 1 / 20000)

    def test_unknown_and_zero_available_inputs(self):
        with self.assertRaises(ValueError):
            Histogram.fit([], 10)
        out = Detector(lambda y: np.ones(len(y))).infer([np.nan] * 4, [False] * 4)
        self.assertFalse(out.identifiable)
        self.assertTrue((out.scores == 0).all())

    def test_proportion_fits_only_available_positions(self):
        h = Histogram(np.array([.4, 1.6]))
        d = Detector(h, method="independent")
        valid = d.infer([.2, .8])
        missing = d.infer([None, .2, .8], [False, True, True])
        self.assertEqual(valid.epsilon, missing.epsilon)
        np.testing.assert_allclose(valid.scores, missing.scores[1:])


class CalibrationTests(unittest.TestCase):
    def test_rank_1901(self):
        threshold = calibrate(np.linspace(0, 1, 2000), .05)
        self.assertEqual(threshold.rank, 1901)
        self.assertEqual(threshold.value, np.linspace(0, 1, 2000)[1900])

    def test_strict_ties_and_insufficient_nulls(self):
        threshold = calibrate([.4] * 32)
        self.assertFalse(threshold.predict([.4])[0])
        self.assertTrue(threshold.predict([.4001])[0])
        unavailable = calibrate([.1] * 10)
        self.assertFalse(unavailable.finite_rank)
        self.assertFalse(unavailable.predict([1.])[0])

    def test_leave_one_out_rank_error_bound(self):
        values = np.linspace(0, 1, 20)
        errors = sum(calibrate(np.delete(values, i)).predict([values[i]])[0] for i in range(20))
        self.assertEqual(errors, 1)


if __name__ == "__main__":
    unittest.main()
