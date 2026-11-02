import json
import sys
import unittest
import numpy as np

sys.path.insert(0, "src")

from opt_hmm.parametric import (FAMILIES, ParametricConditionalDensity,
                                ParametricDensity, _clip)

GRID = np.linspace(1e-6, 1.0 - 1e-6, 20001)
_TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def integrate(density):
    values = np.asarray(density(GRID), dtype=float)
    return float(_TRAPEZOID(values, GRID))


class ParametricDensityTest(unittest.TestCase):
    def sample(self):
        rng = np.random.default_rng(0)
        return rng.beta(2.0, 1.0, size=4000)

    def test_every_family_integrates_to_one(self):
        y = self.sample()
        for family in FAMILIES:
            with self.subTest(family=family):
                density = ParametricDensity.fit(y, family)
                mass = integrate(density)
                self.assertGreater(mass, 0.7)
                self.assertLess(mass, 1.3)

    def test_density_is_positive_and_finite(self):
        y = self.sample()
        for family in FAMILIES:
            with self.subTest(family=family):
                density = ParametricDensity.fit(y, family)
                values = np.asarray(density(GRID), dtype=float)
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertTrue(np.all(values > 0.0))

    def test_beta_recovers_moments(self):
        y = self.sample()
        density = ParametricDensity.fit(y, "beta")
        self.assertAlmostEqual(density.params["a"], 2.0, delta=0.5)
        self.assertAlmostEqual(density.params["b"], 1.0, delta=0.5)

    def test_cap_bounds_density(self):
        y = np.array([1e-9, 0.5, 1.0 - 1e-9])
        density = ParametricDensity.fit(self.sample(), "beta", cap=5.0)
        self.assertLessEqual(float(np.max(density(y))), 5.0)

    def test_rejects_empty_and_unknown(self):
        with self.assertRaises(ValueError):
            ParametricDensity.fit([], "beta")
        with self.assertRaises(ValueError):
            ParametricDensity.fit([0.5], "not_a_family")

    def test_to_dict_is_json_serialisable(self):
        y = self.sample()
        for family in FAMILIES:
            with self.subTest(family=family):
                json.dumps(ParametricDensity.fit(y, family).to_dict())


class ParametricConditionalDensityTest(unittest.TestCase):
    def build(self):
        rng = np.random.default_rng(1)
        s2 = rng.uniform(0.0, 1.0, size=3000)
        # Stronger signal at low S2, mimicking the pivot law.
        y = rng.beta(1.0 + 4.0 * (1.0 - s2), 1.0, size=3000)
        return _clip(y), s2

    def test_call_returns_positive_scalar(self):
        y, s2 = self.build()
        conditional = ParametricConditionalDensity.fit(y, s2, "beta")
        for value, strength in ((0.01, 0.01), (0.5, 0.5), (0.99, 0.99)):
            with self.subTest(y=value, s2=strength):
                result = conditional(value, strength)
                self.assertIsInstance(result, float)
                self.assertGreater(result, 0.0)
                self.assertTrue(np.isfinite(result))

    def test_small_groups_fall_back_to_global(self):
        y, s2 = self.build()
        conditional = ParametricConditionalDensity.fit(y, s2, "beta", min_count=100000)
        self.assertTrue(all(d is None for d in conditional.densities))
        global_at_half = float(np.asarray(conditional.global_density(0.5)).reshape(-1)[0])
        self.assertAlmostEqual(conditional(0.5, 0.5), global_at_half, places=12)

    def test_counts_cover_all_groups(self):
        y, s2 = self.build()
        conditional = ParametricConditionalDensity.fit(y, s2, "beta")
        self.assertEqual(len(conditional.counts), 6)
        self.assertEqual(sum(conditional.counts), len(y))

    def test_to_dict_round_trips_through_json(self):
        y, s2 = self.build()
        payload = ParametricConditionalDensity.fit(y, s2, "gamma").to_dict()
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
