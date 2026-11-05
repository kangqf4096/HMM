import json
import sys
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from opt_hmm.hmm import stationary
from opt_hmm.l0 import em
from opt_hmm.l0.config import validate_l0, validate_l1
from opt_hmm.l0.emission import Capped, ECDFPCHIP, V2_FAMILIES, fit_base, integral_mass
from opt_hmm.l0.evaluate_v2 import PARAM_FAMILIES, PARAM_METHODS, STAGE_METHODS, Run, main_methods, parse_method
from opt_hmm.parametric import ParametricDensity
from opt_hmm.proportion import OptProportion

SHAPE = 4.


def f1(y):
    return SHAPE * np.asarray(y) ** (SHAPE - 1)


def watermark_pivots(rng, size):
    return rng.beta(SHAPE, 1., size)


def markov_documents(rng, docs, n, a00, a11, pi1):
    """Pivots from a 2-state chain: state 0 -> U(0,1), state 1 -> Beta(SHAPE, 1); returns (y, z)."""
    z = np.empty((docs, n), int)
    z[:, 0] = rng.random(docs) < pi1
    for t in range(1, n):
        stay = np.where(z[:, t - 1] == 1, a11, a00)
        z[:, t] = np.where(rng.random(docs) < stay, z[:, t - 1], 1 - z[:, t - 1])
    y = np.where(z == 1, watermark_pivots(rng, (docs, n)), rng.random((docs, n)))
    return np.clip(y, 1e-9, 1 - 1e-9), z


class EMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.y, cls.z = markov_documents(rng, 150, 128, .95, .90, .3)
        cls.g = f1(cls.y)

    def test_pe_recovers_the_transition_and_first_state_law(self):
        fitted, _ = em.fit(self.g, "pe", grid_check=False)
        self.assertAlmostEqual(fitted["a00"], .95, delta=.03)
        self.assertAlmostEqual(fitted["a11"], .90, delta=.04)
        self.assertAlmostEqual(fitted["pi1"], .3, delta=.12)

    def test_pd_with_the_true_first_state_recovers_the_transition(self):
        fitted, _ = em.fit(self.g, "pd", eps=np.full(len(self.g), .3), grid_check=False)
        self.assertAlmostEqual(fitted["a00"], .95, delta=.03)
        self.assertAlmostEqual(fitted["a11"], .90, delta=.04)
        self.assertIsNone(fitted["pi1"])

    def test_pe_and_pd_log_likelihood_never_decreases(self):
        for mode, eps in (("pe", None), ("pd", np.full(len(self.g), .4))):
            out = em.run_em(self.g, mode, starts=[(.6, .6), (.9, .9), (.99, .8)], eps=eps, history=True, tol=0.)
            steps = np.diff(out["history"], axis=0)
            self.assertGreaterEqual(steps.min(), -1e-8, mode)
            self.assertFalse(out["nonmonotone"].any(), mode)

    def test_em_is_not_below_the_grid_optimum(self):
        for mode, eps in (("pe", None), ("p2", None)):
            fitted, surface = em.fit(self.g[:40], mode, eps, grid_check=True)
            self.assertGreaterEqual(fitted["grid_gap"], -1e-6, mode)
            self.assertEqual(surface.shape, (len(em.GRID), len(em.GRID)))

    def test_p1_prior_marginal_matches_eps_when_not_clipped(self):
        a00, a11, n = .98, .97, 128
        eps = np.array([.35, .42])
        star, lam = stationary(a00, a11), a00 + a11 - 1
        prior = em.first_prior("p1", [a00], [a11], None, eps, n)[0]
        for e, p in zip(eps, prior):
            marginal = star + (p - star) * lam ** np.arange(n)
            self.assertAlmostEqual(marginal.mean(), e, places=10)

    def test_p1_falls_back_to_the_stationary_law_when_alpha_bar_is_small(self):
        a00 = a11 = .9                                  # lambda = .8, alpha_bar ~ .0391 < .05
        prior = em.first_prior("p1", [a00], [a11], None, np.array([.9]), 128)[0, 0]
        self.assertAlmostEqual(prior, float(stationary(a00, a11)))

    def test_constrained_fit_ties_the_two_retention_rates(self):
        free, _ = em.fit(self.g, "pe", grid_check=False)
        tied = em.constrained_fit(self.g, "pe")
        self.assertLessEqual(tied["log_likelihood"], free["log_likelihood"] + 1e-6)
        test = em.equal_retention_test(free, tied)
        self.assertTrue(0 <= test["p_value"] <= 1)


class ProportionTests(unittest.TestCase):
    def test_eps_opt_recovers_the_mixing_proportion_including_the_ends(self):
        rng = np.random.default_rng(5)
        context = OptProportion(f1, watermark_pivots(rng, 40000))
        for eps in (.02, .3, .7, .98):
            y = np.where(rng.random(6000) < eps, watermark_pivots(rng, 6000), rng.random(6000))
            estimate = context.estimate(f1(np.clip(y, 1e-9, 1 - 1e-9)))
            self.assertTrue(estimate.identifiable)
            self.assertAlmostEqual(estimate.value, eps, delta=.04, msg=f"eps={eps}")

    def test_null_sample_gives_a_small_proportion(self):
        rng = np.random.default_rng(6)
        context = OptProportion(f1, watermark_pivots(rng, 40000))
        self.assertLess(context.estimate(f1(rng.random(4000))).value, .05)


class EmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(8)
        cls.y = watermark_pivots(rng, 12400)
        cls.grid = np.r_[1e-6, np.linspace(.001, .999, 999), 1 - 1e-6]

    def test_every_family_is_listed_and_fits(self):
        self.assertEqual(set(PARAM_FAMILIES), {"beta", "beta_mixture", "truncnorm", "logitnorm", "gamma", "weibull", "lognormal"})
        self.assertEqual(set(V2_FAMILIES), {"hist25", "hist200", "ecdf400", *PARAM_FAMILIES})
        for name in V2_FAMILIES:
            base = fit_base(name, self.y, {"pseudocount": .5})
            f = np.asarray(base(self.grid), float)
            self.assertTrue(np.all(f > 0) and np.all(np.isfinite(f)), name)

    def test_uncapped_parametric_and_nonparametric_densities_integrate_to_one(self):
        for name in V2_FAMILIES:
            density = Capped(fit_base(name, self.y, {"pseudocount": .5}), None)
            self.assertAlmostEqual(integral_mass(density), 1., delta=.02, msg=name)

    def test_cap_bounds_every_family_and_leaves_the_null_density_alone(self):
        for name in V2_FAMILIES:
            density = Capped(fit_base(name, self.y, {"pseudocount": .5}), 10.)
            self.assertLessEqual(float(np.max(density(self.grid))), 10. + 1e-9, name)
        with self.assertRaises(ValueError):
            Capped(fit_base("beta", self.y, {}), 1.)

    def test_capped_parametric_density_round_trips_through_json(self):
        density = Capped(fit_base("beta_mixture", self.y, {}), 25.)
        again = Capped.from_dict(json.loads(json.dumps(density.to_dict())))
        np.testing.assert_allclose(again(self.grid), density(self.grid))

    def test_ecdf_density_is_normalised_and_the_cdf_monotone(self):
        ecdf = ECDFPCHIP.fit(self.y, 400)
        self.assertAlmostEqual(integral_mass(ecdf), 1., delta=.02)
        self.assertTrue(np.all(np.diff(ecdf.cdf) > 0))

    def test_old_l0_names_keep_their_meaning(self):
        from opt_hmm.l0.emission import fit_emission
        density = fit_emission("beta", self.y, {"cap": 25})
        self.assertIsInstance(density, ParametricDensity)
        self.assertEqual(density.cap, 25)


class MethodNameTests(unittest.TestCase):
    def test_parametric_names_parse_including_underscored_and_nocap(self):
        self.assertEqual(parse_method("pd_beta_mixture")["family"], "beta_mixture")
        spec = parse_method("pe_beta_mixture_nocap")
        self.assertEqual((spec["mode"], spec["family"], spec["nocap"]), ("pe", "beta_mixture", True))
        with self.assertRaises(ValueError):
            parse_method("pd_nonsense")

    def test_stage_lists(self):
        self.assertEqual(len(PARAM_METHODS), 14)
        for stage in ("s3", "s4"):
            self.assertTrue(set(PARAM_METHODS) <= set(STAGE_METHODS[stage]))
        self.assertEqual(main_methods("gamma"), ("pd_hist200", "pe_hist200", "pd_gamma", "pe_gamma"))
        self.assertEqual(main_methods(None), ("pd_hist200", "pe_hist200"))


class ParamStarTests(unittest.TestCase):
    @staticmethod
    def run_with(scores, fallback=()):
        """scores: {family: (pd_iou, pe_iou)}; `fallback` lists methods that found no valid cap."""
        run = Run.__new__(Run)
        run.l1, run.param_star, run.states = {"em": {"bootstrap_draws": 0}}, None, {}
        for family in PARAM_FAMILIES:
            for mode, iou in zip(("pd", "pe"), scores.get(family, (.1, .1))):
                name = f"{mode}_{family}"
                bad = name in fallback
                run.states[name] = {"selection": {"fallback": bad, "iou": None if bad else iou, "cap": 25.}}
        return run

    def test_highest_mean_iou_wins(self):
        run = self.run_with({"gamma": (.50, .40), "weibull": (.60, .20), "beta": (.30, .30)})
        self.assertEqual(run.choose_param_star()["family"], "gamma")

    def test_a_family_needs_both_modes_to_be_eligible(self):
        run = self.run_with({"gamma": (.90, .90), "weibull": (.30, .30)}, fallback=("pe_gamma",))
        result = run.choose_param_star()
        self.assertEqual(result["family"], "weibull")
        self.assertFalse(next(r for r in result["table"] if r["family"] == "gamma")["eligible"])

    def test_ties_go_to_the_family_listed_first(self):
        run = self.run_with({"lognormal": (.4, .4), "logitnorm": (.4, .4), "truncnorm": (.4, .4)})
        self.assertEqual(run.choose_param_star()["family"], "truncnorm")

    def test_no_eligible_family_falls_back_to_beta(self):
        run = self.run_with({}, fallback=PARAM_METHODS)
        result = run.choose_param_star()
        self.assertEqual(result["family"], "beta")
        self.assertTrue(result["fallback"])

    def test_returns_none_when_the_candidates_were_not_run(self):
        run = self.run_with({})
        del run.states["pd_beta"]
        self.assertIsNone(run.choose_param_star())
        self.assertIsNone(run.param_star)


class ConfigTests(unittest.TestCase):
    def config(self):
        return json.loads((root / "configs" / "text_l0_v2.json").read_text(encoding="utf-8"))

    def test_shipped_v2_config_is_valid(self):
        config = self.config()
        validate_l0(config)
        validate_l1(config)

    def test_param_grid_must_be_finite_and_present(self):
        config = self.config()
        config["l1"]["cap_grid"]["param"] = [5, 10, None]
        with self.assertRaises(ValueError):
            validate_l1(config)
        del config["l1"]["cap_grid"]["param"]
        with self.assertRaises(ValueError):
            validate_l1(config)


if __name__ == "__main__":
    unittest.main()
