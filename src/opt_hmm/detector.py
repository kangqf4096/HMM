"""Fitted per-document inference accepts pivots and mask, never ground truth."""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .emissions import check_pivots
from .hmm import forward_backward, forward_backward_two_state, forward_log_likelihood
from .proportion import Proportion, estimate_proportion

DEFAULT_RHOS = (0., .5, .8, .9, .95, .98, .99)
METHODS = ("independent", "fixed_e", "opt_hmm", "fixed_rho", "window")
# Global-transition HMMs (A estimated by EM and frozen) and the old method with the eps_opt estimator.
# Kept out of METHODS: the old L0 evaluation iterates METHODS.
GENERAL_METHODS = ("pd", "pe", "p1", "p2")
LI_METHODS = ("opt_hmm_li",)


@dataclass
class Detection:
    scores: np.ndarray
    epsilon: float
    rho: float
    identifiable: bool
    seconds: float
    log_likelihood: float | None = None
    stage_seconds: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


class Detector:
    """`params` = {"a00", "a11", "pi1"} for the global-transition methods (pi1 used by pe only);
    `proportion` = OptProportion bound to the same capped density (pd, p1, opt_hmm_li, and eps reporting)."""

    def __init__(self, density, method="opt_hmm", rhos=DEFAULT_RHOS,
                 fixed_e=.5, fixed_rho=.95, window=20, params=None, proportion=None):
        if method not in METHODS + GENERAL_METHODS + LI_METHODS and method != "oracle":
            raise ValueError(f"Unknown method {method}")
        if method in GENERAL_METHODS and (not params or proportion is None):
            raise ValueError(f"{method} needs frozen transition params and an eps_opt estimator.")
        if method in LI_METHODS and proportion is None:
            raise ValueError(f"{method} needs an eps_opt estimator.")
        self.params, self.proportion = params, proportion
        if not 0 <= fixed_e <= 1 or not 0 <= fixed_rho < 1 or window < 1:
            raise ValueError("Invalid detector configuration.")
        self.density, self.method = density, method
        self.rhos = sorted(set(float(r) for r in rhos))
        if not self.rhos or any(not 0 <= r < 1 for r in self.rhos):
            raise ValueError("Invalid rho candidates.")
        self.fixed_e, self.fixed_rho, self.window = fixed_e, fixed_rho, window

    def infer(self, y, mask=None):
        started = perf_counter()
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or not len(y):
            raise ValueError("Need a nonempty 1D sequence.")
        mask = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
        if mask.shape != y.shape:
            raise ValueError("Mask must match sequence length.")
        g = np.ones(len(y))
        g[mask] = self.density(check_pivots(y[mask]))
        return self._infer_ratios(g, mask, started)

    def infer_many(self, ys):
        """Same results as infer() per document; global-transition methods batch equal-length documents."""
        if self.method not in GENERAL_METHODS:
            return [self.infer(y) for y in ys]
        started = perf_counter()
        rows = [self.density(check_pivots(np.asarray(y, float))) for y in ys]
        results, by_length = [None] * len(ys), {}
        for index, g in enumerate(rows):
            by_length.setdefault(len(g), []).append(index)
        for indices in by_length.values():
            for i, r in zip(indices, self.infer_many_ratios(np.stack([rows[i] for i in indices]))):
                results[i] = r
        for r in results:
            r.seconds = (perf_counter() - started) / len(ys)
        return results

    def _first_prior(self, eps, n):
        """pi_1 per document and the diagnostics that belong to that rule."""
        from .hmm import stationary
        from .l0.em import ALPHA_MIN, alpha_bar, first_prior
        a00, a11 = self.params["a00"], self.params["a11"]
        pi1 = first_prior(self.method, np.array([a00]), np.array([a11]),
                          np.array([self.params.get("pi1", .5)]), eps, n)[0]
        pi1 = np.broadcast_to(pi1, eps.shape).astype(float)
        info = {}
        if self.method == "p1":
            ab = float(alpha_bar(a00, a11, n))
            lam, star = a00 + a11 - 1, stationary(a00, a11)
            marginal = star + (pi1 - star) * np.mean(lam ** np.arange(n))
            info = {"alpha_bar": ab, "pi_unidentifiable": ab < ALPHA_MIN,
                    "pi_clipped": np.abs(marginal - eps) > .02, "marginal_gap": np.abs(marginal - eps)}
        return pi1, info

    def infer_ratios(self, g, mask=None):
        """Inference using supplied, label-free positive per-position ratios."""
        started = perf_counter()
        g = np.asarray(g, dtype=float)
        if g.ndim != 1 or not len(g):
            raise ValueError('Need a nonempty 1D ratio vector.')
        mask = np.ones(len(g), bool) if mask is None else np.asarray(mask, bool)
        if (mask.shape != g.shape
                or np.any(~np.isfinite(g)) or np.any(g <= 0)):
            raise ValueError('Need a nonempty positive finite ratio vector and matching mask.')
        return self._infer_ratios(g, mask, started)

    def _infer_ratios(self, g, mask, started):
        if not np.any(mask):
            return Detection(np.zeros(len(g)), 0, 0, False, perf_counter() - started)
        if self.method in GENERAL_METHODS:
            if not mask.all():
                raise ValueError("Global-transition methods run on full sequences (L0 has no masked positions).")
            return self._general(g, started)
        stages = {"density_lookup": perf_counter() - started}
        phase = perf_counter()
        if self.method in LI_METHODS:
            opt = self.proportion.estimate(g[mask])
            fitted = Proportion(opt.value, opt.identifiable, 0)
        else:
            fitted = estimate_proportion(g[mask])
        stages["proportion_fit"] = perf_counter() - phase
        e = self.fixed_e if self.method in ("fixed_e", "oracle") else fitted.value
        if not fitted.identifiable:
            return Detection(np.zeros(len(g)), e, 0, False, perf_counter() - started,
                             stage_seconds=stages)
        if self.method == "window":
            phase = perf_counter()
            # Fixed 0.5 prior, centered average log density ratio. Missing
            # positions break a window, just as they break reported segments.
            score = np.zeros(len(g))
            for start, stop in intervals(mask):
                logs = np.log(g[start:stop])
                prefix = np.r_[0., np.cumsum(logs)]
                for local in range(stop - start):
                    lo = max(0, local - self.window // 2)
                    hi = min(len(logs), local + (self.window + 1) // 2)
                    avg = (prefix[hi] - prefix[lo]) / (hi - lo)
                    score[start + local] = 1 / (1 + np.exp(-np.clip(avg, -700, 700)))
            rho, ll = 0., None
            stages["window_score"] = perf_counter() - phase
        else:
            phase = perf_counter()
            if self.method == "independent":
                rho = 0.
            elif self.method in ("fixed_rho", "oracle"):
                rho = self.fixed_rho
            else:
                # Python max preserves the first (smaller rho) exact tie.
                rho = max(self.rhos, key=lambda r: forward_log_likelihood(g, e, r, mask))
            stages["rho_selection"] = perf_counter() - phase
            phase = perf_counter()
            result = forward_backward(g, e, rho, mask)
            score, ll = result.gamma, result.log_likelihood
            stages["forward_backward"] = perf_counter() - phase
        score[~mask] = 0
        return Detection(score, e, rho, fitted.identifiable, perf_counter() - started, ll, stages)


    def _general(self, g, started):
        result, = self.infer_many_ratios(g[None])
        result.seconds = perf_counter() - started
        return result

    def infer_many_ratios(self, g):
        """Batch of equal-length ratio rows (one document per row) for the global-transition methods."""
        eps_fit = [self.proportion.estimate(row) for row in g]
        results = [None] * len(g)
        ok = [i for i, e in enumerate(eps_fit) if e.identifiable]
        for i in set(range(len(g))) - set(ok):
            results[i] = Detection(np.zeros(g.shape[1]), eps_fit[i].value, 0, False, 0.)
        if ok:
            eps = np.array([eps_fit[i].value for i in ok])
            pi1, info = self._first_prior(eps, g.shape[1])
            a00, a11 = self.params["a00"], self.params["a11"]
            fb = forward_backward_two_state(g[ok], pi1, a00, a11, want_gamma=True)
            for row, i in enumerate(ok):
                results[i] = Detection(fb.gamma[row], float(eps[row]), a00 + a11 - 1, True, 0.,
                                       float(fb.log_likelihood[row]),
                                       info={"pi1": float(pi1[row]), **{k: v[row] if np.ndim(v) else v
                                                                      for k, v in info.items()}})
        return results


def intervals(labels):
    """Zero-based half-open intervals. Unavailable positions must be False."""
    values = np.asarray(labels, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Labels must be 1D.")
    edges = np.diff(np.r_[False, values, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
