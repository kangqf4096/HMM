"""Marginal composite likelihood; this is not a joint HMM MLE."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Proportion:
    value: float
    identifiable: bool
    iterations: int


def estimate_proportion(g, tolerance=1e-8, max_iterations=60):
    g = np.asarray(g, dtype=float)
    if g.ndim != 1 or not len(g) or np.any(~np.isfinite(g)) or np.any(g <= 0):
        raise ValueError("Density ratios must be finite, positive, and nonempty.")
    if np.all(np.abs(g - 1) <= 1e-12):
        return Proportion(0., False, 0)
    d = g - 1

    def score(e):
        return np.sum(d / ((1 - e) + e * g))

    # Exact boundary solutions are allowed, including all-null and all-signal.
    if score(0) <= 0:
        return Proportion(0., True, 0)
    if score(1) >= 0:
        return Proportion(1., True, 0)
    lo, hi = 0., 1.
    for iteration in range(1, max_iterations + 1):
        mid = (lo + hi) / 2
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance:
            break
    return Proportion((lo + hi) / 2, True, iteration)


@dataclass(frozen=True)
class ProportionOpt:
    value: float
    identifiable: bool
    boundary: bool
    iterations: int


class OptProportion:
    """Li et al. (2025) moment-identity estimator eps_opt, bound to one (f1, cap) density.

    v(e, x) = (1 - g(x)) / ((1 - e) + e g(x)); T(e) = [Int v dF0 - mean_t v(e, Y_t)] /
    [Int v dF0 - mean_wm v(e, Y_wm)]; eps_opt is a root of T(e) - e on [lo, 1 - lo].
    Int v dF0 is a quadrature on y = sigmoid(s); the pure-watermark mean uses the pivots that
    fitted f1. Both depend only on (density, e), so one context serves every article.
    """

    def __init__(self, density, wm_pivots, lo=1e-3, points=20001, tolerance=1e-10, max_iterations=80):
        s = np.linspace(-27., 27., points)
        y = 1. / (1. + np.exp(-s))
        self.g_grid = np.asarray(density(y), float)
        self.w_grid = y * (1. - y) * (s[1] - s[0])
        self.w_grid[[0, -1]] /= 2
        self.g_wm = np.asarray(density(np.asarray(wm_pivots, float)), float)
        self.lo, self.tolerance, self.max_iterations = lo, tolerance, max_iterations

    def _v_mean(self, e, g, weights=None):
        v = (1. - g) / ((1. - e) + e * g)
        return float(np.sum(v * weights)) if weights is not None else float(v.mean())

    def _residual(self, e, g):
        null_mean = self._v_mean(e, self.g_grid, self.w_grid)
        denominator = null_mean - self._v_mean(e, self.g_wm)
        if abs(denominator) < 1e-14:
            return None
        return (null_mean - self._v_mean(e, g)) / denominator - e

    def estimate(self, g):
        g = np.asarray(g, float)
        if g.ndim != 1 or not len(g) or np.any(~np.isfinite(g)) or np.any(g <= 0):
            raise ValueError("Density ratios must be finite, positive, and nonempty.")
        if np.all(np.abs(g - 1) <= 1e-12) or np.all(np.abs(self.g_wm - 1) <= 1e-12):
            return ProportionOpt(0., False, False, 0)
        a, b = self.lo, 1. - self.lo
        fa, fb = self._residual(a, g), self._residual(b, g)
        if fa is None or fb is None:
            return ProportionOpt(0., False, False, 0)
        if fa * fb > 0:
            return ProportionOpt(a if abs(fa) <= abs(fb) else b, True, True, 0)
        for iteration in range(1, self.max_iterations + 1):
            mid = (a + b) / 2
            fm = self._residual(mid, g)
            if fm is None:
                return ProportionOpt(mid, False, False, iteration)
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if b - a <= self.tolerance:
                break
        return ProportionOpt((a + b) / 2, True, False, iteration)
