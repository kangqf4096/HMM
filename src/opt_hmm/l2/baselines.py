"""Pivot-only localization baselines from the cached paper library, scored on pivots Y_t alone (p_t = 1 - Y_t).

SPOT: Algorithm 1 of docs/2608.14906v2 (grid u in [0.005, 0.98], T_hat(u) = (1 - eps_hat) n^-u / max(S_hat(u), 1/n), first u
with T_hat <= lambda, flag Y > 1 - n^-u). tau(u) increases with u, so the tokens above tau(u_hat) are exactly those with
min_{u: Y_t > tau(u)} T_hat(u) <= lambda; `SpotCurve.scores` is the per-position score 1 / (1 + that minimum), and
"score >= 1 / (1 + lambda)" is Algorithm 1 itself. The paper's fallback to the largest scanned u when no u qualifies is
not applied: it flags about one null token per article whatever lambda is, which no threshold on the score can undo.

win_trgof: Tr-GoF (Li et al. 2025, Eqs. 5-8) in a centred window of B + 1 tokens, truncated at c+ = 1 / m, the upper end
of the admissible range 0 <= c+ <= 1/n. win_ars: the Aaronson sum of -log(1 - Y_t) in the same window with its exact
Gamma(m, 1) tail probability, so shorter windows at the text ends stay on the same null scale. Both adapt a detection
statistic to localization by thresholding the window score, as the window baselines of 2511.06645 (B.5) do.
"""
from math import ceil, log

import numpy as np

U_MIN, U_MAX = .005, .98
TAIL_CUT = .85
EPS = 1e-12


def spot_grid(n):
    return np.linspace(U_MIN, U_MAX, max(20, ceil(3.5 * log(n) ** 1.5)))


def tail_eps(y, cut=TAIL_CUT):
    """Calibration-free surviving fraction of the SPOT simulations: clip((P_n(Y > cut) - (1 - cut)) / cut, 0, 1)."""
    return np.clip(((np.asarray(y) > cut).mean(axis=-1) - (1. - cut)) / cut, 0., 1.)


class SpotCurve:
    """T_hat(u) and the tail counts #{Y_t > tau(u)} of every article of a (D, n) pivot array."""

    def __init__(self, y, eps):
        y = np.asarray(y, float)
        self.y, self.n = y, y.shape[1]
        self.u = spot_grid(self.n)
        self.cut = 1. - self.n ** -self.u
        ys = np.sort(y, axis=1)
        self.count = self.n - np.stack([np.searchsorted(row, self.cut, side="right") for row in ys])
        mass = np.maximum(self.count / self.n, 1. / self.n)
        self.curve = (1. - np.asarray(eps, float)[:, None]) * self.n ** -self.u[None, :] / mass

    def scores(self):
        """Per-position score without the fallback; 0 for tokens at or below tau(u_1)."""
        best = np.concatenate([np.full((len(self.curve), 1), np.inf), np.minimum.accumulate(self.curve, axis=1)], axis=1)
        j = np.stack([np.searchsorted(self.cut, row, side="left") for row in self.y])
        return 1. / (1. + np.take_along_axis(best, j, axis=1))


def _windows(a, B, fill):
    """(D, n, B + 1) centred windows [t - B/2, t + B/2] of a (D, n) array; `fill` outside the text."""
    half = B // 2
    return np.lib.stride_tricks.sliding_window_view(np.pad(a, ((0, 0), (half, half)), constant_values=fill), B + 1, axis=1)


def _chunked(y, B, window_score, chunk=64):
    out = np.empty(np.shape(y))
    for start in range(0, len(y), chunk):
        out[start:start + chunk] = window_score(_windows(y[start:start + chunk], B, np.nan))
    return out


def _bounded(x):
    return x / (1. + x)


def _trgof(y_window, s):
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.sort(1. - y_window, axis=-1)
        m = np.isfinite(p).sum(axis=-1)
        k = np.arange(1, p.shape[-1] + 1)
        u = k / m[..., None]
        v = np.clip(p, EPS, 1. - EPS)
        valid = (k <= m[..., None]) & (v < u) & (u < 1.)
        if s == 1:
            div = u * np.log(u / v) + (1. - u) * np.log((1. - u) / (1. - v))
        else:
            div = (u - v) ** 2 / (2. * v * (1. - v))
        first = np.maximum((p <= 1. / m[..., None]).sum(axis=-1) - 1, 0)
        allowed = valid & (k - 1 >= first[..., None])
        return _bounded(m * np.where(allowed, div, 0.).max(axis=-1))


def window_trgof(y, B, s):
    """Tr-GoF score m * S+_m(s) of the centred window around every position, mapped to [0, 1) by x / (1 + x)."""
    if s not in (1, 2):
        raise ValueError("Only the Berk-Jones (s = 1) and Higher Criticism (s = 2) members are implemented.")
    return _chunked(np.asarray(y, float), B, lambda w: _trgof(w, s))


def _gamma_tail(y_window):
    a = -np.log1p(-np.clip(y_window, 0., 1. - EPS))
    m = np.isfinite(a).sum(axis=-1)
    total = np.nansum(a, axis=-1)
    k = np.arange(a.shape[-1])
    log_factorial = np.concatenate([[0.], np.cumsum(np.log(np.arange(1, a.shape[-1])))])
    log_total = np.log(np.maximum(total, 1e-300))[..., None]
    terms = np.where(k < m[..., None], k * log_total - log_factorial, -np.inf)
    top = terms.max(axis=-1, keepdims=True)
    log_q = -total + top[..., 0] + np.log(np.exp(terms - top).sum(axis=-1))
    return _bounded(np.maximum(-log_q, 0.))


def window_aaronson(y, B):
    """-log of the exact Gamma(m, 1) upper-tail probability of the Aaronson sum in each centred window, as x / (1 + x)."""
    return _chunked(np.asarray(y, float), B, _gamma_tail)
