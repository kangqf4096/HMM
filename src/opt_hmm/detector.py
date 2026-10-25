"""Fitted per-document inference accepts pivots and mask, never ground truth."""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .emissions import check_pivots
from .hmm import forward_backward, forward_log_likelihood
from .proportion import estimate_proportion

DEFAULT_RHOS = (0., .5, .8, .9, .95, .98, .99)
METHODS = ("independent", "fixed_e", "opt_hmm", "fixed_rho", "window")


@dataclass
class Detection:
    scores: np.ndarray
    epsilon: float
    rho: float
    identifiable: bool
    seconds: float
    log_likelihood: float | None = None
    stage_seconds: dict = field(default_factory=dict)


class Detector:
    def __init__(self, density, method="opt_hmm", rhos=DEFAULT_RHOS,
                 fixed_e=.5, fixed_rho=.95, window=20):
        if method not in METHODS and method != "oracle":
            raise ValueError(f"Unknown method {method}")
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
        stages = {"density_lookup": perf_counter() - started}
        phase = perf_counter()
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


def intervals(labels):
    """Zero-based half-open intervals. Unavailable positions must be False."""
    values = np.asarray(labels, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Labels must be 1D.")
    edges = np.diff(np.r_[False, values, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
