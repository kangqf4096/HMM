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
