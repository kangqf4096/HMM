"""Continuous Gumbel-max pivots and positive normalized histogram densities."""
from dataclasses import dataclass

import numpy as np


def check_pivots(y):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)) or np.any((y <= 0) | (y >= 1)):
        raise ValueError("Pivots must be a finite 1D array strictly inside (0, 1).")
    return y


def probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not len(p) or np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Invalid next-token probability vector.")
    if not np.isclose(p.sum(), 1):
        raise ValueError("Probabilities must sum to one.")
    return p[p > 0] / p.sum()


def theoretical_density(y, p):
    y, p = check_pivots(y), probabilities(p)
    return np.exp(np.log(y[:, None]) * (1 / p - 1)).sum(axis=1)


def theoretical_cdf(y, p):
    y, p = check_pivots(y), probabilities(p)
    return (p * np.exp(np.log(y[:, None]) / p)).sum(axis=1)


def sample_signal(rng, count, p):
    p = probabilities(p)
    chosen = rng.choice(len(p), size=count, p=p)
    # A finite-precision safeguard; not a statistical truncation rule.
    return np.clip(rng.random(count) ** p[chosen], np.nextafter(0., 1.), np.nextafter(1., 0.))


@dataclass
class Histogram:
    density: np.ndarray
    sample_count: int = 0
    pseudocount: float = 0.5

    @classmethod
    def fit(cls, y, bins=50, pseudocount=0.5):
        y = check_pivots(y)
        if not len(y) or bins < 1 or pseudocount <= 0:
            raise ValueError("Need nonempty calibration pivots, positive bins and pseudocount.")
        counts, _ = np.histogram(y, bins=bins, range=(0, 1))
        density = bins * (counts + pseudocount) / (len(y) + bins * pseudocount)
        return cls(density, len(y), pseudocount)

    def __post_init__(self):
        self.density = np.asarray(self.density, dtype=float)
        if (self.density.ndim != 1 or not len(self.density)
                or np.any(~np.isfinite(self.density)) or np.any(self.density <= 0)
                or not np.isclose(self.density.mean(), 1)):
            raise ValueError("Histogram density must be positive and integrate to one.")

    def __call__(self, y):
        y = check_pivots(y)
        index = np.minimum((y * len(self.density)).astype(int), len(self.density) - 1)
        return self.density[index]

    def to_dict(self):
        return {"density": self.density.tolist(), "sample_count": self.sample_count,
                "pseudocount": self.pseudocount}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
