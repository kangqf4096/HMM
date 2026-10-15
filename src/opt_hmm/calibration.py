"""Document any-FP calibration using a strict empirical rank threshold."""
from dataclasses import dataclass
from math import ceil, sqrt
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Threshold:
    value: float
    count: int
    rank: int
    alpha: float
    finite_rank: bool

    def predict(self, scores, mask=None):
        scores = np.asarray(scores, float)
        mask = np.ones(len(scores), bool) if mask is None else np.asarray(mask, bool)
        return (scores > self.value) & mask


def calibrate(scores, alpha=.05):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not 0 < alpha < 1:
        raise ValueError("Need nonempty document scores and alpha in (0,1).")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Calibration scores must be inside [0,1].")
    rank = ceil((len(scores) + 1) * (1 - alpha))
    finite = rank <= len(scores)
    value = float(np.sort(scores)[rank - 1]) if finite else 1.
    return Threshold(value, len(scores), rank, alpha, finite)


def wilson(successes, count, confidence=.95):
    if count == 0:
        return [None, None]
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - radius), min(1., center + radius)]
