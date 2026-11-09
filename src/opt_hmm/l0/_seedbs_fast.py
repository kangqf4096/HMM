"""Optional compiled, integer-exact KS scans without bootstrap comparison tensors.

Only distinct thresholds matter. Their sorted group ends give the total CDF
counts, including ties. All remaining arithmetic is integer arithmetic.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def ks_integers(ps):
    rows, length = ps.shape
    out = np.zeros((rows, length - 1), dtype=np.int64)
    for row in range(rows):
        thresholds = np.sort(ps[row])
        for end in range(length - 1):
            if thresholds[end] == thresholds[end + 1]:
                continue
            # The final threshold includes every position and contributes zero.
            threshold, total, count = thresholds[end], end + 1, 0
            for j in range(1, length):
                count += int(ps[row, j - 1] <= threshold)
                value = abs(length * count - j * total)
                if value > out[row, j - 1]:
                    out[row, j - 1] = value
    return out


@njit(cache=True)
def ks_exceeds(ps, observed_max):
    """Exactly max_j KS_integer >= observed_max, stopping once it is proven."""
    rows, length = ps.shape
    out = np.zeros(rows, dtype=np.bool_)
    for row in range(rows):
        if observed_max <= 0:
            out[row] = True
            continue
        thresholds = np.sort(ps[row])
        for end in range(length - 1):
            if thresholds[end] == thresholds[end + 1]:
                continue
            threshold, total, count = thresholds[end], end + 1, 0
            for j in range(1, length):
                count += int(ps[row, j - 1] <= threshold)
                if abs(length * count - j * total) >= observed_max:
                    out[row] = True
                    break
            if out[row]:
                break
    return out
