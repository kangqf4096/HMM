"""SeedBS-NOT (Li, Li & Zhang 2024, Algorithm 1) adapted to the context-hash watermark; plan section 5.3.

Differences from the paper: no key alignment (windows pair with the same positions); the "new key"
replicates are iid U(0,1) shared by all windows of one replicate; a segment test supplies the
per-position score 1 - p(segment) so that the existing null calibration, labelling and any-FP code apply.
"""
import hashlib
import math
import os
from functools import lru_cache
from time import perf_counter

import numpy as np

DEFAULTS = dict(T=999, Tp=1999, a=2 ** -0.5, min_factor=2.5)
_BUDGET = 16_000_000  # elements of the (R, l, l) bootstrap tensor per chunk


def article_seed(global_seed, article_id):
    digest = hashlib.blake2b(f"{int(global_seed)}:{article_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


def seedbs_intervals(m, B, a=DEFAULTS["a"], min_factor=DEFAULTS["min_factor"]):
    """Seeded intervals (r, s] of Algorithm 1 as 0-based half-open [r, s); layers with l_k < min_factor*B dropped."""
    layers = max(1, math.ceil(math.log(m) / math.log(1 / a) - 1e-12))
    out = []
    for k in range(1, layers + 1):
        if k == 1:
            length, count = float(m), 1
        else:
            length = m * a ** (k - 1)
            count = 2 * math.ceil((1 / a) ** (k - 1) - 1e-12) - 1
        if length < min_factor * B:
            continue
        shift = (m - length) / (count - 1) if count > 1 else 0.
        for i in range(count):
            r = math.floor(i * shift + 1e-12)
            s = min(m, math.ceil(i * shift + length - 1e-12))
            if s - r >= 2 and (r, s) not in out:
                out.append((r, s))
    return out


def window_pvalues(y, B, T, rng):
    """Randomized window p-values, eq. (8) with the key alignment removed."""
    y = np.asarray(y, float)
    m = len(y)
    i = np.arange(m)
    lo, hi = np.maximum(i - B // 2, 0), np.minimum(i + B // 2, m - 1) + 1
    width = hi - lo

    def window_means(h):
        prefix = np.concatenate([np.zeros(h.shape[:-1] + (1,)), np.cumsum(h, axis=-1)], axis=-1)
        return (prefix[..., hi] - prefix[..., lo]) / width

    observed = window_means(np.log(y) + 1.)
    null = window_means(np.log(rng.random((T, m))) + 1.)
    return (1. + (observed[None, :] <= null).sum(axis=0)) / (T + 1.)


def ks_scan(p):
    """S(j), j = 1..l-1 (eq. 9) via S(j) = max_t |N_j(t) - j N(t) / l| / sqrt(l); integer maxima returned too."""
    p = np.asarray(p, float)
    d = _ks_integers(p[None, :])[0]
    return d / (len(p) * math.sqrt(len(p))), d


def _ks_integers(ps):
    """ps: (R, l). Returns (R, l-1) of max_t |l N_j(t) - j N(t)| for j = 1..l-1 (int)."""
    fast = _fast_kernels()
    if fast is not None and np.isfinite(ps).all():
        return fast.ks_integers(ps)
    return _ks_numpy(ps)


@lru_cache(maxsize=1)
def _load_fast():
    try:
        from . import _seedbs_fast
    except ImportError:
        return None
    return _seedbs_fast


def _fast_kernels():
    backend = os.environ.get("OPT_HMM_SEEDBS_BACKEND", "auto")
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError("OPT_HMM_SEEDBS_BACKEND must be auto, numpy or numba.")
    if backend == "numpy":
        return None
    fast = _load_fast()
    if backend == "numba" and fast is None:
        raise ImportError("Install the optional CPU accelerator: pip install -r requirements/cpu-fast.txt")
    return fast


def _ks_numpy(ps):
    """Reference tensor implementation, also used when Numba is unavailable."""
    r, l = ps.shape
    e = ps[:, None, :] <= ps[:, :, None]                    # [rep, t index, position]
    cs = np.cumsum(e, axis=2, dtype=np.int32 if l > 180 else np.int16)
    n = cs[:, :, -1:]
    j = np.arange(1, l, dtype=cs.dtype)
    d = cs[:, :, :l - 1] * cs.dtype.type(l) - n * j
    return np.abs(d).max(axis=1).astype(np.int64)


def block_bootstrap_pvalue(q, block, replicates, rng, observed_max):
    """Moving block bootstrap p-value, eq. (10); `observed_max` is the integer max_j max_t |l N_j - j N|."""
    q = np.asarray(q, float)
    l = len(q)
    block = min(block, l)
    count = math.ceil(l / block)
    chunk = max(1, _BUDGET // (l * l))
    exceed, done = 0, 0
    offsets = np.arange(block)
    while done < replicates:
        r = min(chunk, replicates - done)
        starts = rng.integers(0, l - block + 1, size=(r, count))
        index = (starts[:, :, None] + offsets).reshape(r, -1)[:, :l]
        samples = q[index]
        fast = _fast_kernels()
        if fast is not None and np.isfinite(samples).all():
            exceed += int(fast.ks_exceeds(samples, observed_max).sum())
        else:
            exceed += int((_ks_numpy(samples).max(axis=1) >= observed_max).sum())
        done += r
    return (1. + exceed) / (replicates + 1.)


def seedbs_precompute(y, B, seed, T=DEFAULTS["T"], Tp=DEFAULTS["Tp"], a=DEFAULTS["a"],
                      min_factor=DEFAULTS["min_factor"]):
    """Everything that depends on B but not on zeta: p-values, seeded intervals, KS argmax and bootstrap p."""
    started = perf_counter()
    y = np.asarray(y, float)
    if B % 2 or B < 2:
        raise ValueError("B must be an even integer >= 2.")
    rng = np.random.default_rng(seed)
    p = window_pvalues(y, B, T, rng)
    intervals, cut, ptilde = seedbs_intervals(len(y), B, a, min_factor), [], []
    for r, s in intervals:
        _, d = ks_scan(p[r:s])
        j = int(np.argmax(d)) + 1
        cut.append(r + j)
        ptilde.append(block_bootstrap_pvalue(p[r:s], B, Tp, rng, int(d.max())))
    return {"B": B, "m": len(y), "T": T, "Tp": Tp, "a": a, "min_factor": min_factor, "pvalues": p,
            "intervals": intervals, "cut": cut, "ptilde": ptilde, "seconds": perf_counter() - started}


def seedbs_select(pre, zeta):
    """Narrowest-over-threshold selection (Algorithm 1): an interval (r, s] contains cut c when r < c <= s."""
    alive = [i for i, v in enumerate(pre["ptilde"]) if v < zeta]
    found = []
    while alive:
        i = min(alive, key=lambda k: (pre["intervals"][k][1] - pre["intervals"][k][0], k))
        c = pre["cut"][i]
        found.append(c)
        alive = [k for k in alive if not pre["intervals"][k][0] < c <= pre["intervals"][k][1]]
    return sorted(found)


def _log_poisson_terms(s, ks):
    return -s + ks * math.log(s) - np.array([math.lgamma(k + 1.) for k in ks])


def segment_pvalue(s, length):
    """P(Gamma(L, 1) <= s) = P(Poisson(s) >= L); no 1 - x cancellation when the answer is small."""
    if s <= 0:
        return 0.
    if s <= length:
        ks = np.arange(length, length + 4 + int(60 + 12 * math.sqrt(length)))
        logs = _log_poisson_terms(s, ks)
        return float(min(1., math.exp(np.logaddexp.reduce(logs))))
    ks = np.arange(0, length)
    return float(min(1., max(0., 1. - math.exp(np.logaddexp.reduce(_log_poisson_terms(s, ks))))))


def segment_scores(y, changepoints):
    y = np.asarray(y, float)
    edges = [0] + sorted(int(c) for c in changepoints) + [len(y)]
    scores = np.empty(len(y))
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            scores[lo:hi] = 1. - segment_pvalue(float(-np.log(y[lo:hi]).sum()), hi - lo)
    return np.clip(scores, 0., 1.)


def seedbs_scores(y, B, zeta, seed, **kw):
    pre = seedbs_precompute(y, B, seed, **kw)
    cuts = seedbs_select(pre, zeta)
    return segment_scores(y, cuts), cuts
