"""Pooled EM for the global 2-parameter transition matrix and the initial distribution (plan 5.2).

f1 is frozen; only (a00, a11) and, for PE, pi1 are learned. Four ways to give the first-position law:
  pe  global free pi1 (from 0.5), exact EM          pd  pi_d = (1 - eps_d, eps_d), fixed, exact EM
  p1  pi_d solved from eps_d and A (appendix)       p2  stationary law of A (appendix)
All replicates (start points, bootstrap draws, grid points) run as one vectorised batch.
"""
from math import erfc, sqrt

import numpy as np

from ..hmm import forward_backward_two_state, stationary, two_state_log_likelihood

MODES = ("pe", "pd", "p1", "p2")
A_LO, A_HI = .50, .999
PI_LO, PI_HI = .01, .99
STARTS = ((.9, .9), (.99, .8), (.8, .99), (.6, .6))
CONSTRAINED_STARTS = ((.9, .9), (.6, .6), (.97, .97))
ALPHA_MIN = .05
GRID = np.r_[np.arange(50, 100) / 100, .995, .999]     # 0.50 ... 0.99 step 0.01, then 0.995, 0.999 (52 values)


def alpha_bar(a00, a11, n):
    """Mean over the n positions of lambda^(t-1), lambda = a00 + a11 - 1."""
    lam = np.asarray(a00 + a11 - 1., float)
    safe = np.where(lam < 1e-12, .5, lam)
    value = (1 - safe ** n) / (n * (1 - safe))
    return np.where(lam < 1e-12, 1. / n, value)


def first_prior(mode, a00, a11, pi_free, eps, n):
    """pi_1 for every (replicate, document): shape (R, D) or (R, 1). a00, a11, pi_free have shape (R,)."""
    a00, a11 = np.asarray(a00, float)[:, None], np.asarray(a11, float)[:, None]
    if mode == "pe":
        return np.asarray(pi_free, float)[:, None]
    if mode == "p2":
        return stationary(a00, a11)
    eps = np.asarray(eps, float)[None, :]
    if mode == "pd":
        return np.broadcast_to(eps, (len(a00), eps.shape[1]))
    star, ab = stationary(a00, a11), alpha_bar(a00, a11, n)
    solved = np.clip(star + (eps - star) / ab, 0., 1.)
    return np.where(ab < ALPHA_MIN, star, solved)


def _check(g, mode, eps):
    g = np.asarray(g, float)
    if g.ndim != 2 or mode not in MODES:
        raise ValueError("Need a (documents, positions) ratio array and a known mode.")
    if mode in ("pd", "p1") and (eps is None or len(eps) != len(g)):
        raise ValueError(f"mode {mode} needs one eps_hat per document.")
    return g


def pooled_log_likelihood(g, mode, a00, a11, pi_free=None, eps=None, weights=None, mask=None):
    """Exact pooled log-likelihood (relative to all-null) for parameter vectors of shape (R,)."""
    g = _check(g, mode, eps)
    a00, a11 = np.atleast_1d(np.asarray(a00, float)), np.atleast_1d(np.asarray(a11, float))
    pi_free = np.full(len(a00), .5) if pi_free is None else np.atleast_1d(np.asarray(pi_free, float))
    pi = first_prior(mode, a00, a11, pi_free, eps, g.shape[1])
    ll = two_state_log_likelihood(g[None], pi, a00[:, None], a11[:, None], mask)
    return _weighted(ll, weights)


def _weighted(per_document, weights):
    return per_document.sum(-1) if weights is None else (np.asarray(weights, float) * per_document).sum(-1)


def run_em(g, mode, starts=STARTS, eps=None, weights=None, constrained=False, max_iter=500, tol=1e-6,
           mask=None, history=False):
    """Batch EM over `starts` ((a00, a11) or (a00, a11, pi1) tuples) and, with `weights` of shape (R, D), replicates.

    Returns per-replicate arrays. pe/pd are exact EM (log-likelihood must not decrease). For p1/p2 the
    first-position term is ignored in the M-step, so each step is checked with the exact likelihood:
    on a drop larger than 1e-8 the previous (best) parameters are restored and `nonmonotone` is set.
    """
    g = _check(g, mode, eps)
    n_docs, n = g.shape
    starts = [tuple(s) + (.5,) * (3 - len(s)) for s in starts]
    a00 = np.array([s[0] for s in starts], float)
    a11 = np.array([s[1] for s in starts], float)
    pi1 = np.array([s[2] for s in starts], float)
    if constrained:
        a11 = a00.copy()
    r = len(starts)
    weights = np.ones((r, n_docs)) if weights is None else np.broadcast_to(np.asarray(weights, float), (r, n_docs))
    total_weight = weights.sum(-1)
    active = np.ones(r, bool)
    converged = np.zeros(r, bool)
    nonmonotone = np.zeros(r, bool)
    iterations = np.zeros(r, int)
    previous_ll = np.full(r, -np.inf)
    previous = (a00.copy(), a11.copy(), pi1.copy())
    trace = []
    ll = previous_ll
    for step in range(max_iter + 1):
        pi_first = first_prior(mode, a00, a11, pi1, eps, n)
        result = forward_backward_two_state(g[None], pi_first, a00[:, None], a11[:, None], mask)
        ll = (weights * result.log_likelihood).sum(-1)
        if history:
            trace.append(ll.copy())
        drop = active & (ll < previous_ll - 1e-8)
        if drop.any():
            nonmonotone |= drop
            a00[drop], a11[drop], pi1[drop] = (v[drop] for v in previous)
            ll = np.where(drop, previous_ll, ll)
            active &= ~drop
            converged |= drop
        done = active & (ll - previous_ll < tol)
        converged |= done
        active &= ~done
        iterations[active | done] = step
        if not active.any() or step == max_iter:
            break
        previous_ll = np.where(active, ll, previous_ll)
        previous = (a00.copy(), a11.copy(), pi1.copy())
        xi = np.einsum("rd,rdij->rij", weights, result.xi_sum)
        if constrained:
            new00 = new11 = (xi[:, 0, 0] + xi[:, 1, 1]) / xi.sum(axis=(1, 2))
        else:
            new00 = xi[:, 0, 0] / (xi[:, 0, 0] + xi[:, 0, 1])
            new11 = xi[:, 1, 1] / (xi[:, 1, 0] + xi[:, 1, 1])
        a00 = np.where(active, np.clip(new00, A_LO, A_HI), a00)
        a11 = np.where(active, np.clip(new11, A_LO, A_HI), a11)
        if mode == "pe":
            pi1 = np.where(active, np.clip((weights * result.gamma_first).sum(-1) / total_weight, PI_LO, PI_HI), pi1)
    out = {"a00": a00, "a11": a11, "pi1": pi1, "log_likelihood": ll, "iterations": iterations,
           "converged": converged, "nonmonotone": nonmonotone}
    if history:
        out["history"] = np.array(trace)
    return out


def grid_log_likelihood(g, mode, eps=None, grid=GRID, weights=None, mask=None):
    """Pooled log-likelihood on the (a00, a11) grid. PE profiles pi1 out exactly (concave, bisection).

    Returns (ll[len(grid), len(grid)], pi1[len(grid), len(grid)] or None), axes (a00, a11).
    """
    g = _check(g, mode, eps)
    n_docs, n = g.shape
    aa, bb = (x.ravel() for x in np.meshgrid(grid, grid, indexing="ij"))
    a00, a11 = aa[:, None], bb[:, None]
    shape = (len(grid), len(grid))
    if mode != "pe":
        return pooled_log_likelihood(g, mode, aa, bb, eps=eps, weights=weights, mask=mask).reshape(shape), None
    l0 = two_state_log_likelihood(g[None], 0., a00, a11, mask)        # first state fixed to 0
    l1 = two_state_log_likelihood(g[None], 1., a00, a11, mask)        # first state fixed to 1
    w = np.ones(n_docs) if weights is None else np.asarray(weights, float)

    def derivative(p):
        top = np.logaddexp(np.log1p(-p)[:, None] + l0, np.log(p)[:, None] + l1)
        return (w * (np.exp(l1 - top) - np.exp(l0 - top))).sum(-1)

    lo, hi = np.full(len(aa), PI_LO), np.full(len(aa), PI_HI)
    for _ in range(60):
        mid = (lo + hi) / 2
        up = derivative(mid) > 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    p = np.clip((lo + hi) / 2, PI_LO, PI_HI)
    top = np.logaddexp(np.log1p(-p)[:, None] + l0, np.log(p)[:, None] + l1)
    return (w * top).sum(-1).reshape(shape), p.reshape(shape)


def plateau(ll, grid=GRID, drop=1.92):
    """Grid points within `drop` of the maximum (half the chi2_1 95% cut): the flat part of the surface."""
    inside = ll >= ll.max() - drop
    rows, cols = np.nonzero(inside)
    return {"points": int(inside.sum()), "a00_range": [float(grid[rows.min()]), float(grid[rows.max()])],
            "a11_range": [float(grid[cols.min()]), float(grid[cols.max()])]}


def fit(g, mode, eps=None, grid_check=True, max_iter=500, tol=1e-6, mask=None):
    """Free-parameter EM from the four standard starts plus the grid optimum; returns the best fit and health."""
    g = _check(g, mode, eps)
    starts = [list(s) for s in STARTS]
    grid_info = None
    if grid_check:
        ll_grid, pi_grid = grid_log_likelihood(g, mode, eps, mask=mask)
        i, j = np.unravel_index(int(np.argmax(ll_grid)), ll_grid.shape)
        grid_info = {"best": [float(GRID[i]), float(GRID[j])], "log_likelihood": float(ll_grid[i, j]),
                     "pi1": None if pi_grid is None else float(pi_grid[i, j]), "plateau": plateau(ll_grid),
                     "surface": ll_grid}
        starts.append([GRID[i], GRID[j]] + ([] if pi_grid is None else [pi_grid[i, j]]))
    out = run_em(g, mode, starts, eps, None, False, max_iter, tol, mask)
    best = int(np.argmax(np.where(out["nonmonotone"], -np.inf, out["log_likelihood"])))
    standard = slice(0, len(STARTS))
    spread = max(np.ptp(out["a00"][standard]), np.ptp(out["a11"][standard]))
    result = {"mode": mode, "a00": float(out["a00"][best]), "a11": float(out["a11"][best]),
              "pi1": float(out["pi1"][best]) if mode == "pe" else None,
              "log_likelihood": float(out["log_likelihood"][best]), "iterations": int(out["iterations"][best]),
              "nonmonotone": bool(out["nonmonotone"][best]),
              "any_nonmonotone": bool(out["nonmonotone"].any()),
              "multimodal": bool(spread > .01), "start_spread": float(spread),
              "starts": {"a00": out["a00"].tolist(), "a11": out["a11"].tolist(),
                         "log_likelihood": out["log_likelihood"].tolist()},
              "at_bound": bool(min(out["a00"][best], out["a11"][best]) <= A_LO + 1e-9
                               or max(out["a00"][best], out["a11"][best]) >= A_HI - 1e-9)}
    result["lambda"] = result["a00"] + result["a11"] - 1
    result["stationary"] = float(stationary(result["a00"], result["a11"]))
    if grid_info:
        result["grid"] = {k: v for k, v in grid_info.items() if k != "surface"}
        result["grid_gap"] = result["log_likelihood"] - grid_info["log_likelihood"]     # >= -1e-6 expected
    return result, (grid_info["surface"] if grid_info else None)


def constrained_fit(g, mode, eps=None, mask=None, max_iter=500, tol=1e-6):
    """EM with a00 = a11 = a (same first-position rule); the null of the equal-retention test."""
    out = run_em(g, mode, CONSTRAINED_STARTS, eps, None, True, max_iter, tol, mask)
    best = int(np.argmax(np.where(out["nonmonotone"], -np.inf, out["log_likelihood"])))
    return {"a": float(out["a00"][best]), "log_likelihood": float(out["log_likelihood"][best])}


def equal_retention_test(free, constrained):
    """Likelihood ratio 2 (l_free - l_constrained) against chi2_1 (a diagnostic; EM is only a local optimum)."""
    stat = max(0., 2 * (free["log_likelihood"] - constrained["log_likelihood"]))
    return {"statistic": stat, "p_value": erfc(sqrt(stat / 2))}


def bootstrap(g, mode, fitted, eps=None, draws=500, seed=0, max_iter=300, tol=1e-6, mask=None):
    """Resample documents, rerun EM warm-started at the best fit; percentile intervals (95%)."""
    g = _check(g, mode, eps)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=draws).astype(float)
    start = [fitted["a00"], fitted["a11"]] + ([fitted["pi1"]] if mode == "pe" else [])
    out = run_em(g, mode, [start] * draws, eps, weights, False, max_iter, tol, mask)
    keep = ~out["nonmonotone"]
    summary = {}
    for name, values in (("a00", out["a00"]), ("a11", out["a11"]), ("diff", out["a00"] - out["a11"]),
                         ("lambda", out["a00"] + out["a11"] - 1),
                         ("stationary", stationary(out["a00"], out["a11"]))) + \
            ((("pi1", out["pi1"]),) if mode == "pe" else ()):
        v = values[keep]
        summary[name] = {"mean": float(v.mean()), "ci95": np.quantile(v, [.025, .975]).tolist()}
    summary["draws"] = int(keep.sum())
    return summary
