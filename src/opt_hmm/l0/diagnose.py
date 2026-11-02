"""Replay-free diagnostics (plan 4.2 split-half, 4.3 boundedness). They never touch L0 main-table numbers."""
import numpy as np

from .emission import fit_emission


def _grid(points=20001):
    s = np.linspace(-27., 27., points)
    y = 1. / (1. + np.exp(-s))
    return y, y * (1. - y) * (s[1] - s[0])


def split_half(articles_y, emissions, config, repeats=200, seed=0):
    """Fit f1 on two disjoint random halves of the calibration articles; report TV and KS of the fits."""
    if len(articles_y) < 2:
        return {"skipped": "need at least 2 calibration articles"}
    y, weight = _grid()
    rng = np.random.default_rng(seed)
    half = len(articles_y) // 2
    out = {"calibration_articles": len(articles_y), "half_size": half, "repeats": repeats}
    for name in emissions:
        tv, ks = [], []
        for _ in range(repeats):
            order = rng.permutation(len(articles_y))
            fits = [fit_emission(name, np.concatenate([articles_y[i] for i in order[a:b]]), config)
                    for a, b in ((0, half), (half, 2 * half))]
            f1, f2 = (np.asarray(f(y), float) for f in fits)
            tv.append(float(.5 * np.sum(np.abs(f1 - f2) * weight)))
            ks.append(float(np.max(np.abs(np.cumsum(f1 * weight) - np.cumsum(f2 * weight)))))
        out[name] = {k: {"mean": float(np.mean(v)), "p05": float(np.quantile(v, .05)),
                         "p50": float(np.median(v)), "p95": float(np.quantile(v, .95))}
                     for k, v in (("tv", tv), ("ks", ks))}
    return out


def tail_stats(log_f):
    log_f = np.asarray(log_f, float)
    return {"n": int(len(log_f)), "max": float(log_f.max()), "p99.9": float(np.quantile(log_f, .999)),
            "p99.99": float(np.quantile(log_f, .9999))}


def boundedness(emissions, calibration_y, scenario_y):
    """max log f1(y), its high quantiles per emission x scenario, and the calibration-set density extremes."""
    result = {}
    for name, density in emissions.items():
        cal = np.asarray(calibration_y, float)
        q = np.quantile(cal, [.999, .9999])
        result[name] = {
            "calibration": {"max_f1": float(np.max(density(cal))), "y_p99.9": float(q[0]), "y_p99.99": float(q[1]),
                            "f1_at_y_p99.9": float(density(np.array([q[0]]))[0]),
                            "f1_at_y_p99.99": float(density(np.array([q[1]]))[0])},
            "log_f1_by_scenario": {s: tail_stats(np.log(density(np.asarray(y, float))))
                                   for s, y in scenario_y.items() if len(y)}}
    return result
