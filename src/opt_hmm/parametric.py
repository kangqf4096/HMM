"""Parametric alternatives to binned emission densities for the pivot law.

The alternative hypothesis gives the pivot Y = U_{t, X_t} a density f1 on
(0, 1); the null keeps f0 == 1. Everything here estimates f1, so the returned
object can be plugged into Detector as a density ratio against f0 == 1.

Families fitted on y directly return f1(y). Families fitted on X = -log(y)
return f_X(-log y) / y, the same density after the Jacobian. Every returned
density integrates to one on (0, 1).

Fitting is closed-form method of moments, plus a short EM for the Beta
mixture. numpy only: the project does not depend on scipy.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .pivot_diagnostics import S2_EDGES

_EPS = np.finfo(float).tiny
_LOG_FLOOR = -700.0
_LOG_CEIL = 700.0

#: Families fitted on y directly.
Y_FAMILIES = ("beta", "beta_mixture", "truncnorm", "logitnorm")
#: Families fitted on X = -log(y), then mapped back through the Jacobian.
X_FAMILIES = ("gamma", "weibull", "lognormal")
FAMILIES = Y_FAMILIES + X_FAMILIES


def _clip(y):
    y = np.asarray(y, dtype=float)
    return np.clip(y, 1e-12, 1.0 - 1e-12)


def _from_log(log_density):
    return np.exp(np.clip(log_density, _LOG_FLOOR, _LOG_CEIL))


def _lgamma(x):
    return float(math.lgamma(max(float(x), _EPS)))


def _beta_logpdf(y, a, b):
    y = _clip(y)
    return (a - 1.0) * np.log(y) + (b - 1.0) * np.log1p(-y) - (_lgamma(a) + _lgamma(b) - _lgamma(a + b))


def _norm_cdf(z):
    """Abramowitz & Stegun 26.2.17; absolute error below 7.5e-8."""
    z = np.asarray(z, dtype=float)
    b = (0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
    p = 0.2316419
    t = 1.0 / (1.0 + p * np.abs(z))
    poly = b[0] + t * (b[1] + t * (b[2] + t * (b[3] + t * b[4])))
    upper_tail = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) * t * poly
    return np.where(z >= 0.0, 1.0 - upper_tail, upper_tail)


def _norm_pdf(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------------------
# Method-of-moments fits. Each returns a plain dict so it serialises to JSON.
# --------------------------------------------------------------------------

def _fit_beta(y):
    m = float(np.mean(y))
    v = max(float(np.var(y)), 1e-9)
    common = m * (1.0 - m) / v - 1.0
    if not np.isfinite(common) or common <= 0.0:
        return {"a": 1.0, "b": 1.0}
    return {"a": float(max(m * common, 1e-3)), "b": float(max((1.0 - m) * common, 1e-3))}


def _fit_gamma(x):
    m, v = float(np.mean(x)), float(np.var(x))
    if not np.isfinite(m) or m <= 0.0 or not np.isfinite(v) or v <= 0.0:
        return {"k": 1.0, "theta": float(max(m, 1e-12))}
    return {"k": float(max(m * m / v, 1e-3)), "theta": float(max(v / m, 1e-12))}


def _fit_weibull(x):
    m, s = float(np.mean(x)), float(np.std(x))
    if not np.isfinite(m) or m <= 0.0 or not np.isfinite(s) or s <= 0.0:
        return {"k": 1.0, "lam": float(max(m, 1e-12))}
    shape = max((s / m) ** -1.086, 1e-3)
    return {"k": float(shape), "lam": float(max(m / math.gamma(1.0 + 1.0 / shape), 1e-12))}


def _fit_lognormal(x):
    lx = np.log(np.maximum(x, 1e-12))
    return {"mu": float(np.mean(lx)), "sigma": float(max(np.std(lx), 1e-3))}


def _fit_truncnorm(y):
    return {"mu": float(np.mean(y)), "sigma": float(max(np.std(y), 1e-3))}


def _fit_logitnorm(y):
    z = np.log(y) - np.log1p(-y)
    return {"mu": float(np.mean(z)), "sigma": float(max(np.std(z), 1e-3))}


def _fit_beta_mixture(y, components=3, iterations=80):
    """EM with per-component moment updates; means stay in (0, 1)."""
    y = _clip(y)
    n = len(y)
    if n == 0:
        raise ValueError("Cannot fit an empty density.")
    components = int(max(1, min(components, max(1, n // 10))))
    if components == 1:
        beta = _fit_beta(y)
        return {"weights": [1.0], "a": [beta["a"]], "b": [beta["b"]]}
    order = np.argsort(y)
    chunks = np.array_split(order, components)
    weights = np.array([max(len(c), 1) / n for c in chunks], dtype=float)
    weights /= weights.sum()
    a = np.ones(components)
    b = np.ones(components)
    for index, chunk in enumerate(chunks):
        if len(chunk):
            beta = _fit_beta(y[chunk])
            a[index], b[index] = beta["a"], beta["b"]
    for _ in range(iterations):
        log_joint = np.log(np.maximum(weights, 1e-12))[:, None] + np.stack(
            [_beta_logpdf(y, a[i], b[i]) for i in range(components)])
        peak = log_joint.max(axis=0)
        resp = np.exp(log_joint - peak)
        resp /= np.maximum(resp.sum(axis=0), _EPS)
        mass = resp.sum(axis=1) + 1e-9
        weights = np.maximum(mass / n, 1e-6)
        weights /= weights.sum()
        for i in range(components):
            w = resp[i]
            mean = float((w * y).sum() / mass[i])
            var = max(float((w * (y - mean) ** 2).sum() / mass[i]), 1e-9)
            common = mean * (1.0 - mean) / var - 1.0
            if not np.isfinite(common) or common <= 0.0:
                a[i], b[i] = 1.0, 1.0
            else:
                a[i] = float(max(mean * common, 1e-3))
                b[i] = float(max((1.0 - mean) * common, 1e-3))
    return {"weights": weights.tolist(), "a": a.tolist(), "b": b.tolist()}


# --------------------------------------------------------------------------
# Densities on y.
# --------------------------------------------------------------------------

def _beta_log_density(y, params):
    return _beta_logpdf(y, params["a"], params["b"])


def _beta_mixture_log_density(y, params):
    weights = np.asarray(params["weights"], float)
    a = np.asarray(params["a"], float)
    b = np.asarray(params["b"], float)
    terms = np.log(np.maximum(weights, 1e-12))[:, None] + np.stack(
        [_beta_logpdf(y, a[i], b[i]) for i in range(len(weights))])
    peak = terms.max(axis=0)
    return peak + np.log(np.maximum(np.exp(terms - peak).sum(axis=0), _EPS))


def _truncnorm_log_density(y, params):
    mu, sigma = params["mu"], params["sigma"]
    z = (y - mu) / sigma
    mass = float(_norm_cdf((1.0 - mu) / sigma) - _norm_cdf((0.0 - mu) / sigma))
    return (-0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
            - math.log(sigma) - math.log(max(mass, 1e-12)))


def _logitnorm_log_density(y, params):
    mu, sigma = params["mu"], params["sigma"]
    z = np.log(y) - np.log1p(-y)
    return (-0.5 * ((z - mu) / sigma) ** 2 - math.log(sigma)
            - 0.5 * math.log(2.0 * math.pi) - np.log(y) - np.log1p(-y))


def _gamma_density_on_y(y, params):
    k, theta = params["k"], params["theta"]
    x = -np.log(y)
    log_density = ((k - 1.0) * np.log(np.maximum(x, _EPS))
                   - x / theta - k * math.log(theta) - _lgamma(k))
    return _from_log(log_density) / y


def _weibull_density_on_y(y, params):
    k, lam = params["k"], params["lam"]
    x = -np.log(y)
    scaled = np.maximum(x, _EPS) / lam
    log_density = math.log(k) - math.log(lam) + (k - 1.0) * np.log(scaled) - scaled ** k
    return _from_log(log_density) / y


def _lognormal_density_on_y(y, params):
    mu, sigma = params["mu"], params["sigma"]
    x = np.maximum(-np.log(y), 1e-12)
    log_density = (-0.5 * ((np.log(x) - mu) / sigma) ** 2
                   - np.log(x) - math.log(sigma) - 0.5 * math.log(2.0 * math.pi))
    return _from_log(log_density) / y


_Y_LOG_DENSITIES = {
    "beta": _beta_log_density,
    "beta_mixture": _beta_mixture_log_density,
    "truncnorm": _truncnorm_log_density,
    "logitnorm": _logitnorm_log_density,
}
_X_DENSITIES = {
    "gamma": _gamma_density_on_y,
    "weibull": _weibull_density_on_y,
    "lognormal": _lognormal_density_on_y,
}


@dataclass
class ParametricDensity:
    """A fitted f1(y) on (0, 1), positive everywhere and capped on request."""

    family: str
    params: dict
    cap: float | None = None
    sample_count: int = 0

    @classmethod
    def fit(cls, values, family="beta", cap=None):
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}.")
        y = _clip(values)
        if not len(y):
            raise ValueError("Cannot fit an empty density.")
        if family == "beta":
            params = _fit_beta(y)
        elif family == "beta_mixture":
            params = _fit_beta_mixture(y)
        elif family == "truncnorm":
            params = _fit_truncnorm(y)
        elif family == "logitnorm":
            params = _fit_logitnorm(y)
        else:
            x = -np.log(y)
            params = {"gamma": _fit_gamma, "weibull": _fit_weibull,
                      "lognormal": _fit_lognormal}[family](x)
        return cls(family, params, cap, len(y))

    def __call__(self, y):
        y = _clip(y)
        if self.family in _Y_LOG_DENSITIES:
            density = _from_log(_Y_LOG_DENSITIES[self.family](y, self.params))
        else:
            density = _X_DENSITIES[self.family](y, self.params)
        density = np.maximum(density, _EPS)
        if self.cap is not None:
            density = np.minimum(density, float(self.cap))
        return density

    def to_dict(self):
        return {"family": self.family, "params": self.params,
                "cap": self.cap, "sample_count": self.sample_count}


@dataclass
class ParametricConditionalDensity:
    """Fixed S2 groups with a parametric f1 per group, shrunk to the global fit.

    This mirrors ConditionalHistogram: the same S2 edges and the same
    prior_count shrinkage, replacing the 25-bin histogram with a parametric
    density. Groups with fewer than ``min_count`` pivots fall back to the
    global fit rather than fitting an unstable curve.
    """

    family: str
    global_density: ParametricDensity
    densities: list
    counts: list
    prior_count: float = 50.0
    min_count: int = 30
    group: np.ndarray = field(default=None, repr=False)

    @classmethod
    def fit(cls, y, s2, family="beta", prior_count=50.0, min_count=30, cap=None):
        y = _clip(y)
        s2 = np.asarray(s2, dtype=float)
        if y.shape != s2.shape or not len(y):
            raise ValueError("Need matched pivots and S2 values.")
        group = np.minimum(np.searchsorted(S2_EDGES, s2, side="right") - 1,
                           len(S2_EDGES) - 2)
        global_density = ParametricDensity.fit(y, family, cap)
        densities, counts = [], []
        for index in range(len(S2_EDGES) - 1):
            selected = y[group == index]
            counts.append(int(len(selected)))
            if len(selected) >= min_count:
                densities.append(ParametricDensity.fit(selected, family, cap))
            else:
                densities.append(None)
        return cls(family, global_density, densities, counts, prior_count, min_count, group)

    def __call__(self, y, s2):
        index = min(max(int(np.searchsorted(S2_EDGES, s2, side="right") - 1), 0),
                    len(self.densities) - 1)
        pooled = float(np.asarray(self.global_density(y)).reshape(-1)[0])
        local = self.densities[index]
        if local is None:
            return max(pooled, _EPS)
        fitted = float(np.asarray(local(y)).reshape(-1)[0])
        weight = self.counts[index] / (self.counts[index] + self.prior_count)
        return max(weight * fitted + (1.0 - weight) * pooled, _EPS)

    def to_dict(self):
        return {"s2_edges": S2_EDGES.tolist(), "family": self.family,
                "counts": self.counts, "prior_count": self.prior_count,
                "min_count": self.min_count, "fit_role": "calibration only",
                "global": self.global_density.to_dict(),
                "densities": [d.to_dict() if d is not None else None for d in self.densities]}
