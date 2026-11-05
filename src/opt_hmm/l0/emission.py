"""f1 families under one density cap K, fitted on pure-watermark calibration pivots only."""
import numpy as np

from ..emissions import Histogram, check_pivots
from ..parametric import FAMILIES as PARAM_FAMILIES, ParametricDensity


class CappedHistogram:
    """Histogram density with the same upper bound as the parametric families."""

    def __init__(self, histogram, cap):
        self.histogram, self.cap = histogram, cap

    def __call__(self, y):
        density = self.histogram(y)
        return density if self.cap is None else np.minimum(density, float(self.cap))

    def to_dict(self):
        return {"family": "histogram", **self.histogram.to_dict(), "cap": self.cap}


class ECDFPCHIP:
    """Density of a monotone cubic (PCHIP) interpolant of the empirical CDF through equal-mass knots.

    Knots: (0,0), the ceil(k n / segments)-th order statistics at height k/segments, (1,1).
    Each segment carries mass 1/segments, so tails get narrow, dense segments. The density is the
    analytic derivative (piecewise quadratic); it integrates to exactly one.
    """
    FLOOR = 1e-9

    def __init__(self, x, cdf):
        x, cdf = np.asarray(x, float), np.asarray(cdf, float)
        if x.ndim != 1 or len(x) < 3 or x.shape != cdf.shape or x[0] != 0 or x[-1] != 1                 or np.any(np.diff(x) <= 0) or np.any(np.diff(cdf) <= 0) or cdf[0] != 0 or cdf[-1] != 1:
            raise ValueError("ECDF knots must be strictly increasing, from (0,0) to (1,1).")
        self.x, self.cdf = x, cdf
        self.slope = self._derivatives(x, cdf)

    @classmethod
    def fit(cls, y, segments=400):
        y = np.sort(check_pivots(y))
        if segments < 2 or len(y) < segments:
            raise ValueError("Need at least as many pivots as ECDF segments.")
        k = np.arange(1, segments)
        x = y[np.ceil(k * len(y) / segments).astype(int) - 1]
        keep = np.r_[True, np.diff(x) > 0]          # ties would give zero-width segments
        k, x = k[keep], x[keep]
        return cls(np.r_[0., x, 1.], np.r_[0., k / segments, 1.])

    @staticmethod
    def _derivatives(x, cdf):
        h, delta = np.diff(x), np.diff(cdf) / np.diff(x)
        d = np.empty(len(x))
        w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
        d[1:-1] = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])

        def end(h0, h1, d0, d1):
            e = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
            if e <= 0:
                return 0.
            return min(e, 3 * d0) if d0 * d1 <= 0 else e
        d[0] = end(h[0], h[1], delta[0], delta[1])
        d[-1] = end(h[-1], h[-2], delta[-1], delta[-2])
        return d

    def __call__(self, y):
        y = check_pivots(y)
        i = np.clip(np.searchsorted(self.x, y, side="right") - 1, 0, len(self.x) - 2)
        h = self.x[i + 1] - self.x[i]
        t = (y - self.x[i]) / h
        delta = (self.cdf[i + 1] - self.cdf[i]) / h
        f = (6 * t * (1 - t) * delta + (3 * t * t - 4 * t + 1) * self.slope[i]
             + (3 * t * t - 2 * t) * self.slope[i + 1])
        return np.maximum(f, self.FLOOR)

    def to_dict(self):
        return {"family": "ecdf", "x": self.x.tolist(), "cdf": self.cdf.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["x"], data["cdf"])


class Capped:
    """Any base density under an upper bound `cap` (None = none), optionally with a power-law tail.

    tail = {"y0": y0, "beta": beta}: for y > y0 the density is f(y0) ((1-y)/(1-y0))^(-beta), i.e.
    log f is linear in s = -log(1-y) with slope beta, then bounded by `cap`.
    """

    def __init__(self, base, cap=None, tail=None):
        if cap is not None and cap <= 1:
            raise ValueError("cap must exceed the null density 1 or be None.")
        if tail is not None and not (0 < tail["y0"] < 1 and 0 <= tail["beta"] < 1):
            raise ValueError("tail needs y0 in (0,1) and beta in [0,1).")
        self.base, self.cap, self.tail = base, cap, tail
        self._f_y0 = float(base(np.array([tail["y0"]]))[0]) if tail else None

    def raw(self, y):
        y = check_pivots(y)
        f = np.asarray(self.base(y), float)
        if self.tail:
            y0, beta = self.tail["y0"], self.tail["beta"]
            far = y > y0
            f = np.where(far, self._f_y0 * ((1 - y) / (1 - y0)) ** (-beta), f)
        return f

    def __call__(self, y):
        f = self.raw(y)
        return f if self.cap is None else np.minimum(f, float(self.cap))

    def to_dict(self):
        return {"family": "capped", "base": self.base.to_dict(), "cap": self.cap, "tail": self.tail}

    @classmethod
    def from_dict(cls, data):
        base = data["base"]
        if base["family"] == "ecdf":
            base = ECDFPCHIP.from_dict(base)
        elif base["family"] in PARAM_FAMILIES:
            base = ParametricDensity(base["family"], base["params"], None, base.get("sample_count", 0))
        else:
            base = Histogram.from_dict({k: v for k, v in base.items() if k != "family"})
        return cls(base, data["cap"], data["tail"])


# v2 families: name -> (kind, size). The seven parametric families are the candidates for param*.
V2_FAMILIES = {"hist25": ("hist", 25), "hist200": ("hist", 200), "ecdf400": ("ecdf", 400),
               **{name: ("param", None) for name in PARAM_FAMILIES}}


def fit_base(family, y, config):
    """Uncapped f1 for a v2 family, fitted once and then frozen."""
    kind, size = V2_FAMILIES[family]
    if kind == "hist":
        return Histogram.fit(y, size, config.get("pseudocount", .5))
    if kind == "param":
        return ParametricDensity.fit(y, family, None)
    return ECDFPCHIP.fit(y, size)


def tail_beta(ecdf, y0):
    """Slope of log f against s = -log(1-y) over the ECDF knots beyond y0, clipped to [0, 0.95]."""
    mid = (ecdf.x[:-1] + ecdf.x[1:]) / 2
    mid = mid[mid > y0]
    if len(mid) < 3:
        return 0.
    s, logf = -np.log1p(-mid), np.log(ecdf(mid))
    return float(np.clip(np.polyfit(s, logf, 1)[0], 0., .95))


def fit_emission(name, y, config, cap="config", tail=None):
    """Old L0 names (histogram, beta, ..., *_nocap) and the v2 families (hist25, hist200, ecdf400)."""
    if name in V2_FAMILIES and V2_FAMILIES[name][0] != "param":   # "beta" etc. keep their old L0 meaning here
        return Capped(fit_base(name, y, config), config.get("cap") if cap == "config" else cap, tail)
    cap = None if name.endswith("_nocap") else (config.get("cap") if cap == "config" else cap)
    family = name.removesuffix("_nocap")
    if family == "histogram":
        return CappedHistogram(Histogram.fit(y, config["bins"], config["pseudocount"]), cap)
    return ParametricDensity.fit(y, family, cap)


def integral_mass(density, cap=None, points=200001):
    """Integral over (0,1) of min(f, cap) by quadrature on y = sigmoid(s); 1 - this is the mass the cap removes."""
    s = np.linspace(-27., 27., points)
    y = 1. / (1. + np.exp(-s))
    f = np.asarray(density(y), float)
    if cap is not None:
        f = np.minimum(f, cap)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(f * y * (1. - y), s))


def describe(density, cap):
    """Shape summary used in the report: density at a few pivots, max, and mass under the cap."""
    grid = np.array([.001, .01, .1, .25, .5, .75, .9, .99, .999, .9999])
    return {"f1_at": {f"{v:g}": float(d) for v, d in zip(grid, density(grid))},
            "mass_uncapped": integral_mass(density) if cap is None else None,
            "mass_under_cap": integral_mass(density, cap) if cap is not None else None,
            "cap": cap}
