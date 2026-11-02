"""f1 families under one density cap K, fitted on pure-watermark calibration pivots only."""
import numpy as np

from ..emissions import Histogram
from ..parametric import ParametricDensity


class CappedHistogram:
    """Histogram density with the same upper bound as the parametric families."""

    def __init__(self, histogram, cap):
        self.histogram, self.cap = histogram, cap

    def __call__(self, y):
        density = self.histogram(y)
        return density if self.cap is None else np.minimum(density, float(self.cap))

    def to_dict(self):
        return {"family": "histogram", **self.histogram.to_dict(), "cap": self.cap}


def fit_emission(name, y, config):
    cap = None if name.endswith("_nocap") else config.get("cap")
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
