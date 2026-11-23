"""hist200 emissions of L2: pure-watermark f1, the e'-dependent f1(e') and eps_opt on bin counts.

Every density here is piecewise constant on `bins` equal bins, so any sum over positions can be done on the
200 bin values with per-article bin counts. That is exact, and it lets eps_opt run for all articles at once.
"""
import numpy as np

from ..emissions import check_pivots

BINS = 200
SIGMA = 3.            # Gaussian kernel is cut at SIGMA * h


def bin_index(y, bins=BINS):
    return np.minimum((np.asarray(y) * bins).astype(int), bins - 1)


def bin_counts(index, bins=BINS):
    """(articles, positions) bin indices -> (articles, bins) counts."""
    index = np.atleast_2d(index)
    out = np.zeros((len(index), bins), np.int64)
    rows = np.repeat(np.arange(len(index)), index.shape[1])
    np.add.at(out, (rows, index.ravel()), 1)
    return out


def hist_density(counts, pseudocount=.5):
    """Same formula as emissions.Histogram.fit on pooled counts."""
    counts = np.asarray(counts, float)
    bins = len(counts)
    return bins * (counts + pseudocount) / (counts.sum() + bins * pseudocount)


def capped(density, cap):
    density = np.asarray(density, float)
    return density if cap is None else np.minimum(density, float(cap))


class HistOpt:
    """eps_opt (Li et al. 2025, as in proportion.OptProportion) for a histogram density under a cap.

    Same root-finding as OptProportion (bisection on [lo, 1-lo], same boundary rules), with the three means
    (null quadrature, pure-watermark pivots, one article) evaluated on bins.
    """

    def __init__(self, density, wm_counts, cap=None, lo=1e-3, points=20001, tolerance=1e-10, max_iterations=80):
        self.g = capped(density, cap)
        self.bins = len(self.g)
        s = np.linspace(-27., 27., points)
        y = 1. / (1. + np.exp(-s))
        w = y * (1. - y) * (s[1] - s[0])
        w[[0, -1]] /= 2
        self.null_w = np.bincount(bin_index(y, self.bins), weights=w, minlength=self.bins)
        wm = np.asarray(wm_counts, float)
        self.wm_w = wm / wm.sum()
        self.wm_flat = bool(np.all(np.abs(self.g[wm > 0] - 1) <= 1e-12))
        self.lo, self.tolerance, self.max_iterations = lo, tolerance, max_iterations

    def _v(self, e):
        return (1. - self.g)[None, :] / ((1. - e)[:, None] + e[:, None] * self.g[None, :])

    def _residual(self, e, counts, n):
        """(residual, ok) per article at per-article e."""
        v = self._v(e)
        null_mean, wm_mean = v @ self.null_w, v @ self.wm_w
        den = null_mean - wm_mean
        ok = np.abs(den) >= 1e-14
        doc_mean = (v * counts).sum(axis=1) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (null_mean - doc_mean) / den - e
        return np.where(ok, r, 0.), ok

    def estimate_many(self, counts):
        """counts: (D, bins). Returns value (D,), identifiable (D,), boundary (D,)."""
        counts = np.asarray(counts, float)
        d = len(counts)
        n = counts.sum(axis=1)
        informative = ((counts > 0) & (np.abs(self.g - 1) > 1e-12)[None, :]).any(axis=1)
        ident = informative & (not self.wm_flat)
        value, boundary = np.zeros(d), np.zeros(d, bool)
        a, b = np.full(d, self.lo), np.full(d, 1. - self.lo)
        fa, oka = self._residual(a, counts, n)
        fb, okb = self._residual(b, counts, n)
        ident &= oka & okb
        live = ident.copy()
        flat = live & (fa * fb > 0)
        value[flat] = np.where(np.abs(fa) <= np.abs(fb), a, b)[flat]
        boundary[flat] = True
        active = live & ~flat
        for _ in range(self.max_iterations):
            if not active.any():
                break
            mid = (a + b) / 2
            fm, okm = self._residual(mid, counts, n)
            lost = active & ~okm
            ident[lost] = False
            value[lost] = mid[lost]
            active &= okm
            left = active & (fa * fm <= 0)
            right = active & ~left
            b[left], a[right], fa[right] = mid[left], mid[right], fm[right]
            done = active & (b - a <= self.tolerance)
            value[done] = ((a + b) / 2)[done]
            active &= ~done
        value[active] = ((a + b) / 2)[active]
        return value, ident, boundary


def _kernel_weights(eps, e, h):
    """Truncated Gaussian kernel weights of per-article e' around target e (uniform when h is None)."""
    if h is None:
        return np.ones(len(eps))
    z = (eps - e) / h
    return np.where(np.abs(z) <= SIGMA, np.exp(-.5 * z * z), 0.)


def _fill_unsupported(grid, rows, supported):
    """Interpolate interior unsupported grid entries and constant-clamp edges from supported entries."""
    grid = np.asarray(grid, float)
    out = np.array(rows, float, copy=True)
    idx = np.flatnonzero(supported)
    for i in np.flatnonzero(~supported):
        pos = int(np.searchsorted(grid[idx], grid[i]))
        if pos <= 0:
            out[i] = out[idx[0]]
        elif pos >= len(idx):
            out[i] = out[idx[-1]]
        else:
            left, right = idx[pos - 1], idx[pos]
            lam = (grid[i] - grid[left]) / (grid[right] - grid[left])
            out[i] = (1. - lam) * out[left] + lam * out[right]
    return out


def _supported_mask(total, neff, min_neff):
    has_data = np.asarray(total, float) > 0
    if not has_data.any():
        return has_data
    if min_neff is not None:
        enough = has_data & (np.asarray(neff, float) >= float(min_neff))
        if enough.any():
            return enough
    return has_data


class KernelGrid:
    """f1(.; e): kernel-weighted hist200 on a grid of target proportions, linearly interpolated in e.

    Training position i carries the weight K((e'_{d(i)} - e) / h) of its article. Weights are rescaled so that
    they add up to n_eff(e) = (sum w)^2 / sum w^2; h = None (infinite bandwidth) gives equal weights, i.e. the
    plain histogram of all training positions.
    """

    def __init__(self, grid, density, neff):
        self.grid = np.asarray(grid, float)
        self.density = np.asarray(density, float)
        self.neff = np.asarray(neff, float)
        if self.density.shape != (len(self.grid), self.density.shape[1]) or not np.allclose(self.density.mean(axis=1), 1):
            raise ValueError("Each grid density must integrate to one.")

    @classmethod
    def fit(cls, counts, eps, grid, h, pseudocount=.5, clamp_sparse=False, min_neff=None):
        """counts: (articles, bins) truth=1 pivot counts; eps: (articles,) e' of each article."""
        counts, eps = np.asarray(counts, float), np.asarray(eps, float)
        size = counts.sum(axis=1)
        bins = counts.shape[1]
        density, totals, neff = [], [], []
        for e in grid:
            w = _kernel_weights(eps, e, h)
            total = float((w * size).sum())
            totals.append(total)
            if total <= 0:
                if not clamp_sparse:
                    raise ValueError(f"No training positions within {SIGMA:g} bandwidths of e = {e:g}; add data, do not widen h.")
                density.append(np.ones(bins))
                neff.append(0.)
                continue
            n_eff = float(total ** 2 / (w * w * size).sum())
            weighted = (w[:, None] * counts).sum(axis=0) * (n_eff / total)
            density.append(bins * (weighted + pseudocount) / (n_eff + bins * pseudocount))
            neff.append(n_eff)
        supported = _supported_mask(totals, neff, min_neff if clamp_sparse else None)
        if not supported.any():
            raise ValueError(f"No training positions within {SIGMA:g} bandwidths of e = {grid[0]:g}; add data, do not widen h.")
        dens_arr = np.array(density)
        if clamp_sparse and not supported.all():
            dens_arr = _fill_unsupported(grid, dens_arr, supported)
        return cls(grid, dens_arr, np.array(neff))

    def densities(self, eps):
        """Interpolated bin densities for each article's e': (articles, bins)."""
        eps = np.clip(np.asarray(eps, float), self.grid[0], self.grid[-1])
        j = np.clip(np.searchsorted(self.grid, eps, side="right") - 1, 0, len(self.grid) - 2)
        lam = ((eps - self.grid[j]) / (self.grid[j + 1] - self.grid[j]))[:, None]
        return (1. - lam) * self.density[j] + lam * self.density[j + 1]

    def to_dict(self):
        return {"grid": self.grid.tolist(), "density": self.density.tolist(), "neff": self.neff.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["grid"], data["density"], data["neff"])


class BetaGrid(KernelGrid):
    """KernelGrid whose grid densities are Beta(a, b) fitted by weighted moments instead of weighted histograms.

    Weights are exactly those of KernelGrid (so the family is the only difference). The mean and variance come from
    the bin midpoints (variance minus the bin-width term); the Beta is averaged over every bin on a fine sub-grid and
    rescaled to mean one, so the result is again a hist200 density that the HMM code reads like any other.
    """
    SUB = 200

    def __init__(self, grid, density, neff, params):
        super().__init__(grid, density, neff)
        self.params = np.asarray(params, float)

    @staticmethod
    def moments(weighted_counts):
        """(a, b) of the Beta with the weighted mean and variance of binned pivots; (1, 1) if the variance is degenerate."""
        w = np.asarray(weighted_counts, float)
        bins = len(w)
        mid = (np.arange(bins) + .5) / bins
        mean = float((w * mid).sum() / w.sum())
        var = float((w * (mid - mean) ** 2).sum() / w.sum()) - 1. / (12. * bins * bins)
        common = mean * (1. - mean) / var - 1. if var > 0 else 0.
        return (1., 1.) if not np.isfinite(common) or common <= 0 else (mean * common, (1. - mean) * common)

    @staticmethod
    def bins(a, b, bins=BINS):
        u = (np.arange(bins * BetaGrid.SUB) + .5) / (bins * BetaGrid.SUB)
        log_pdf = (a - 1.) * np.log(u) + (b - 1.) * np.log1p(-u)
        density = np.exp(log_pdf - log_pdf.max()).reshape(bins, BetaGrid.SUB).mean(axis=1)
        return density / density.mean()

    @classmethod
    def fit(cls, counts, eps, grid, h, pseudocount=.5, clamp_sparse=False, min_neff=None):
        counts, eps = np.asarray(counts, float), np.asarray(eps, float)
        size = counts.sum(axis=1)
        bins = counts.shape[1]
        density, totals, neff, params = [], [], [], []
        for e in grid:
            w = _kernel_weights(eps, e, h)
            total = float((w * size).sum())
            totals.append(total)
            if total <= 0:
                if not clamp_sparse:
                    raise ValueError(f"No training positions within {SIGMA:g} bandwidths of e = {e:g}; add data, do not widen h.")
                density.append(np.ones(bins))
                neff.append(0.)
                params.append((1., 1.))
                continue
            a, b = cls.moments((w[:, None] * counts).sum(axis=0))
            density.append(cls.bins(a, b, bins))
            neff.append(float(total ** 2 / (w * w * size).sum()))
            params.append((a, b))
        supported = _supported_mask(totals, neff, min_neff if clamp_sparse else None)
        if not supported.any():
            raise ValueError(f"No training positions within {SIGMA:g} bandwidths of e = {grid[0]:g}; add data, do not widen h.")
        dens_arr, params_arr = np.array(density), np.array(params, float)
        if clamp_sparse and not supported.all():
            dens_arr = _fill_unsupported(grid, dens_arr, supported)
            params_arr = _fill_unsupported(grid, params_arr, supported)
        return cls(grid, dens_arr, np.array(neff), params_arr)

    @classmethod
    def constant(cls, counts, grid):
        """One Beta of the pooled counts at every grid point: the fixed-f1 control of the adaptive Beta."""
        a, b = cls.moments(np.asarray(counts, float))
        n = float(np.asarray(counts).sum())
        return cls(grid, np.tile(cls.bins(a, b, len(counts)), (len(grid), 1)), np.full(len(grid), n),
                   np.tile([a, b], (len(grid), 1)))

    def to_dict(self):
        return {**super().to_dict(), "params": self.params.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["grid"], data["density"], data["neff"], data["params"])


def ratios(index, per_article_density, cap=None):
    """Density ratios g[d, t] = f(y[d, t]) from per-article bin densities (D, bins) or one shared (bins,)."""
    dens = capped(per_article_density, cap)
    if dens.ndim == 1:
        return dens[index]
    return np.take_along_axis(dens, index, axis=1)


def pivot_index(y):
    return bin_index(check_pivots(np.asarray(y, float).ravel())).reshape(np.shape(y))
