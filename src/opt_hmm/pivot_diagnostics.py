"""Probability-law diagnostics; independent of the GPU inference runtime."""
import numpy as np

S2_EDGES = np.array([0., .05, .2, .5, .8, .95, 1.0000001])


def probabilities_from_logits(logits, temperature=1.):
    logits = np.asarray(logits, dtype=np.float64) / temperature
    if temperature <= 0 or logits.ndim != 1 or not np.any(np.isfinite(logits)):
        raise ValueError('Need finite supported logits and positive temperature.')
    if np.any(np.isnan(logits)) or np.any(np.isposinf(logits)):
        raise ValueError('Invalid logits.')
    p = np.exp(logits - np.max(logits))
    return p / p.sum()


def probability_summary(p):
    p = np.asarray(p, dtype=float)
    positive = p[p > 0]
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('Invalid probabilities.')
    s2, s3 = float(p @ p), float(np.sum(p ** 3))
    return {'s2': s2, 's3': s3, 'pmax': float(p.max()),
            'entropy_nats': float(-np.sum(positive * np.log(positive))),
            'effective_vocabulary': 1 / s2,
            'predicted_var_x': max(0., 2 * s3 - s2 * s2)}


def pivot_law(p, y):
    """Exact ideal CDF/log PDF, with no density underflow at tiny pivots."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if not 0 < y < 1 or not len(p) or not np.isclose(p.sum(), 1):
        raise ValueError('Need probabilities and a pivot in (0,1).')
    log_y = np.log(y)
    with np.errstate(over='ignore', under='ignore'):
        powers = log_y / p
        cdf = float(np.sum(p * np.exp(powers)))
        exponents = powers - log_y
        peak = float(np.max(exponents))
        log_pdf = peak + float(np.log(np.exp(exponents - peak).sum()))
    return {'cdf': min(1., max(0., cdf)), 'log_pdf': log_pdf}


def law_ratio(log_pdf):
    # A numerical floor for likelihood products, recorded by the diagnostic run.
    return float(np.exp(np.clip(log_pdf, -700, 700)))


def uniform_distance(values):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return None
    n = len(values)
    return float(max(np.max(np.arange(1, n + 1) / n - values),
                     np.max(values - np.arange(n) / n)))


def source_for_record(record, sources):
    variant = record['variant']
    mode = 'null' if variant.startswith('null_n') else ('switch' if variant == 'switch' else 'watermarked')
    return sources[f"{record['group']}/{mode}"]


class ConditionalHistogram:
    """Fixed S2 groups, shrunk toward the global calibration-only histogram."""
    def __init__(self, y, s2, bins=25, pseudocount=.5, prior_count=50.):
        from .emissions import Histogram
        y, s2 = np.asarray(y), np.asarray(s2)
        if y.shape != s2.shape or not len(y) or prior_count <= 0:
            raise ValueError('Need matched pure calibration pivots and S2.')
        self.global_density = Histogram.fit(y, bins, pseudocount)
        self.prior_count = prior_count
        self.group = np.minimum(np.searchsorted(S2_EDGES, s2, side='right') - 1, len(S2_EDGES) - 2)
        self.counts, self.densities = [], []
        for group in range(len(S2_EDGES) - 1):
            selected = y[self.group == group]
            counts, _ = np.histogram(selected, bins=bins, range=(0, 1))
            density = bins * (counts + prior_count * self.global_density.density / bins) / (len(selected) + prior_count)
            self.counts.append(len(selected))
            self.densities.append(Histogram(density))

    def __call__(self, y, s2):
        group = min(max(int(np.searchsorted(S2_EDGES, s2, side='right') - 1), 0), len(self.densities) - 1)
        return float(self.densities[group]([y])[0])

    def to_dict(self):
        return {'s2_edges': S2_EDGES.tolist(), 'counts': self.counts, 'prior_count': self.prior_count,
                'fit_role': 'calibration only', 'density': [d.to_dict() for d in self.densities]}
