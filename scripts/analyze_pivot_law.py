"""Numerical research checks for docs/pivot_law_derivation_20261025.md.

Uses fresh ideal uniforms, not the fixed context-key protocol or an actual LLM.
Run from Linux: python scripts/analyze_pivot_law.py --output tmp/piv-law
NumPy is required. --plot additionally needs matplotlib.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def cdf(y, p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return np.exp(np.log(np.asarray(y)[:, None]) / p) @ p


def density(y, p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return np.exp(np.log(np.asarray(y)[:, None]) * (1 / p - 1)).sum(axis=1)


def ks_uniform(x):
    x = np.sort(x)
    n = len(x)
    return float(max(np.max(np.arange(1, n + 1) / n - x),
                     np.max(x - np.arange(n) / n)))


def direct_sample(rng, p, count):
    """Full independent random vectors and Gumbel competition; no pivot-law sampler."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    result = []
    selected_prob = []
    for start in range(0, count, 8192):
        u = np.maximum(rng.random((min(8192, count - start), len(p))),
                       np.nextafter(0., 1.))
        winner = (-np.log(u) / p).argmin(axis=1)
        result.append(u[np.arange(len(u)), winner])
        selected_prob.append(p[winner])
    return np.concatenate(result), np.concatenate(selected_prob)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/piv-law'))
    parser.add_argument('--samples', type=int, default=120_000)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.samples < 10000:
        parser.error('--samples must be at least 10000 for these research checks')
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    scenarios = {
        'uniform_2': np.array([.5, .5]),
        'uniform_8': np.full(8, 1 / 8),
        'concentrated': np.array([.99, .01]),
        'almost_deterministic': np.array([.999, .001]),
        'same_s2_A': np.array([.5, .25, .25]),
        'same_s2_B': np.array([5 / 12, 5 / 12, 1 / 6]),
        'deterministic_limit': np.array([1.]),
    }
    nodes, weights = np.polynomial.legendre.leggauss(512)
    quad_y, quad_w = (nodes + 1) / 2, weights / 2
    rows = {}
    largest_z = 0.
    for name, p in scenarios.items():
        y, picked_p = direct_sample(rng, p, args.samples)
        x = -np.log(y)
        checks = []
        for k in (1, 2, 3):
            expected = math.factorial(k) * np.sum(p ** (k + 1))
            variance = math.factorial(2 * k) * np.sum(p ** (2 * k + 1)) - expected ** 2
            se = np.sqrt(variance / args.samples)
            actual = float(np.mean(x ** k))
            z = abs(actual - expected) / se
            largest_z = max(largest_z, z)
            checks.append({'k': k, 'predicted': float(expected), 'observed': actual,
                           'monte_carlo_standard_error': float(se), 'absolute_z': float(z)})
        f = density(quad_y, p)
        divergence = float(np.sum(quad_w * f * np.log(f)))
        entropy = float(-np.sum(p * np.log(p)))
        s2 = float(p @ p)
        upper = entropy - 1 + s2
        pit_ks = ks_uniform(cdf(y, p))
        selected_pit_ks = ks_uniform(np.exp(np.log(y) / picked_p))
        assert pit_ks < .012 and selected_pit_ks < .012, name
        assert -1e-9 <= divergence <= upper + 1e-7, name
        rows[name] = {'p': p.tolist(), 's2': s2, 's3': float(np.sum(p ** 3)),
                      'entropy_nats': entropy, 'var_minus_log_y': float(2 * np.sum(p ** 3) - s2 ** 2),
                      'pivot_kl_to_uniform_quadrature': divergence,
                      'kl_upper_bound': upper,
                      'pit_ks': pit_ks, 'conditional_on_w_pit_ks': selected_pit_ks,
                      'moment_checks': checks}
    assert largest_z < 6, largest_z
    assert abs(rows['uniform_2']['pivot_kl_to_uniform_quadrature'] - (math.log(2) - .5)) < 1e-6
    assert abs(rows['same_s2_A']['s2'] - rows['same_s2_B']['s2']) < 1e-14

    grid = np.linspace(.0001, .9999, 2001)
    a, b = scenarios['same_s2_A'], scenarios['same_s2_B']
    same_s2_ks = float(np.max(np.abs(cdf(grid, a) - cdf(grid, b))))
    assert same_s2_ks > .001
    largest_cdf_tv_ratio = 0.
    smallest_majorization_difference = 1.
    for _ in range(200):
        p, q = rng.dirichlet(np.full(8, .5), size=2)
        tv = float(np.sum(np.abs(p - q)) / 2)
        ks = float(np.max(np.abs(cdf(grid, p) - cdf(grid, q))))
        assert ks <= tv + 1e-12
        largest_cdf_tv_ratio = max(largest_cdf_tv_ratio, ks / tv)
        flatter = .5 * p + .5 / len(p)
        gap = float(np.min(cdf(grid, p) - cdf(grid, flatter)))
        smallest_majorization_difference = min(smallest_majorization_difference, gap)
        assert gap >= -1e-12

    # Exact binned population likelihood, independent of Monte Carlo estimation.
    # Calibration assumes uniform_2; all target tokens are watermark-enabled,
    # with only fraction a using that strong P. The rest have near-singular P.
    edges = np.linspace(0, 1, 101)

    def bin_probs(p):
        return np.diff(np.r_[0., cdf(edges[1:-1], p), 1.])

    null_mass = np.diff(edges)
    cal_mass = bin_probs(scenarios['uniform_2'])
    weak_mass = bin_probs(scenarios['almost_deterministic'])
    eps = np.linspace(0, 1, 10001)
    log_model = np.log((1 - eps[:, None]) * null_mass + eps[:, None] * cal_mass)
    mismatch = []
    for strong_fraction in (.25, .5, .75):
        target = strong_fraction * cal_mass + (1 - strong_fraction) * weak_mass
        fitted = float(eps[np.argmax(log_model @ target)])
        target_s2 = strong_fraction * rows['uniform_2']['s2'] + (1 - strong_fraction) * rows['almost_deterministic']['s2']
        moment_estimate = (1 - target_s2) / (1 - rows['uniform_2']['s2'])
        mismatch.append({'true_watermark_enabled_fraction': 1., 'strong_condition_fraction': strong_fraction,
                         'fitted_epsilon_binned_population_mle': fitted,
                         'moment_estimator_using_calibration_s2': float(moment_estimate)})
        assert abs(fitted - strong_fraction) < .03

    report = {'model': 'Fresh independent ideal uniforms at every sample; not real LLM validation',
              'seed': args.seed, 'samples_per_scenario': args.samples,
              'scenarios': rows, 'max_absolute_moment_z': float(largest_z),
              'same_s2_cdf_difference': same_s2_ks,
              'stability_checks': {'random_pairs': 200, 'max_cdf_distance_over_token_tv': largest_cdf_tv_ratio,
                                   'min_cdf_gap_after_flattening': smallest_majorization_difference},
              'population_mismatch': mismatch}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.plot:
        os.environ.setdefault('MPLCONFIGDIR', str(Path('tmp/piv-mpl').resolve()))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        ygrid = np.linspace(.001, .999, 1000)
        for name, label in [('uniform_2', 'P=(0.5,0.5)'), ('uniform_8', 'P=Uniform(8)'),
                            ('concentrated', 'P=(0.99,0.01)')]:
            axes[0].plot(ygrid, cdf(ygrid, scenarios[name]), label=label)
        axes[0].plot(ygrid, ygrid, '--', color='#555555', label='Null: Uniform(0,1)')
        axes[0].set(xlabel='Pivot y', ylabel='CDF', title='A. Conditional alternative laws')
        axes[0].legend(fontsize=8)
        axes[1].plot(ygrid, density(ygrid, a), label='P=(1/2,1/4,1/4)')
        axes[1].plot(ygrid, density(ygrid, b), '--', label='Q=(5/12,5/12,1/6)')
        axes[1].set(xlabel='Pivot y', ylabel='Density', title='B. Equal collision probability, different laws')
        axes[1].legend(fontsize=8)
        fractions = [row['strong_condition_fraction'] for row in mismatch]
        estimates = [row['fitted_epsilon_binned_population_mle'] for row in mismatch]
        axes[2].plot(fractions, estimates, 'o-', color='#d45d30', label='Fitted watermark fraction')
        axes[2].axhline(1, color='#246c8f', linestyle='--', label='True enabled fraction = 1')
        axes[2].set(xlabel='Fraction of strong conditions among enabled tokens', ylabel='Estimated fraction',
                    title='C. Bias from applying a strong calibration law', ylim=(0, 1.08), xticks=fractions)
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output / 'relations.png', dpi=180)
        plt.close(fig)
    print(json.dumps({'samples_per_scenario': args.samples, 'max_absolute_moment_z': largest_z,
                      'same_s2_cdf_difference': same_s2_ks, 'population_mismatch': mismatch}, indent=2))


if __name__ == '__main__':
    main()
