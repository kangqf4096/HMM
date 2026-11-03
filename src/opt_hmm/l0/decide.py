"""Null-calibrated thresholds (tau for max score, q for flagged fraction) and the M / P document rules.

Rule P compares the fraction of positions with gamma > tau_P against q. If tau_P were
the same tau calibrated on the null maximum, then fraction > 0 holds exactly when
max > tau, and q (the 95th null percentile of a mostly-zero statistic) is 0, so P
would be M again. fraction_tau therefore defaults to a separate posterior cut-off;
the "same" variant is always computed as P_same and reported as an equivalence check.

Because the two rules use different per-position cut-offs, they mark different
positions: `decide` returns rule M's labels for localization and `p_labels` returns
rule P's. Both sets are scored so that "gain under P" is a claim about P's own
localization rather than a relabelled copy of M's.
"""
from dataclasses import asdict

import numpy as np

from ..calibration import Threshold, calibrate


def group_thresholds(values_by_group, alpha, min_size):
    """Strict empirical-rank threshold per length group; small groups fall back to the pooled threshold."""
    pooled = np.concatenate([np.asarray(v, float) for v in values_by_group.values()])
    table = {"pooled": calibrate(pooled, alpha)}
    for group, values in values_by_group.items():
        if len(values) >= min_size:
            table[str(group)] = calibrate(values, alpha)
    return table


def lookup(table, n_eff):
    return table.get(str(n_eff), table["pooled"])


def build_rules(null_items, config):
    """null_items: [(n_eff, scores)] from null_cal articles only."""
    alpha, min_size = config["alpha"], config.get("min_group_size", 100)
    tau_fraction = config.get("fraction_tau", 0.5)
    by_group = {}
    for n_eff, scores in null_items:
        by_group.setdefault(n_eff, []).append(scores)
    maxima = {g: [s.max() for s in v] for g, v in by_group.items()}
    rules = {"M": group_thresholds(maxima, alpha, min_size), "tau_fraction": tau_fraction,
             "group_sizes": {str(g): len(v) for g, v in by_group.items()}}
    same = {g: [float((s > lookup(rules["M"], g).value).mean()) for s in v] for g, v in by_group.items()}
    rules["P_same"] = group_thresholds(same, alpha, min_size)
    if tau_fraction == "same":
        rules["P"] = rules["P_same"]
    else:
        fractions = {g: [float((s > tau_fraction).mean()) for s in v] for g, v in by_group.items()}
        rules["P"] = group_thresholds(fractions, alpha, min_size)
    return rules


def p_cutoff(rules, n_eff):
    """Per-position cutoff used by rule P: the frozen fraction_tau, or tau itself when 'same'."""
    cut = rules["tau_fraction"]
    return lookup(rules["M"], n_eff).value if cut == "same" else float(cut)


def p_labels(scores, n_eff, rules):
    """Positions flagged under rule P. Rule M flags `scores > tau`; P flags `scores > fraction_tau`."""
    return np.asarray(scores, float) > p_cutoff(rules, n_eff)


def decide(scores, n_eff, rules):
    """Return rule-M flagged positions and both document decisions; rule P flags its own positions."""
    scores = np.asarray(scores, float)
    tau = lookup(rules["M"], n_eff).value
    labels = scores > tau
    labels_p = labels if rules["tau_fraction"] == "same" else scores > rules["tau_fraction"]
    fraction_same = float(labels.mean())
    fraction = float(labels_p.mean())
    return labels, {"max_score": float(scores.max()), "fraction": fraction, "tau": tau,
                    "M": bool(scores.max() > tau),
                    "P": bool(fraction > lookup(rules["P"], n_eff).value),
                    "P_same": bool(fraction_same > lookup(rules["P_same"], n_eff).value)}


def rules_to_dict(rules):
    return {"tau_fraction": rules["tau_fraction"], "group_sizes": rules["group_sizes"],
            **{name: {group: asdict(t) for group, t in rules[name].items()} for name in ("M", "P", "P_same")}}


def rules_from_dict(data):
    rules = {"tau_fraction": data["tau_fraction"], "group_sizes": data["group_sizes"]}
    for name in ("M", "P", "P_same"):
        rules[name] = {group: Threshold(**t) for group, t in data[name].items()}
    return rules
