"""L0 evaluation. Gate order is enforced by code order:
fit f1 on calibration -> thresholds on null_cal -> dev check -> write frozen.json -> only then test / null_test.
"""
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ..calibration import wilson
from ..detector import Detector, METHODS, intervals
from ..io import new_output, provenance, read_json, read_jsonl, write_json, write_jsonl
from ..paths import project_path
from ..progress import progress
from .config import validate_l0
from .decide import build_rules, decide, rules_from_dict, rules_to_dict
from .diagnose import boundedness, split_half
from .emission import describe, fit_emission
from .metrics import (auc, bootstrap_ci, bootstrap_indices, boundary_summary, mean_or_none,
                      paired_difference, position_metrics, segment_report)
from .schema import TEST_SCENARIOS, validate_article

CONTROLS = ("independent", "fixed_e", "fixed_rho", "window")
EVAL_KEYS = ("emission", "decision", "bins", "pseudocount", "cap", "alpha", "rho_grid", "fixed_e",
             "fixed_rho", "window", "merge_gap", "fraction_tau", "min_group_size", "bootstrap")
SHAPE_FAMILIES = ("histogram", "beta", "beta_mixture")
SATURATION = .99


def load_articles(directory, key, context_width):
    metas = {a["article_id"]: a for a in read_jsonl(directory / "articles.jsonl")}
    rows = defaultdict(list)
    for row in read_jsonl(directory / "positions.jsonl"):
        rows[row["article_id"]].append(row)
    if set(rows) != set(metas):
        raise ValueError("positions.jsonl and articles.jsonl cover different articles.")
    articles = []
    for article_id, meta in metas.items():
        validate_article(rows[article_id], meta, key, context_width)
        articles.append({"id": article_id, "split": meta["split"], "scenario": meta["scenario"],
                         "n_eff": meta["n_eff"], "eps_true": meta["eps_true"],
                         "y": np.array([r["Y"] for r in rows[article_id]]),
                         "truth": np.array([r["truth"] for r in rows[article_id]], bool)})
    return articles


def infer(articles, detectors, label):
    results = {}
    for number, (key, detector) in enumerate(detectors.items(), 1):
        results[key] = [detector.infer(a["y"]) for a in articles]
        progress(label, number, len(detectors), f"{key[0]}/{key[1]}")
    return results


def score(articles, results, rules, context_width):
    rows = []
    for key, detections in results.items():
        emission, method = key
        for a, det in zip(articles, detections):
            labels, d = decide(det.scores, a["n_eff"], rules[key])
            segments = segment_report(a["truth"], labels)
            rows.append({
                "article_id": a["id"], "split": a["split"], "scenario": a["scenario"],
                "emission": emission, "method": method, "n_eff": a["n_eff"],
                "eps_true": a["eps_true"], "eps_hat": det.epsilon, "eps_error": abs(det.epsilon - a["eps_true"]),
                "eps_endpoint": det.epsilon in (0., 1.), "rho": det.rho, "identifiable": det.identifiable,
                "saturated": float((det.scores > SATURATION).mean()), "seconds": det.seconds,
                **d, **position_metrics(a["truth"], labels),
                **{k: segments[k] for k in ("true_segments", "predicted_segments", "recalled_segments",
                                            "segment_recall", "boundary_errors")},
                "pred_intervals": [[s + context_width, e + context_width] for s, e in intervals(labels)]})
    return rows


def rate(count, n):
    return {"count": int(count), "n": int(n), "rate": count / n if n else None, "wilson95": wilson(count, n)}


def cell(rows, null_rows, indices):
    positive = [r for r in rows if r["signal_count"] > 0]
    out = {"articles": len(rows), "positive_articles": len(positive),
           "iou": bootstrap_ci([r["iou"] for r in positive], indices) if positive else {"mean": None, "ci95": [None, None]},
           "precision": mean_or_none([r["precision"] for r in positive]),
           "recall": mean_or_none([r["recall"] for r in positive]),
           "f1": mean_or_none([r["f1"] for r in positive]),
           "eps_error": mean_or_none([r["eps_error"] for r in rows]),
           "eps_hat": mean_or_none([r["eps_hat"] for r in rows]),
           "eps_endpoint": {"at_0": sum(r["eps_hat"] == 0. for r in rows), "at_1": sum(r["eps_hat"] == 1. for r in rows)},
           "rho_counts": {str(k): sum(r["rho"] == k for r in rows) for k in sorted({r["rho"] for r in rows})},
           "segment_recall": mean_or_none([r["segment_recall"] for r in rows]),
           "boundary": boundary_summary([r["boundary_errors"] for r in rows]),
           "detect_M": rate(sum(r["M"] for r in rows), len(rows)),
           "detect_P": rate(sum(r["P"] for r in rows), len(rows)),
           "posterior_saturation": mean_or_none([r["saturated"] for r in rows]),
           "mean_seconds": mean_or_none([r["seconds"] for r in rows])}
    if null_rows is not None:
        out["any_fp_M"] = rate(sum(r["M"] for r in null_rows), len(null_rows))
        out["any_fp_P"] = rate(sum(r["P"] for r in null_rows), len(null_rows))
    return out


def aggregate(rows, null_rows, config, scenarios=TEST_SCENARIOS, seed=42):
    """Per scenario x emission x method summaries, paired bootstrap, and the continuity-gain verdict."""
    index = defaultdict(list)
    for r in rows:
        index[(r["scenario"], r["emission"], r["method"])].append(r)
    nulls = defaultdict(list)
    for r in null_rows or []:
        nulls[(r["emission"], r["method"])].append(r)
    for bucket in list(index.values()) + list(nulls.values()):
        bucket.sort(key=lambda r: r["article_id"])
    summary, gain, auc_table = {}, {}, {}
    for scenario in scenarios:
        keys = [k for k in index if k[0] == scenario]
        if not keys:
            continue
        positive = [r for r in index[keys[0]] if r["signal_count"] > 0]
        indices = bootstrap_indices(max(len(positive), 1), config["bootstrap"], seed)
        summary[scenario], auc_table[scenario] = {}, {}
        for _, emission, method in keys:
            sel = index[(scenario, emission, method)]
            null_sel = nulls.get((emission, method)) if null_rows is not None else None
            summary[scenario].setdefault(emission, {})[method] = cell(sel, null_sel, indices)
            if null_sel:
                pos = [r for r in sel if r["signal_count"] > 0]
                auc_table[scenario].setdefault(emission, {})[method] = {
                    "M": auc([r["max_score"] for r in pos], [r["max_score"] for r in null_sel]),
                    "P": auc([r["fraction"] for r in pos], [r["fraction"] for r in null_sel])}
        gain[scenario] = {}
        for emission in summary[scenario]:
            if "opt_hmm" not in summary[scenario][emission] or not positive:
                continue
            iou = {m: [r["iou"] for r in index[(scenario, emission, m)] if r["signal_count"] > 0]
                   for m in summary[scenario][emission]}
            diffs = {c: paired_difference(iou["opt_hmm"], iou[c], indices)
                     for c in CONTROLS if c in iou}
            no_worse = all(d["ci95"][0] >= 0 for d in diffs.values())
            fp = summary[scenario][emission]["opt_hmm"]
            entry = {"paired_iou_diff_vs": diffs, "iou_not_below_any_control": no_worse}
            if "any_fp_M" in fp:
                entry["any_fp_M_within_alpha"] = fp["any_fp_M"]["rate"] <= config["alpha"]
                entry["any_fp_P_within_alpha"] = fp["any_fp_P"]["rate"] <= config["alpha"]
                entry["gain_M"] = no_worse and entry["any_fp_M_within_alpha"]
                entry["gain_P"] = no_worse and entry["any_fp_P_within_alpha"]
            gain[scenario][emission] = entry
    return {"summary": summary, "gain": gain, "document_auc": auc_table}


def truth_stats(articles):
    out = {}
    for scenario in (*TEST_SCENARIOS, "null"):
        eps = [a["eps_true"] for a in articles if a["scenario"] == scenario]
        if eps:
            out[scenario] = {"articles": len(eps), "eps_true_mean": float(np.mean(eps)),
                             "eps_true_min": float(np.min(eps)), "eps_true_max": float(np.max(eps)),
                             "articles_without_surviving_signal": int(sum(e == 0 for e in eps))}
    return out


def evaluate_positions(positions_dir, output, override=None):
    t_start, timings = perf_counter(), {}
    directory = Path(project_path(positions_dir))
    manifest = read_json(directory / "manifest.json")
    config = dict(manifest["config"])
    for name, value in (override or {}).items():
        if name not in EVAL_KEYS and config.get(name) != value:
            raise ValueError(f"Evaluation may not change protocol setting {name!r}.")
        config[name] = value
    validate_l0(config)
    m, key = config["context_width"], manifest["key"]
    output = new_output(project_path(output))

    phase = perf_counter()
    articles = load_articles(directory, key, m)
    timings["load_and_validate_seconds"] = perf_counter() - phase
    by_split = defaultdict(list)
    for a in articles:
        by_split[a["split"]].append(a)
    for role in ("calibration", "dev", "test", "null_cal", "null_test"):
        if not by_split[role]:
            raise ValueError(f"Missing role {role}.")
    if any(not a["truth"].all() for a in by_split["calibration"]):
        raise ValueError("Calibration must be pure surviving watermark.")
    if any(a["truth"].any() for a in by_split["null_cal"] + by_split["null_test"]):
        raise ValueError("Null roles contain positive truth.")

    # G2/G3: emissions and detectors from the frozen configuration.
    phase = perf_counter()
    calibration_y = np.concatenate([a["y"] for a in by_split["calibration"]])
    emissions = {name: fit_emission(name, calibration_y, config) for name in config["emission"]}
    timings["emission_fit_seconds"] = perf_counter() - phase
    detectors = {(name, method): Detector(density, method=method, rhos=config["rho_grid"],
                                          fixed_e=config["fixed_e"], fixed_rho=config["fixed_rho"],
                                          window=config["window"])
                 for name, density in emissions.items() for method in METHODS}

    # G4: tau / q on null_cal only, then a dev check, then freeze.
    phase = perf_counter()
    null_results = infer(by_split["null_cal"], detectors, "null_cal 推断")
    rules = {k: build_rules([(a["n_eff"], d.scores) for a, d in zip(by_split["null_cal"], null_results[k])], config)
             for k in detectors}
    timings["null_cal_seconds"] = perf_counter() - phase
    phase = perf_counter()
    dev_rows = score(by_split["dev"], infer(by_split["dev"], detectors, "dev 推断"), rules, m)
    dev_report = aggregate(dev_rows, None, config, seed=1)
    timings["dev_seconds"] = perf_counter() - phase
    warnings = []
    for name in ("M", "P"):
        short = [k for k, r in rules.items() if not r[name]["pooled"].finite_rank]
        if short:
            warnings.append(f"rule {name}: null_cal has {len(by_split['null_cal'])} articles, too few for alpha={config['alpha']}; "
                            f"threshold set to 1 (never fires) for {len(short)}/{len(rules)} emission x method combinations.")
    frozen = {"config": {name: config[name] for name in (*EVAL_KEYS, "tokens", "context_width", "key_id", "roles", "edits")},
              "emissions": {name: density.to_dict() for name, density in emissions.items()},
              "rules": {f"{k[0]}|{k[1]}": rules_to_dict(r) for k, r in rules.items()},
              "dev_check": dev_report["summary"], "warnings": warnings}
    write_json(output / "frozen.json", frozen)
    # G5: from here on only objects reloaded from frozen.json are used.
    rules = {k: rules_from_dict(frozen["rules"][f"{k[0]}|{k[1]}"]) for k in detectors}

    phase = perf_counter()
    test_results = infer(by_split["test"], detectors, "test 推断")
    test_rows = score(by_split["test"], test_results, rules, m)
    null_test_results = infer(by_split["null_test"], detectors, "null_test 推断")
    null_rows = score(by_split["null_test"], null_test_results, rules, m)
    timings["test_and_null_test_seconds"] = perf_counter() - phase

    phase = perf_counter()
    main = aggregate(test_rows, null_rows, config)
    timings["bootstrap_seconds"] = perf_counter() - phase

    phase = perf_counter()
    scenario_y = {s: np.concatenate([a["y"] for a in by_split["test"] if a["scenario"] == s] or [np.array([])])
                  for s in TEST_SCENARIOS}
    scenario_y["null"] = np.concatenate([a["y"] for a in by_split["null_test"]])
    shapes = {}
    for family in SHAPE_FAMILIES:
        density = emissions[family] if family in emissions else fit_emission(family, calibration_y, config)
        shapes[family] = describe(density, config.get("cap"))
    diagnostics = {"split_half": split_half([a["y"] for a in by_split["calibration"]], config["emission"], config),
                   "boundedness": boundedness(emissions, calibration_y, scenario_y),
                   "emission_shapes": shapes}
    timings["diagnostics_seconds"] = perf_counter() - phase

    both = test_rows + null_rows
    equivalence = {"rows": len(both),
                   "P_same_equals_M": float(np.mean([r["P_same"] == r["M"] for r in both])),
                   "P_differs_from_M": float(np.mean([r["P"] != r["M"] for r in both]))}
    report = {"config": config, "provenance": provenance(), "manifest": {k: manifest[k] for k in
              ("articles", "positions", "edit_summary", "toy", "key_id", "protocol")},
              "truth": truth_stats(articles), "roles": {r: len(v) for r, v in by_split.items()},
              **main, "dev": dev_report["summary"], "diagnostics": diagnostics, "rule_equivalence": equivalence,
              "warnings": warnings, "timings": {**timings, "total_seconds": perf_counter() - t_start},
              "thresholds_pooled": {f"{k[0]}|{k[1]}": {"tau": r["M"]["pooled"].value, "q": r["P"]["pooled"].value}
                                    for k, r in rules.items()}}
    report["smoke_check"] = check_nonempty(report)
    slim = lambda rows: [{k: v for k, v in r.items() if k != "boundary_errors"} for r in rows]
    write_jsonl(output / "metrics.jsonl", slim(dev_rows) + slim(test_rows) + slim(null_rows))
    write_jsonl(output / "predictions.jsonl", [
        {"article_id": a["id"], "emission": k[0], "method": k[1], "epsilon": d.epsilon, "rho": d.rho,
         "scores": [round(float(s), 6) for s in d.scores]}
        for k, detections in test_results.items() for a, d in zip(by_split["test"], detections)])
    write_json(output / "report.json", report)
    from .report import render
    (output / "report.md").write_text(render(report), encoding="utf-8")
    return {"report": str(output / "report.md"), "smoke_check": report["smoke_check"],
            "equivalence": equivalence, "seconds": report["timings"]["total_seconds"]}


def check_nonempty(report):
    """Smoke exit criterion: every main-table metric is a number for every scenario x emission x method."""
    missing = []
    for scenario, by_emission in report["summary"].items():
        for emission, by_method in by_emission.items():
            for method, c in by_method.items():
                values = {"iou": c["iou"]["mean"], "precision": c["precision"], "recall": c["recall"], "f1": c["f1"],
                          "eps_error": c["eps_error"], "any_fp_M": c["any_fp_M"]["rate"], "detect_M": c["detect_M"]["rate"],
                          "any_fp_P": c["any_fp_P"]["rate"], "detect_P": c["detect_P"]["rate"]}
                missing += [f"{scenario}/{emission}/{method}/{name}" for name, v in values.items() if v is None]
    return {"all_metrics_present": not missing, "missing": missing}
