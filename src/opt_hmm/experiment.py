"""Full calibration and paired evaluation, shared by simulations and real text."""
from dataclasses import asdict
from time import perf_counter

import numpy as np

from .calibration import calibrate, wilson
from .detector import Detector, METHODS, intervals
from .emissions import Histogram
from .io import provenance, write_json, write_jsonl
from .metrics import evaluate, paired_bootstrap, reliability


def arrays(record):
    y = np.asarray(record["y"], dtype=float)
    mask = np.asarray(record.get("mask", [True] * len(y)), bool)
    truth = np.asarray(record["z"], bool)
    if y.shape != mask.shape or y.shape != truth.shape or not mask.any():
        raise ValueError(f"Invalid record arrays: {record['id']}")
    return y, mask, truth


def length_key(record):
    _, mask, _ = arrays(record)
    return f"n={len(mask)},missing={','.join(map(str, np.flatnonzero(~mask)))}"


def diagnostics(records):
    result = {}
    for role in ("calibration", "null_cal", "null_test"):
        selected = [r for r in records if r["role"] == role]
        if not selected:
            continue
        y = np.concatenate([arrays(r)[0][arrays(r)[1]] for r in selected])
        ordered = np.sort(y)
        count = len(ordered)
        ks = max(np.max(np.arange(1, count + 1) / count - ordered),
                 np.max(ordered - np.arange(count) / count))
        lags = []
        for record in selected:
            values, mask, _ = arrays(record)
            a, b = values[:-1][mask[:-1] & mask[1:]], values[1:][mask[:-1] & mask[1:]]
            if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
                lags.append(float(np.corrcoef(a, b)[0, 1]))
        result[role] = {"documents_or_length_views": len(selected), "pivots": count,
                        "mean_y": float(y.mean()), "uniform_ks_distance": float(ks),
                        "mean_within_document_lag1_correlation": float(np.mean(lags)) if lags else None,
                        "note": "Descriptive distances only; no IID token-level p-values."}
    return result


def run_evaluation(records, output, config, oracle_density=None, oracle_parameters=None):
    groups = {}
    ids = set()
    for record in records:
        if record["id"] in ids:
            raise ValueError("Duplicate record ID.")
        ids.add(record["id"])
        previous = groups.setdefault(record["group"], record["role"])
        if previous != record["role"]:
            raise ValueError(f"Group leaks across roles: {record['group']}")
    required = ("calibration", "null_cal", "null_test", "test")
    for role in required:
        if not any(r["role"] == role for r in records):
            raise ValueError(f"Missing role {role}")
    cal = []
    for record in records:
        if record["role"] == "calibration":
            y, mask, truth = arrays(record)
            if not truth[mask].all():
                raise ValueError("Emission calibration must be pure surviving watermark.")
            cal.extend(y[mask])
    density_started = perf_counter()
    histogram = Histogram.fit(cal, config.get("bins", 50), config.get("pseudocount", .5))
    density_fit_seconds = perf_counter() - density_started
    detectors = {method: Detector(histogram, method=method,
                                  fixed_e=config.get("fixed_e", .5),
                                  fixed_rho=config.get("fixed_rho", .95),
                                  window=config.get("window", 20))
                 for method in METHODS}
    if oracle_density is not None:
        detectors["oracle"] = Detector(oracle_density, method="oracle",
                                      fixed_e=oracle_parameters["epsilon"],
                                      fixed_rho=oracle_parameters["rho"])
    null_by_length = {}
    for record in records:
        if record["role"] == "null_cal":
            _, mask, truth = arrays(record)
            if truth[mask].any():
                raise ValueError("Null calibration contains positive labels.")
            null_by_length.setdefault(length_key(record), []).append(record)
    for key, selected in null_by_length.items():
        if len({r["group"] for r in selected}) != len(selected):
            raise ValueError(f"Null calibration reuses a group at the same length: {key}")
    evaluated = [r for r in records if r["role"] in ("dev", "test", "null_test")]
    missing = {length_key(r) for r in evaluated} - set(null_by_length)
    if missing:
        raise ValueError(f"No independent null calibration for lengths/masks: {missing}")
    predictions, metric_rows, thresholds, summaries, reliability_data = [], [], {}, {}, {}
    calibration_seconds = {}
    for method, detector in detectors.items():
        calibration_started = perf_counter()
        thresholds[method] = {}
        for key, selected in null_by_length.items():
            scores = []
            for record in selected:
                y, mask, _ = arrays(record)
                scores.append(float(detector.infer(y, mask).scores.max()))
            thresholds[method][key] = calibrate(scores, config.get("alpha", .05))
        calibration_seconds[method] = perf_counter() - calibration_started
        method_rows, prob_truths, prob_scores = [], [], []
        for record in evaluated:
            y, mask, truth = arrays(record)
            if record["role"] == "null_test" and truth[mask].any():
                raise ValueError("Null test contains positive labels.")
            result = detector.infer(y, mask)
            threshold = thresholds[method][length_key(record)]
            labels = threshold.predict(result.scores, mask)
            metrics = evaluate(truth, labels, result.scores, result.epsilon, mask)
            row = {"id": record["id"], "group": record["group"],
                   "role": record["role"], "variant": record["variant"],
                   "length_key": length_key(record), "method": method,
                   "epsilon": result.epsilon, "rho": result.rho,
                   "identifiable_in_sample": result.identifiable,
                   "seconds": result.seconds, "stage_seconds": result.stage_seconds, "metrics": metrics}
            metric_rows.append(row)
            method_rows.append(row)
            predictions.append({"id": record["id"], "method": method,
                                "scores": result.scores, "labels": labels,
                                "intervals": intervals(labels), "threshold": threshold.value,
                                "epsilon": result.epsilon, "rho": result.rho})
            if record["role"] == "test":
                prob_truths.append(truth[mask])
                prob_scores.append(result.scores[mask])
        summaries[method] = {}
        for variant in sorted({r["variant"] for r in method_rows if r["role"] == "test"}):
            selected = [r for r in method_rows if r["role"] == "test" and r["variant"] == variant]
            positive = [r for r in selected if r["metrics"]["signal_count"] > 0]
            totals = {name: sum(r["metrics"][name] for r in selected) for name in ("tp", "fp", "fn", "tn")}
            tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
            summaries[method][variant] = {
                "documents": len(selected), "positive_documents": len(positive),
                "mean_iou": float(np.mean([r["metrics"]["iou"] for r in positive])) if positive else None,
                "mean_f1": float(np.mean([r["metrics"]["f1"] for r in positive])) if positive else None,
                "micro_f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
                "empirical_mean_fdp": float(np.mean([r["metrics"]["fdp"] for r in selected])),
                "mean_seconds": float(np.mean([r["seconds"] for r in selected])),
                "counts": totals}
        summaries[method]["null_test"] = {}
        for key in sorted({r["length_key"] for r in method_rows if r["role"] == "null_test"}):
            selected = [r for r in method_rows if r["role"] == "null_test" and r["length_key"] == key]
            if len({r["group"] for r in selected}) != len(selected):
                raise ValueError("Null test must use one view per group at each length.")
            count = sum(r["metrics"]["any_fp"] for r in selected)
            summaries[method]["null_test"][key] = {
                "documents": len(selected), "any_fp_count": count,
                "any_fp_rate": count / len(selected), "wilson95": wilson(count, len(selected))}
        reliability_data[method] = reliability(prob_truths, prob_scores)
    pairs = {method: paired_bootstrap([r for r in metric_rows if r["role"] == "test"], method,
                                      draws=config.get("bootstrap", 1000))
             for method in detectors if method != "independent"}
    report = {"config": config, "provenance": provenance(), "summary": summaries,
              "paired_iou_difference_vs_independent": pairs,
              "diagnostics": diagnostics(records), "density_fit_seconds": density_fit_seconds, "null_calibration_seconds": calibration_seconds,
              "limitations": ["Null calibration controls completely null document any-FP under exchangeability.",
                              "This is not mixed-document FDR control.",
                              "Estimated scores are plug-in model probabilities; check reliability.",
                              "Window scores are transformed evidence, not HMM posteriors."]}
    from .plot import render_example
    example = next(r for r in evaluated if r['role'] == 'test')
    render_example(example, predictions, output / 'example.svg')
    write_json(output / "density.json", histogram.to_dict())
    write_json(output / "thresholds.json", {method: {k: asdict(v) for k, v in values.items()}
                                          for method, values in thresholds.items()})
    write_jsonl(output / "metrics.jsonl", metric_rows)
    write_jsonl(output / "predictions.jsonl", predictions)
    write_json(output / "reliability.json", reliability_data)
    write_json(output / "report.json", report)
    lines = ["# OPT-HMM experiment", "", "Exploratory results; examine null rates alongside localization.", "",
             "| Method | Variant | Positive documents | Mean IoU | Mean F1 |", "|---|---|---:|---:|---:|"]
    for method, values in summaries.items():
        for variant, summary in values.items():
            if variant == "null_test":
                continue
            iou, f1 = summary["mean_iou"], summary["mean_f1"]
            lines.append(f"| {method} | {variant} | {summary['positive_documents']} | "
                         f"{iou if iou is not None else 'NA'} | {f1 if f1 is not None else 'NA'} |")
    lines.extend(["", "| Method | Null length | Documents | Any-FP | Wilson 95% |",
                  "|---|---|---:|---:|---|"])
    for method, values in summaries.items():
        for key, summary in values["null_test"].items():
            lines.append(f"| {method} | {key} | {summary['documents']} | "
                         f"{summary['any_fp_rate']:.4f} | {summary['wilson95']} |")
    lines.extend(["", "All intervals use zero-based half-open token offsets.",
                  "Paired bootstrap resamples original document groups; all variants stay together.",
                  "See report.json, metrics.jsonl and reliability.json for detailed diagnostics.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
