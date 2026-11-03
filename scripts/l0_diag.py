"""Offline diagnostics for an evaluated L0 run; reuses frozen thresholds, never regenerates text.

Answers three questions the main report cannot:

1. Is the observed null any-FP (e.g. 8.5%) a sampling artefact of one threshold, or a real
   calibration shift? Recalibrate on random halves of the pooled 800 null articles and report
   the distribution of the resulting false-positive rate.
2. Does the threshold sit on an atom of the max-score distribution? Report the largest atom at
   tau and the counts strictly above/below it.
3. Does per-document rho selection beat a rho fixed at the model level? Recalibrate and re-score
   the same articles under argmax rho, marginal rho, and four fixed rho values.

Input is the data directory `evaluate` reads (manifest.json, articles.jsonl, positions.jsonl).
Nothing here writes into the run being diagnosed.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_hmm.calibration import calibrate, wilson
from opt_hmm.detector import METHODS, Detector
from opt_hmm.hmm import forward_backward, forward_log_likelihood
from opt_hmm.io import new_output, read_json, write_json
from opt_hmm.l0.config import validate_l0
from opt_hmm.l0.emission import fit_emission
from opt_hmm.l0.evaluate import infer, load_articles
from opt_hmm.l0.metrics import position_metrics
from opt_hmm.paths import project_path
from opt_hmm.proportion import estimate_proportion

ROLES = ("test", "null_cal", "null_test")
FIXED = (0.5, 0.8, 0.95, 0.99)
VARIANTS = ("argmax", "marginal") + tuple(f"fixed_{r}" for r in FIXED)
META = {"tau", "any_fp", "mean_weights"}


def maxima(detections):
    return np.array([d.scores.max() for d in detections], float)


def frequent(detections, cut):
    return np.array([float((d.scores > cut).mean()) for d in detections], float)


def recalibrated_fp(pooled, draws, alpha, seed=0):
    """Split the pooled null max scores in half, calibrate on one half and measure the other."""
    rng = np.random.default_rng(seed)
    half, rates = len(pooled) // 2, []
    for _ in range(draws):
        order = rng.permutation(len(pooled))
        tau = calibrate(pooled[order[:half]], alpha).value
        rates.append(float((pooled[order[half:]] > tau).mean()))
    return rates


def atom_table(values, tau):
    """Distinct-value structure of a max-score sample, plus the atom nearest to tau."""
    values = np.asarray(values, float)
    unique, counts = np.unique(np.round(values, 12), return_counts=True)
    nearest = int(np.argmin(np.abs(unique - tau)))
    return {"distinct": int(len(unique)), "tau": float(tau),
            "tau_atom": int(counts[nearest]), "tau_nearest_value": float(unique[nearest]),
            "at_or_above": int((values >= tau).sum()), "strictly_above": int((values > tau).sum()),
            "largest_atom": int(counts.max()), "largest_atom_share": float(counts.max() / len(values)),
            "top": [{"value": float(v), "count": int(c)} for v, c in zip(unique[-5:], counts[-5:])]}


def bundle(scores, truth):
    """Descending-score cumulative counts, so every threshold of one document is an index lookup."""
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    t = np.asarray(truth, bool)[order].astype(int)
    return {"cum_tp": np.cumsum(t), "cum_fp": np.cumsum(1 - t), "n_true": int(t.sum())}


def iou_at(scores, pack, tau):
    k = int(np.count_nonzero(np.asarray(scores, float) > tau))
    if k == 0:
        return 0.0
    tp, fp = int(pack["cum_tp"][k - 1]), int(pack["cum_fp"][k - 1])
    fn = pack["n_true"] - tp
    return tp / (tp + fp + fn) if tp + fp + fn else None


def iou_curve(scores, truths, thresholds):
    packs = [bundle(s, t) for s, t in zip(scores, truths)]
    return [float(np.mean([iou_at(s, p, tau) for s, p in zip(scores, packs)])) for tau in thresholds]


def at_alpha(fpr, iou, alpha):
    """IoU at the largest false-positive rate not above alpha; None when the grid never gets there."""
    rows = [v for f, v in zip(fpr, iou) if f <= alpha and v is not None]
    return float(rows[0]) if rows else None


def marginal_detection(density, rhos, y):
    """Posterior under model averaging over the rho grid instead of the likelihood argmax."""
    g = density(y)
    fitted = estimate_proportion(g)
    if not fitted.identifiable:
        return np.zeros(len(y)), fitted.value, np.zeros(len(rhos))
    lls = np.array([forward_log_likelihood(g, fitted.value, r) for r in rhos])
    weights = np.exp(lls - lls.max())
    weights /= weights.sum()
    gamma = sum(w * forward_backward(g, fitted.value, r).gamma for w, r in zip(weights, rhos))
    return np.clip(gamma, 0, 1), fitted.value, weights


class Scored:
    """Minimal stand-in for detector.Detection; only scores and epsilon are consumed downstream."""

    def __init__(self, scores, epsilon):
        self.scores, self.epsilon, self.rho, self.identifiable = scores, epsilon, None, True


def diagnose(directory, output, frozen_path=None, draws=200, grid=100, alpha=.05):
    directory = Path(project_path(directory))
    manifest = read_json(directory / "manifest.json")
    config = dict(manifest["config"])
    validate_l0(config)
    m, key = config["context_width"], manifest["key"]
    articles = load_articles(directory, key, m)
    by_role = defaultdict(list)
    for article in articles:
        by_role[article["split"]].append(article)
    for role in ROLES + ("calibration",):
        if not by_role[role]:
            raise ValueError(f"Missing role {role}.")

    calibration_y = np.concatenate([a["y"] for a in by_role["calibration"]])
    emissions = {name: fit_emission(name, calibration_y, config) for name in config["emission"]}
    detectors = {(name, method): Detector(emissions[name], method=method, rhos=config["rho_grid"],
                                          fixed_e=config["fixed_e"], fixed_rho=config["fixed_rho"],
                                          window=config["window"])
                 for name in emissions for method in METHODS}
    results = {role: infer(by_role[role], detectors, f"{role} 推断") for role in ROLES}
    frozen = read_json(Path(project_path(frozen_path))) if frozen_path else None

    report = {"config": {k: config.get(k) for k in ("emission", "alpha", "fraction_tau", "rho_grid", "tokens")},
              "draws": draws, "rules": {}, "null": {}, "atoms": {}, "curves": {}, "rho": {}}
    cut = config.get("fraction_tau", .5)
    scenarios = sorted({a["scenario"] for a in by_role["test"]})

    for name, method in detectors:
        tag = f"{name}|{method}"
        cal_max = maxima(results["null_cal"][(name, method)])
        test_max = maxima(results["null_test"][(name, method)])
        tau = calibrate(cal_max, alpha)
        pooled = np.concatenate([cal_max, test_max])
        rates = recalibrated_fp(pooled, draws, alpha)
        q = calibrate(frequent(results["null_cal"][(name, method)], cut), alpha).value
        flagged_p = int((frequent(results["null_test"][(name, method)], cut) > q).sum())
        rule = {"tau": tau.value, "rank": tau.rank, "count": tau.count, "finite_rank": tau.finite_rank}
        if frozen is not None:
            rule["frozen_tau"] = frozen["rules"][tag]["M"]["pooled"]["value"]
            rule["matches_frozen"] = bool(np.isclose(rule["frozen_tau"], tau.value, rtol=0, atol=1e-12))
        report["rules"][tag] = rule
        report["null"][tag] = {
            "fp_cal": float((cal_max > tau.value).mean()), "fp_test": float((test_max > tau.value).mean()),
            "cal_wilson": wilson(int((cal_max > tau.value).sum()), len(cal_max)),
            "test_wilson": wilson(int((test_max > tau.value).sum()), len(test_max)),
            "recal_mean": float(np.mean(rates)), "recal_p05": float(np.quantile(rates, .05)),
            "recal_p50": float(np.quantile(rates, .50)), "recal_p95": float(np.quantile(rates, .95)),
            "recal_above_alpha": float(np.mean(np.array(rates) > alpha)),
            "q_P": q, "fp_P_test": flagged_p / len(test_max), "fp_P_wilson": wilson(flagged_p, len(test_max))}
        report["atoms"][tag] = {"null_cal": atom_table(cal_max, tau.value),
                                "null_test": atom_table(test_max, tau.value)}

        thresholds = np.quantile(test_max, np.linspace(.80, .999, grid))
        fpr = [float((test_max > t).mean()) for t in thresholds]
        curves = {}
        for scenario in scenarios:
            rows = [a for a in by_role["test"] if a["scenario"] == scenario and a["truth"].any()]
            index = [i for i, a in enumerate(by_role["test"]) if a["scenario"] == scenario and a["truth"].any()]
            scores = [results["test"][(name, method)][i].scores for i in index]
            curves[scenario] = iou_curve(scores, [a["truth"] for a in rows], thresholds)
        report["curves"][tag] = {"thresholds": thresholds.tolist(), "fpr": fpr, "iou": curves,
                                 "iou_at_alpha": {s: at_alpha(fpr, v, alpha) for s, v in curves.items()}}

    rhos = list(config["rho_grid"])
    for name in emissions:
        block = defaultdict(dict)
        for role in ROLES:
            rows = by_role[role]
            block[role]["argmax"] = results[role][(name, "opt_hmm")]
            block[role]["marginal"], weights = [], []
            for article in rows:
                gamma, epsilon, w = marginal_detection(emissions[name], rhos, article["y"])
                block[role]["marginal"].append(Scored(gamma, epsilon))
                weights.append(w)
            block[role]["mean_weights"] = np.mean(weights, axis=0)
            for r in FIXED:
                det = Detector(emissions[name], method="fixed_rho", rhos=rhos, fixed_rho=r)
                block[role][f"fixed_{r}"] = [det.infer(a["y"]) for a in rows]
        entry = {}
        for variant in VARIANTS:
            cal_max = maxima(block["null_cal"][variant])
            tau = calibrate(cal_max, alpha).value
            ious = {}
            for scenario in scenarios:
                index = [i for i, a in enumerate(by_role["test"])
                         if a["scenario"] == scenario and a["truth"].any()]
                ious[scenario] = float(np.mean([
                    position_metrics(by_role["test"][i]["truth"], block["test"][variant][i].scores > tau)["iou"]
                    for i in index]))
            entry[variant] = {"tau": tau, "any_fp": float((maxima(block["null_test"][variant]) > tau).mean()),
                              **ious}
            if variant == "marginal":
                entry[variant]["mean_weights"] = block["null_cal"]["mean_weights"].tolist()
        report["rho"][name] = entry

    out = new_output(project_path(output))
    write_json(out / "diag.json", report)
    (out / "diag.md").write_text(render(report), encoding="utf-8")
    return {"report": str(out / "diag.md"), "rules": report["rules"]}


def render(report):
    alpha = report["config"]["alpha"]
    lines = ["# L0 诊断（离线重放）", "",
             f"- 重校准抽样次数：{report['draws']}；α = {alpha}；fraction_tau = {report['config']['fraction_tau']}",
             "- 只重算阈值与决策；不重新拟合发射、不重新生成文本。", "",
             "## 1. 阈值复核", "", "| 规则 | tau | rank | 与 frozen 一致 |", "|---|---:|---:|---|"]
    for tag, r in report["rules"].items():
        same = r.get("matches_frozen")
        lines.append(f"| {tag} | {r['tau']:.6f} | {r['rank']} | {'是' if same else '否' if same is False else '未提供'} |")
    lines += ["", "## 2. null 误报：单阈值 vs 重校准", "",
              "| 规则 | FP(null_cal) | FP(null_test) | 重校准 p5 | 中位 | p95 | P(FP>α) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for tag, n in report["null"].items():
        lines.append(f"| {tag} | {n['fp_cal']:.4f} | {n['fp_test']:.4f} | {n['recal_p05']:.4f} | "
                     f"{n['recal_p50']:.4f} | {n['recal_p95']:.4f} | {n['recal_above_alpha']:.3f} |")
    lines += ["", "## 3. 原子诊断（τ 附近的重复取值）", "",
              "| 规则 | 角色 | 不同取值 | τ 处原子 | ≥τ | >τ | 最大原子占比 |", "|---|---|---:|---:|---:|---:|---:|"]
    for tag, atoms in report["atoms"].items():
        for role, t in atoms.items():
            lines.append(f"| {tag} | {role} | {t['distinct']} | {t['tau_atom']} | {t['at_or_above']} | "
                         f"{t['strictly_above']} | {t['largest_atom_share']:.3f} |")
    first = next(iter(report["curves"].values()))["iou_at_alpha"]
    lines += ["", "## 4. 等误报工作点（按 null_test 的 max 分位取阈值）", "",
              "| 规则 | " + " | ".join(first) + " |",
              "|---|" + "---:|" * len(first)]
    for tag, c in report["curves"].items():
        cells = " | ".join("NA" if v is None else f"{v:.3f}" for v in c["iou_at_alpha"].values())
        lines.append(f"| {tag} | {cells} |")
    lines += ["", "## 5. rho 选择方式对比", ""]
    for name, variants in report["rho"].items():
        scenarios = sorted({k for v in variants.values() for k in v if k not in META})
        lines += [f"### {name}", "",
                  "| 变体 | tau | any-FP | " + " | ".join(scenarios) + " |",
                  "|---|---:|---:|" + "---:|" * len(scenarios)]
        for variant in VARIANTS:
            v = variants[variant]
            lines.append(f"| {variant} | {v['tau']:.4f} | {v['any_fp']:.4f} | "
                         + " | ".join(f"{v[s]:.3f}" for s in scenarios) + " |")
        weights = variants["marginal"].get("mean_weights")
        if weights:
            lines += ["", f"marginal 在 null_cal 上的平均 rho 权重：`{[round(w, 3) for w in weights]}`（网格 {report['config']['rho_grid']}）", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, help="评估数据目录（含 manifest/articles/positions）")
    parser.add_argument("--output", required=True, help="新的输出目录，必须为空")
    parser.add_argument("--frozen", default=None, help="可选：runs/<名称>/frozen.json，用于核对阈值")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--grid", type=int, default=100)
    args = parser.parse_args()
    result = diagnose(args.positions, args.output, args.frozen, args.draws, args.grid)
    print(f"report: {result['report']}")
    mismatched = [tag for tag, r in result["rules"].items() if r.get("matches_frozen") is False]
    if mismatched:
        print(f"阈值与 frozen 不一致：{mismatched}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
