"""Offline diagnostics for an evaluated L0 run; reuses frozen thresholds, never regenerates text.

Answers four questions the main report cannot:

1. Is the observed null any-FP (e.g. 8.5%) a sampling artefact of one threshold, or a real
   calibration shift? Recalibrate on random halves of the pooled 800 null articles and report
   the distribution of the resulting false-positive rate.
2. Does the threshold sit on an atom of the max-score distribution? Report the largest atom at
   tau and the counts strictly above/below it.
3. Does per-document rho selection beat a rho fixed at the model level? Recalibrate and re-score
   the same articles under argmax rho, marginal rho, and four fixed rho values.
4. Is the equal-false-positive ordering real? Resample articles (and the null articles that set
   each method's own tau) to put a paired 95% interval on every equal-FP IoU and on
   opt_hmm minus its best control in that cell.

The data directory normally carries manifest.json + articles.jsonl. If only positions.jsonl is
present (a pulled copy), pass --frozen so the frozen config can stand in for the manifest; the
protocol check then relies on eval having passed validate_l0 before writing that file.

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
from opt_hmm.io import new_output, read_json, read_jsonl, write_json
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


def positions_articles(directory):
    """Article list rebuilt from positions.jsonl alone, for data dirs without manifest/articles."""
    rows = defaultdict(list)
    for row in read_jsonl(directory / "positions.jsonl"):
        rows[row["article_id"]].append(row)
    articles = []
    for article_id, group in rows.items():
        group.sort(key=lambda r: r["t"])
        articles.append({"id": article_id, "split": group[0]["split"], "scenario": group[0]["scenario"],
                         "n_eff": len(group), "eps_true": None,
                         "y": np.array([r["Y"] for r in group], float),
                         "truth": np.array([r["truth"] for r in group], bool)})
    return articles


def padded_pack(scores, truths):
    """Per-article descending score / cumulative tp / cumulative fp, padded so one tau serves all rows."""
    values = [np.asarray(s, float) for s in scores]
    width, n = max(len(s) for s in values), len(values)
    pack = {"score": np.full((n, width), -np.inf), "tp": np.zeros((n, width), int),
            "fp": np.zeros((n, width), int), "n_true": np.zeros(n, int)}
    for i, (s, t) in enumerate(zip(values, truths)):
        order = np.argsort(-s, kind="stable")
        label = np.asarray(t, bool)[order].astype(int)
        pack["score"][i, :len(s)] = s[order]
        pack["tp"][i, :len(s)] = np.cumsum(label)
        pack["fp"][i, :len(s)] = np.cumsum(1 - label)
        pack["n_true"][i] = int(label.sum())
    return pack


def iou_rows(pack, tau):
    """Position IoU per article at one threshold; an article that flags nothing scores 0."""
    k = (pack["score"] > tau).sum(axis=1)
    out = np.zeros(len(k), float)
    hit = k > 0
    index = np.flatnonzero(hit)
    tp, fp = pack["tp"][index, k[hit] - 1], pack["fp"][index, k[hit] - 1]
    total = tp + fp + (pack["n_true"][index] - tp)
    out[index] = np.where(total > 0, tp / np.where(total > 0, total, 1), np.nan)
    return out


def mean_iou(pack, tau):
    values = iou_rows(pack, tau)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def tau_at_alpha(null_max, alpha, grid=100):
    """Same grid rule as the equal-FP section: largest null-FP not above alpha."""
    thresholds = np.quantile(null_max, np.linspace(.80, .999, grid))
    fpr = (null_max[None, :] > thresholds[:, None]).mean(axis=1)
    ok = np.flatnonzero(fpr <= alpha)
    return float(thresholds[ok[0]]) if len(ok) else None


def bootstrap_fixed(packs, taus, draws, seed=42):
    """Paired article-level bootstrap at each method's own realized tau.

    tau is not resampled: the equal-FP table picks it on the same null_test that defines the
    false-positive rate, so replicating that in-sample extremeness is what a null bootstrap
    cannot do (its order statistic is biased toward the centre). Holding tau fixed answers the
    question the claim needs: given the frozen operating point, how noisy is the mean IoU gap?
    """
    rng = np.random.default_rng(seed)
    tags = list(packs)
    scenarios = list(packs[tags[0]])
    out = {s: {} for s in scenarios}
    for s in scenarios:
        n = len(packs[tags[0]][s]["n_true"])
        counts = rng.multinomial(n, np.full(n, 1. / n), size=draws).astype(float)
        for tag in tags:
            values = np.nan_to_num(iou_rows(packs[tag][s], taus[tag]), nan=0.)
            out[s][tag] = counts @ values / n
    return out


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


def diagnose(directory, output, frozen_path=None, draws=200, grid=100, alpha=.05, boot=2000):
    directory = Path(project_path(directory))
    frozen = read_json(Path(project_path(frozen_path))) if frozen_path else None
    short = directory / "manifest.json"
    if short.exists():
        manifest = read_json(short)
        config = dict(manifest["config"])
        validate_l0(config)
        articles = load_articles(directory, manifest["key"], config["context_width"])
    else:
        if frozen is None:
            raise ValueError("数据目录没有 manifest.json：用 --frozen 指定 frozen.json 以取得冻结配置。")
        config = dict(frozen["config"])
        articles = positions_articles(directory)
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

    report = {"config": {k: config.get(k) for k in ("emission", "alpha", "fraction_tau", "rho_grid", "tokens")},
              "draws": draws, "rules": {}, "null": {}, "atoms": {}, "curves": {}, "rho": {}}
    cut = config.get("fraction_tau", .5)
    scenarios = sorted({a["scenario"] for a in by_role["test"]})
    samples, null_max = {}, {}

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
        curves, kept = {}, {}
        for scenario in scenarios:
            rows = [a for a in by_role["test"] if a["scenario"] == scenario and a["truth"].any()]
            index = [i for i, a in enumerate(by_role["test"]) if a["scenario"] == scenario and a["truth"].any()]
            scores = [results["test"][(name, method)][i].scores for i in index]
            curves[scenario] = iou_curve(scores, [a["truth"] for a in rows], thresholds)
            kept[scenario] = (scores, [a["truth"] for a in rows])
        samples[tag], null_max[tag] = kept, test_max
        report["curves"][tag] = {"thresholds": thresholds.tolist(), "fpr": fpr, "iou": curves,
                                 "iou_at_alpha": {s: at_alpha(fpr, v, alpha) for s, v in curves.items()}}

    if boot:
        report["boot"] = {"draws": boot, "seed": 42, "mode": "fixed_tau", "cells": {},
                          "strictly_best": 0, "point_best": 0, "pack_agrees": True}
        for emission in config["emission"]:
            tags = [t for t in samples if t.split("|")[0] == emission]
            packs = {t: {s: padded_pack(*samples[t][s]) for s in scenarios} for t in tags}
            observed = {t: report["curves"][t]["iou_at_alpha"] for t in tags}
            taus = {t: tau_at_alpha(null_max[t], alpha, grid) for t in tags}
            for tag in tags:
                for s in scenarios:
                    point = observed[tag][s]
                    if point is not None and abs(mean_iou(packs[tag][s], taus[tag]) - point) > 1e-12:
                        report["boot"]["pack_agrees"] = False
            values = bootstrap_fixed(packs, taus, boot)
            cells = report["boot"]["cells"].setdefault(emission, {})
            for scenario in scenarios:
                entry = {"methods": {}, "opt": f"{emission}|opt_hmm"}
                for tag in tags:
                    draws_tag = np.asarray(values[scenario][tag], float)
                    draws_tag = draws_tag[np.isfinite(draws_tag)]
                    entry["methods"][tag] = {
                        "iou": observed[tag][scenario],
                        "ci95": [float(np.quantile(draws_tag, .025)), float(np.quantile(draws_tag, .975))]
                        if len(draws_tag) else [None, None]}
                optimum = entry["opt"]
                others = [t for t in tags if t != optimum]
                best = max(others, key=lambda t: -np.inf if observed[t][scenario] is None
                           else observed[t][scenario])
                diff = np.asarray(values[scenario][optimum], float) - np.asarray(values[scenario][best], float)
                diff = diff[np.isfinite(diff)]
                entry["best_other"] = best
                entry["opt_minus_best"] = {
                    "value": (None if observed[optimum][scenario] is None or observed[best][scenario] is None
                              else observed[optimum][scenario] - observed[best][scenario]),
                    "ci95": [float(np.quantile(diff, .025)), float(np.quantile(diff, .975))]
                    if len(diff) else [None, None]}
                entry["opt_strictly_best"] = bool(len(diff) and np.quantile(diff, .025) > 0)
                entry["opt_point_best"] = bool(observed[optimum][scenario] is not None
                                               and observed[optimum][scenario] >= observed[best][scenario])
                report["boot"]["strictly_best"] += int(entry["opt_strictly_best"])
                report["boot"]["point_best"] += int(entry["opt_point_best"])
                entry["full_wm_note"] = "full_wm 的 truth 全为 1，IoU 恒等于 recall"
                cells[scenario] = entry

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
    boot = report.get("boot")
    if boot:
        scenarios = list(next(iter(boot["cells"].values())))
        lines += ["", f"## 4b. 等误报工作点的配对区间（{boot['draws']} 次文章级重采样，τ 固定）", "",
                  f"- 点估计下 opt_hmm 为该格最优：**{boot['point_best']}/8**；"
                  f"95% 区间严格为正：**{boot['strictly_best']}/8**。",
                  "- 每次重采样只抽测试文章，τ 固定在各方法实测的等误报工作点；两方法共用同一组下标，故为配对。",
                  "- τ 不重抽：该表的 τ 与误报率取自同一批 null_test，重抽会让序统计量向中心收缩、IoU 系统性抬高"
                  "（实测 beta/fixed_e 在 resample_0.3 上 0.470 → 区间 [0.494, 0.646]，区间不含自己的点估计）。"
                  "故这里只报“条件于 τ”的抽样误差，τ 本身的不确定性需另算。",
                  "- `full_wm` 的真值全为 1，IoU 恒等于 recall，这一列只反映“标得多不多”。"]
        if not boot["pack_agrees"]:
            lines += ["- **警告**：按 τ 重算的 IoU 与第 4 节不一致，4b 不可用。"]
        lines += ["", "| 场景 | 发射 | 最优对照 | Δ(opt−该对照) | 95% CI | 区间不含 0 |",
                  "|---|---|---|---:|---|---|"]
        for emission, cells in boot["cells"].items():
            for scenario, e in cells.items():
                d = e["opt_minus_best"]
                lo, hi = d["ci95"]
                value = "NA" if d["value"] is None else f"{d['value']:+.3f}"
                span = "NA" if lo is None else f"[{lo:+.3f}, {hi:+.3f}]"
                mark = "是" if e["opt_strictly_best"] else "否"
                lines.append(f"| {scenario} | {emission} | {e['best_other'].split('|')[1]} | "
                             f"{value} | {span} | {mark} |")
        for emission, cells in boot["cells"].items():
            tags = list(next(iter(cells.values()))["methods"])
            lines += ["", f"### {emission}：各方法的等误报 IoU（点估计 [95% CI]）", "",
                      "| 方法 | " + " | ".join(scenarios) + " |", "|---|" + "---|" * len(scenarios)]
            for tag in tags:
                row = []
                for scenario in scenarios:
                    m = cells[scenario]["methods"][tag]
                    lo, hi = m["ci95"]
                    row.append("NA" if m["iou"] is None else
                               f"{m['iou']:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{m['iou']:.3f}")
                lines.append(f"| {tag.split('|')[1]} | " + " | ".join(row) + " |")
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
    parser.add_argument("--boot", type=int, default=2000, help="等误报表的配对重采样次数；0 表示不做")
    args = parser.parse_args()
    result = diagnose(args.positions, args.output, args.frozen, args.draws, args.grid, boot=args.boot)
    print(f"report: {result['report']}")
    mismatched = [tag for tag, r in result["rules"].items() if r.get("matches_frozen") is False]
    if mismatched:
        print(f"阈值与 frozen 不一致：{mismatched}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
