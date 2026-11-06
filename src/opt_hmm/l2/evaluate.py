"""L2 test stage (plan 4.1 step 8): frozen methods, one run on the test side, blocks and paired bootstrap."""
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ..calibration import wilson
from ..io import new_output, read_json, write_json, write_jsonl
from ..l0.decide import decide, rules_from_dict
from ..l0.evaluate_v2 import recalibrated_false_positive
from ..l0.metrics import (bootstrap_indices, boundary_summary, changepoint_recovery, changepoints_from_labels,
                          position_metrics, rand_index, segment_report)
from ..paths import project_path
from ..progress import progress, tick
from .config import validate_l2
from .data import load_side
from .emission import KernelGrid
from .methods import HistMethod, Stage1, method_names
from .seedbs_run import SEEDBS, SeedBSRunner, paper_name, seedbs_names
from .stages import default_progress_path, install_overall

SATURATION = .99
BLOCKS = {"paper": ("test_paper", "paper"), "set1": ("test_paper", "set1"), "alt": ("test_alt", "alt"),
          "r2": ("test_r2", "r2"), "r0": ("test_r0", "r0")}
BLOCK_ORDER = ("paper", "alt", "r2", "r0")
TEST_SPLITS = ("test_paper", "test_alt", "test_r2", "test_r0")
TEST_STAGES = ("R1", "REP")
PAPER_RAND = ("paper", "alt", "set1")
FROZEN_KEYS = ("global_seed", "cap_k0", "cap_grid", "h_grid", "kernel_grid", "arms", "cap_sweep", "seedbs", "em")


def load_frozen(path):
    path = Path(project_path(path))
    return read_json(path / "frozen.json" if path.is_dir() else path)


def rebuild_methods(frozen):
    stage1 = {k: Stage1.from_dict(v) for k, v in frozen["stage1"].items()}
    out = {}
    for name, entry in frozen["methods"].items():
        k = str(frozen["l2"]["arms"][entry["arm"]]["pure_articles"])
        kernel = None if entry["kernel"] is None else KernelGrid.from_dict(entry["kernel"])
        out[name] = HistMethod(name, stage1[k], entry["cap"], kernel, entry["h"], entry["params"],
                               rules_from_dict(entry["rules"]))
    return out


def rebuild_seedbs(frozen):
    return {n: {"name": n, "B": s["B"], "zeta": s["zeta"], "rules": rules_from_dict(s["rules"])}
            for n, s in frozen["seedbs"].items()}


def score_rows(name, docs, detections, rules, seedbs=False):
    """One row per article under rule M, with the paper-style and the S-minus-B Rand index."""
    rows = []
    for doc, det in zip(docs, detections):
        labels, d = decide(det.scores, doc["n_eff"], rules)
        pm = position_metrics(doc["truth"], labels)
        seg = segment_report(doc["truth"], labels)
        pred = list(det.info["changepoints"]) if seedbs else changepoints_from_labels(labels)
        truth_cp = changepoints_from_labels(doc["truth"])
        paper = doc["paper_cps"]
        rows.append({"article_id": doc["id"], "group": doc["group"], "split": doc["split"],
                     "scenario": doc["scenario"], "cls": doc["cls"], "method": name, "n_eff": doc["n_eff"],
                     "eps_true": doc["eps_true"], "eps_hat": None if seedbs else det.epsilon,
                     "eps_error": None if seedbs else abs(det.epsilon - doc["eps_true"]),
                     "identifiable": det.identifiable, "tau": d["tau"], "max_score": d["max_score"], "M": d["M"],
                     "saturated": float((det.scores > SATURATION).mean()), "labels_flagged": int(labels.sum()),
                     "rand_paper": rand_index(paper, pred, doc["n_eff"]) if doc["cls"] in PAPER_RAND else None,
                     "rand_sb": rand_index(truth_cp, pred, doc["n_eff"]),
                     "paper_changepoints": len(paper) if doc["cls"] in PAPER_RAND else None,
                     "predicted_changepoints": len(pred), **pm,
                     **{k: seg[k] for k in ("true_segments", "predicted_segments", "recalled_segments",
                                            "segment_recall", "boundary_errors")},
                     **{f"sb_{k}": v for k, v in changepoint_recovery(truth_cp, pred).items()}})
    return rows


def mean_of(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


class Blocks:
    """Per-method metric matrices (prompts x scenarios) of each test block; one prompt is one bootstrap unit."""

    def __init__(self, rows, methods, draws, seed):
        self.draws, self.seed, self.boot = draws, seed, {}
        self.index, self.scenarios = defaultdict(dict), {}
        for r in rows:
            if r["method"] not in methods:
                continue
            self.index[(r["method"], r["split"], r["cls"])][(r["group"], r["scenario"])] = r
            seen = self.scenarios.setdefault((r["split"], r["cls"]), [])
            if r["scenario"] not in seen:
                seen.append(r["scenario"])

    def matrix(self, method, block, metric):
        split, cls = BLOCKS[block]
        scenarios = self.scenarios.get((split, cls), [])
        cells = self.index.get((method, split, cls), {})
        groups = sorted({g for g, _ in cells})
        out = np.full((len(groups), len(scenarios)), np.nan)
        for i, g in enumerate(groups):
            for j, s in enumerate(scenarios):
                r = cells.get((g, s))
                if r is not None and r.get(metric) is not None:
                    out[i, j] = r[metric]
        return out

    def group_means(self, method, block, metric, scenario=None):
        m = self.matrix(method, block, metric)
        if scenario is not None:
            m = m[:, [self.scenarios[BLOCKS[block]].index(scenario)]]
        count = (~np.isnan(m)).sum(axis=1)
        total = np.where(np.isnan(m), 0., m).sum(axis=1)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    def _indices(self, n, tag):
        key = (n, tag)
        if key not in self.boot:
            self.boot[key] = bootstrap_indices(n, self.draws, self.seed + sum(map(ord, tag)))
        return self.boot[key]

    def ci(self, values, tag):
        values = np.asarray(values, float)
        values = values[~np.isnan(values)]
        if not len(values):
            return {"mean": None, "ci95": [None, None], "groups": 0}
        means = values[self._indices(len(values), tag)].mean(axis=1)
        return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
                "groups": int(len(values))}

    def summary(self, method, block, metric, scenario=None):
        return self.ci(self.group_means(method, block, metric, scenario), f"{block}|{scenario}")

    def delta(self, a, b, block, metric="iou", scenario=None):
        return self.ci(self.group_means(a, block, metric, scenario) - self.group_means(b, block, metric, scenario),
                       f"{block}|{scenario}")


def rate_cell(count, n):
    return {"count": int(count), "n": int(n), "rate": count / n if n else None, "wilson95": wilson(count, n)}


def check_gate(frozen, config, manifest):
    if manifest.get("side") != "test":
        raise ValueError("The test stage reads the test side only.")
    if manifest["key_id"] != config["l2"]["sides"]["test"]["key_id"] or manifest["key_id"] == frozen["train_key_id"]:
        raise ValueError("Test positions must carry the test key, different from the training key.")
    if (manifest.get("l_top") or {}).get("L_top") != (frozen.get("l_top") or {}).get("L_top"):
        raise ValueError("Test and training sides were built with different L_top.")
    for key in FROZEN_KEYS:
        if frozen["l2"][key] != config["l2"][key]:
            raise ValueError(f"Configuration differs from the frozen one in l2.{key}.")
    for key in ("tokens", "context_width", "bins", "alpha"):
        if frozen["config"][key] != config[key]:
            raise ValueError(f"Configuration differs from the frozen one in {key}.")


def interpret(report):
    """C2a / C2b joint reading, written before any test number existed (plan 5.2)."""
    d = report["deltas"]["fe_vs_m400"]
    c2a = {mode: d[mode]["r2"]["ci95"][0] is not None and d[mode]["r2"]["ci95"][0] > 0 for mode in ("pd", "pe")}
    holds = all(c2a.values())
    r0_mean = [d[m]["r0"]["mean"] for m in ("pd", "pe")]
    paper_mean = [d[m]["paper"]["mean"] for m in ("pd", "pe")]
    nonpositive = lambda xs: all(x is not None and x <= 0 for x in xs)
    out = {"C2a": {"pd": c2a["pd"], "pe": c2a["pe"], "holds": holds}}
    if not holds:
        out["verdict"] = "C2a 不成立：f1(e') 在本数据上没有帮助，不改主线。"
    elif nonpositive(r0_mean) and nonpositive(paper_mean):
        out["verdict"] = "支持：收益来自熵选择性编辑，不来自额外的光滑或样本量；域外使用有代价（R0、论文块上 fe − m400 ≤ 0）。"
    elif not nonpositive(r0_mean):
        out["verdict"] = "来源未明：R0 上 fe − m400 > 0，收益不能归因于熵机制，不声称机制被证实。"
    else:
        out["verdict"] = ("来源未明（保守读法）：R0 上无收益但论文块上 fe − m400 > 0，计划的联合解读规则未覆盖此情形，"
                          "按未证实处理。")
    mech = report.get("mechanism")
    if mech is not None:
        out["M1"], out["M2"] = mech.get("M1", {}).get("holds"), mech.get("M2", {}).get("holds")
        if not out["M1"]:
            out["verdict"] += " 机制对照 M1 不成立：fe 仅作机制不存在时的对照，不当作新方案。"
        elif not out["M2"]:
            out["verdict"] += " 机制对照 M1 成立而 M2 不成立：偏离不来自熵选择，fe 的假设需重写（另起版本）。"
    return out


def build_test_report(config, frozen, side, rows, null_rows, fit_report, mechanism, seconds):
    methods = list(method_names()) + list(seedbs_names(config))
    hmm = list(method_names())
    blocks = Blocks(rows, methods, config["bootstrap"], config["l2"]["global_seed"])
    report = {"methods": methods, "hmm_methods": hmm, "blocks": {}, "tiers": {}, "deltas": {}, "any_fp": {}}
    for block in BLOCK_ORDER:
        report["blocks"][block] = {m: {"iou": blocks.summary(m, block, "iou"),
                                       "rand_paper": blocks.summary(m, block, "rand_paper") if block in PAPER_RAND else None,
                                       "rand_sb": blocks.summary(m, block, "rand_sb")} for m in methods}
    for block in ("paper", "alt", "r2", "r0"):
        for scenario in blocks.scenarios.get(BLOCKS[block], []):
            per = {}
            for m in methods:
                sel = [r for r in rows if r["method"] == m and r["scenario"] == scenario and r["split"] == BLOCKS[block][0]]
                per[m] = {"iou": blocks.summary(m, block, "iou", scenario),
                          "rand_paper": blocks.summary(m, block, "rand_paper", scenario) if block in PAPER_RAND else None,
                          "rand_sb": blocks.summary(m, block, "rand_sb", scenario),
                          "true_changepoints": mean_of([r["paper_changepoints"] for r in sel]),
                          "predicted_changepoints": mean_of([r["predicted_changepoints"] for r in sel]),
                          "eps_true": mean_of([r["eps_true"] for r in sel])}
            report["tiers"][scenario] = {"block": block, "methods": per}
    set1 = {m: {"recall": mean_of([r["recall"] for r in rows if r["method"] == m and r["cls"] == "set1"]),
                "false_changepoints": mean_of([r["predicted_changepoints"] for r in rows
                                               if r["method"] == m and r["cls"] == "set1"]),
                "rand_paper": blocks.summary(m, "set1", "rand_paper")} for m in methods}
    report["set1"] = set1
    d = report["deltas"]
    d["fe_vs_m400"] = {mode: {b: blocks.delta(f"{mode}_fe", f"{mode}_m400", b) for b in BLOCK_ORDER}
                       for mode in ("pd", "pe")}
    d["m400_vs_m100"] = {mode: {b: blocks.delta(f"{mode}_m400", f"{mode}_m100", b) for b in BLOCK_ORDER}
                         for mode in ("pd", "pe")}
    d["vs_seedbs"] = {}
    for m in hmm:
        d["vs_seedbs"][m] = {}
        for ref in seedbs_names(config):
            d["vs_seedbs"][m][ref] = {b: {"iou": blocks.delta(m, ref, b),
                                          "rand_paper": blocks.delta(m, ref, b, "rand_paper") if b in PAPER_RAND else None,
                                          "rand_sb": blocks.delta(m, ref, b, "rand_sb")} for b in BLOCK_ORDER}
    d["vs_seedbs_tiers"] = {m: {s: {"iou": blocks.delta(m, SEEDBS, t["block"], "iou", s),
                                    "rand_paper": blocks.delta(m, SEEDBS, t["block"], "rand_paper", s)
                                    if t["block"] in PAPER_RAND else None}
                                for s, t in report["tiers"].items() if t["block"] in ("paper", "alt")} for m in hmm}
    for m in methods:
        nulls = [r for r in null_rows if r["method"] == m]
        scores = [r["max_score"] for r in nulls]
        entry = {"frozen_tau": rate_cell(sum(r["M"] for r in nulls), len(nulls)),
                 "recalibrated": recalibrated_false_positive(scores, config["alpha"], seed=config["l2"]["global_seed"])}
        entry["C1"] = bool(entry["recalibrated"]["p95"] <= 2 * config["alpha"]
                           and entry["recalibrated"]["median"] <= config["alpha"])
        report["any_fp"][m] = entry
    report["secondary"] = {"eps_error": {m: mean_of([r["eps_error"] for r in rows if r["method"] == m]) for m in hmm},
                           "boundary": {m: boundary_summary([r["boundary_errors"] for r in rows if r["method"] == m])
                                        for m in methods},
                           "saturation": {m: mean_of([r["saturated"] for r in rows if r["method"] == m]) for m in hmm},
                           "tau": {m: mean_of([r["tau"] for r in rows if r["method"] == m]) for m in methods}}
    report["eps_true"] = {s: mean_of([r["eps_true"] for r in rows if r["scenario"] == s and r["method"] == hmm[0]
                                      and r["split"] == split])
                          for split in ("test_r2", "test_r0") for s in blocks.scenarios.get((split, split[5:]), [])}
    report["transition"] = None if fit_report is None else {
        n: {"params": frozen["methods"][n]["params"], "cap": frozen["methods"][n]["cap"], "h": frozen["methods"][n]["h"],
            "bootstrap": fit_report["methods"].get(n, {}).get("uncertainty", {}).get("bootstrap"),
            "equal_retention": fit_report["methods"].get(n, {}).get("uncertainty", {}).get("equal_retention")}
        for n in hmm}
    report["mechanism"] = mechanism
    report["interpretation"] = interpret(report)
    report["criteria"] = {"C1": {m: report["any_fp"][m]["C1"] for m in hmm}, "C2a": report["interpretation"]["C2a"]}
    report["seconds"] = seconds
    report["l_top"] = frozen.get("l_top")
    return report


def run_test(frozen_path, test_dir, output, workers=1, cache=None, progress_path=None, weights=None,
             mechanism=None, fit_report=None):
    started = perf_counter()
    frozen = load_frozen(frozen_path)
    frozen_dir = Path(project_path(frozen_path))
    frozen_dir = frozen_dir if frozen_dir.is_dir() else frozen_dir.parent
    side = load_side(test_dir)
    config = side["manifest"]["config"]
    validate_l2(config)
    check_gate(frozen, config, side["manifest"])
    splits = side["splits"]
    for need in TEST_SPLITS + ("null_test",):
        if need not in splits:
            raise ValueError(f"Test side lacks {need}.")
    output = new_output(project_path(output))
    overall = install_overall(config, progress_path or default_progress_path(output), TEST_STAGES, weights)
    methods, sb_states = rebuild_methods(frozen), rebuild_seedbs(frozen)
    rows, null_rows = [], []
    for number, name in enumerate(method_names(), 1):
        progress("评价", number, len(methods), name)
        method = methods[name]
        for split in TEST_SPLITS:
            stack = splits[split]
            rows += score_rows(name, stack.docs, method.detections(stack), method.rules)
            tick("R1", len(stack))
        null = splits["null_test"]
        null_rows += score_rows(name, null.docs, method.detections(null), method.rules)
        tick("R1", len(null))
    runner = SeedBSRunner(config, cache or output / "cache" / "seedbs", workers)
    docs = [d for split in TEST_SPLITS + ("null_test",) for d in splits[split].docs]
    pres = {}
    for name in seedbs_names(config):
        state = sb_states[name]
        if state["B"] not in pres:
            pres[state["B"]] = runner.precompute(docs, state["B"], stage="R1")
        else:
            tick("R1", len(docs))
        detections = runner.detections(docs, pres[state["B"]], state["zeta"])
        n_test = len(docs) - len(splits["null_test"])
        rows += score_rows(name, docs[:n_test], detections[:n_test], state["rules"], seedbs=True)
        null_rows += score_rows(name, docs[n_test:], detections[n_test:], state["rules"], seedbs=True)
    if fit_report is None and (frozen_dir / "fit_report.json").is_file():
        fit_report = read_json(frozen_dir / "fit_report.json")
    if isinstance(mechanism, (str, Path)):
        mechanism = read_json(project_path(mechanism))
    report = build_test_report(config, frozen, side, rows, null_rows, fit_report, mechanism, perf_counter() - started)
    from .report import jsonable, render_test
    report = jsonable(report)
    tick("REP")
    report["usage"] = overall.usage()
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render_test(report), encoding="utf-8")
    from .curves import render_curves
    svg = render_curves(report)
    if svg:
        (output / "curves.svg").write_text(svg, encoding="utf-8")
    slim = lambda rs: [{k: v for k, v in r.items() if k != "boundary_errors"} for r in rs]
    write_jsonl(output / "metrics.jsonl", slim(rows) + slim(null_rows))
    overall.finish()
    return {"report": str(output / "report.md"), "seconds": report["seconds"]}
