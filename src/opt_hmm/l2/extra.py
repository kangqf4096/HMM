"""Extra arms on an existing L2 run: `fb` (Beta-shaped f1(e')) and the SPOT baseline (docs/2608.14906v2, Algorithm 1).

Both are fitted on the training side only (dev selection, null_cal calibration), written to `frozen_extra.json`, and
only then run once on the test side. `fe`, `m400` and `seedbs` are not recomputed: their per-article rows are read
from the reference run's `metrics.jsonl`, so every difference is a paired bootstrap on the same test articles.

SPOT here is the plug-in variant: eps_hat is eps_opt under the frozen stage-1 hist200 (as in the paper's real-LLM
experiments), the tail grid is u in [0.005, 0.98] with M = max(20, ceil(3.5 ln(n)^1.5)) points, eta = 0, and the
calibration constant C (lambda = C / ln n) is chosen on dev among the values whose null_cal any-FP is at most alpha.
"""
import copy
from math import ceil, log
from pathlib import Path
from time import perf_counter

import numpy as np

from ..calibration import Threshold
from ..detector import Detection
from ..io import new_output, provenance, read_jsonl, write_json
from ..l0.decide import rules_to_dict
from ..l0.metrics import position_metrics
from ..paths import project_path
from .config import validate_l2
from .data import load_side
from .evaluate import BLOCK_ORDER, TEST_SPLITS, Blocks, check_gate, load_frozen, rate_cell, score_rows
from .fit import check_side, fit_method
from .methods import DEV_CLASSES, MODES, Trainer

SPOT_C = np.logspace(-2, 1, 60)
U_MIN, U_MAX = .005, .98
REFERENCE = ("pd_m400", "pe_m400", "pd_fe", "pe_fe", "seedbs")


def beta_names():
    return tuple(f"{mode}_fb" for mode in MODES)


class Spot:
    """SPOT (Algorithm 1) on the pivots only; F0(y) = y, so the tail test is Y > 1 - n^-u."""

    def __init__(self, stage1, cap, C=None):
        self.stage1, self.cap, self.C = stage1, cap, C

    def grid(self, n):
        m = max(20, ceil(3.5 * log(n) ** 1.5))
        return np.linspace(U_MIN, U_MAX, m)

    def tail_curve(self, y):
        """T_hat(u) on the grid for one article, and eps_hat, identifiable."""
        n = len(y)
        value, ident, _ = self.stage1.opt(self.cap).estimate_many(np.bincount(
            np.minimum((y * 200).astype(int), 199), minlength=200)[None, :])
        eps = float(np.clip(value[0], 0., 1.))
        u = self.grid(n)
        cut = 1. - n ** -u
        s = np.array([(y > c).mean() for c in cut])
        return (1. - eps) * n ** -u / np.maximum(s, 1. / n), eps, bool(ident[0]), cut

    def prepare(self, docs):
        return [self.tail_curve(d["y"]) for d in docs]

    def detections(self, docs, prepared, C):
        out = []
        for doc, (curve, eps, ident, cut) in zip(docs, prepared):
            n = len(doc["y"])
            lam = C / log(n)
            ok = np.flatnonzero(curve <= lam)
            # Deviation from Algorithm 1: when no scanned u meets the criterion the paper falls back to the largest u;
            # here the article gets an empty output, otherwise C cannot control the null any-FP at all.
            threshold = cut[ok[0]] if len(ok) else None
            scores = (doc["y"] > threshold).astype(float) if ident and threshold is not None else np.zeros(n)
            out.append(Detection(scores, eps, 0., ident, 0., info={"threshold": threshold}))
        return out


def fixed_rules(alpha, n):
    """Rule dict for a method whose scores are already 0/1 decisions: flag when score > .5."""
    t = Threshold(.5, n, n, alpha, True)
    return {"tau_fraction": .5, "group_sizes": {}, "M": {"pooled": t}, "P": {"pooled": t}, "P_same": {"pooled": t}}


def select_spot(spot, splits, config):
    """C per criterion: 'doc' = null_cal any-FP <= alpha, 'position' = null_cal flagged-position share <= alpha.

    Returns {criterion: best row or None} and the full table; None means no C on the grid satisfies the criterion.
    """
    null, dev = splits["null_cal"], splits["dev"]
    prep_null, prep_dev = spot.prepare(null.docs), spot.prepare(dev.docs)
    alpha, table = config["alpha"], []
    best = {"doc": None, "position": None}
    for C in SPOT_C:
        dets = spot.detections(null.docs, prep_null, C)
        fp = float(np.mean([bool(d.scores.any()) for d in dets]))
        fpr = float(np.mean([float((d.scores > .5).mean()) for d in dets]))
        by_class = {}
        for doc, det in zip(dev.docs, spot.detections(dev.docs, prep_dev, C)):
            iou = position_metrics(doc["truth"], det.scores > .5)["iou"]
            if iou is not None and doc["cls"] in DEV_CLASSES:
                by_class.setdefault(doc["cls"], []).append(iou)
        classes = {c: float(np.mean(v)) for c, v in by_class.items()}
        iou = float(np.mean([classes[c] for c in DEV_CLASSES])) if len(classes) == len(DEV_CLASSES) else 0.
        row = {"C": float(C), "null_any_fp": fp, "null_position_fpr": fpr, "iou": iou, "iou_by_class": classes}
        table.append(row)
        for name, value in (("doc", fp), ("position", fpr)):
            if value <= alpha and (best[name] is None or iou > best[name]["iou"]):
                best[name] = row
    return best, table


def run_extra(frozen_path, train_dir, test_dir, reference_dir, output):
    started = perf_counter()
    frozen = load_frozen(frozen_path)
    train = load_side(train_dir)
    config = train["manifest"]["config"]
    validate_l2(config)
    check_side(train, config)
    out = new_output(project_path(output))
    inject = copy.deepcopy(config)
    inject["l2"]["arms"]["fb"] = dict(config["l2"]["arms"]["fe"])
    trainer = Trainer(inject, train)
    fitted = {name: fit_method(trainer, name) for name in beta_names()}
    stage1 = trainer.arm_stage1["fe"]
    spot = Spot(stage1, config["l2"]["cap_k0"])
    best, spot_table = select_spot(spot, train["splits"], config)
    spot_variants = {name: row for name, row in (("spot", best["doc"]), ("spot_fpr", best["position"])) if row}
    extra = {"stage": "extra", "train_key_id": train["manifest"]["key_id"], "reference_fit": str(project_path(frozen_path)),
             "methods": {n: {**f["method"].to_dict(), "rules": rules_to_dict(f["method"].rules)}
                         for n, f in fitted.items()},
             "spot": {"cap": spot.cap, "variants": {n: {"C": r["C"], "selection": r} for n, r in spot_variants.items()},
                      "infeasible": [n for n in ("spot", "spot_fpr") if n not in spot_variants]}}
    write_json(out / "frozen_extra.json", extra)
    write_json(out / "fit_extra.json", {
        "fb": {n: {"cap": f["method"].cap, "h": f["method"].h, "params": f["method"].params,
                   "tau": f["method"].rules["M"]["pooled"].value, "selection": f["selection"],
                   "beta_params": f["method"].kernel.params.tolist(), "neff": f["neff"]} for n, f in fitted.items()},
        "spot": {"selected": best, "table": spot_table}})

    test = load_side(test_dir)
    check_gate(frozen, test["manifest"]["config"], test["manifest"])
    splits = test["splits"]
    rows, null_rows = [], []
    for name, fit in fitted.items():
        method = fit["method"]
        for split in TEST_SPLITS:
            rows += score_rows(name, splits[split].docs, method.detections(splits[split]), method.rules)
        null_rows += score_rows(name, splits["null_test"].docs, method.detections(splits["null_test"]), method.rules)
    rules = fixed_rules(config["alpha"], len(train["splits"]["null_cal"]))
    for split in TEST_SPLITS + ("null_test",):
        docs = splits[split].docs
        prepared = spot.prepare(docs)
        for name, row in spot_variants.items():
            dets = spot.detections(docs, prepared, row["C"])
            (null_rows if split == "null_test" else rows).extend(score_rows(name, docs, dets, rules))
    reference = [r for r in read_jsonl(Path(project_path(reference_dir)) / "metrics.jsonl") if r["method"] in REFERENCE]
    ref_main = [r for r in reference if r["split"] != "null_test"]
    ref_null = [r for r in reference if r["split"] == "null_test"]
    all_rows, all_null = rows + ref_main, null_rows + ref_null
    names = list(beta_names()) + list(spot_variants) + list(REFERENCE)
    blocks = Blocks(all_rows, names, config["bootstrap"], config["l2"]["global_seed"])
    report = {"methods": names, "iou": {b: {m: blocks.summary(m, b, "iou") for m in names} for b in BLOCK_ORDER},
              "deltas": {}, "any_fp": {}}
    pairs = [(f"{mode}_fb", f"{mode}_fe") for mode in MODES] + [(f"{mode}_fb", f"{mode}_m400") for mode in MODES] + [
        ("pd_fb", "seedbs"), ("pe_fb", "seedbs")]
    pairs += [(v, ref) for v in spot_variants for ref in ("pd_fe", "pe_fe", "seedbs")]
    for a, b in pairs:
        report["deltas"][f"{a} - {b}"] = {blk: blocks.delta(a, b, blk) for blk in BLOCK_ORDER}
    for m in names:
        nulls = [r for r in all_null if r["method"] == m]
        report["any_fp"][m] = rate_cell(sum(r["M"] for r in nulls), len(nulls))
    report["seconds"] = perf_counter() - started
    from .report import jsonable
    report = jsonable(report)
    write_json(out / "report.json", report)
    (out / "report.md").write_text(render_extra(report, extra), encoding="utf-8")
    write_json(out / "provenance.json", provenance())
    return {"report": str(out / "report.md"), "seconds": report["seconds"]}


def render_extra(report, extra):
    from .report import ci_text, table
    names = report["methods"]
    out = ["# L2 补充臂：Beta 自适应 f1(e′)（fb）与 SPOT 基线", "",
           "fb 与 SPOT 在训练侧拟合、冻结后才读测试侧，只跑一次。fe、m400、seedbs 的逐篇结果取自参照运行，"
           "所有差值都是按 prompt 配对的 bootstrap。SPOT 是 plug-in 版本（ε̂ 用冻结的 hist200 做 ε_opt；无阈值满足准则时输出空，论文原为回退到最大 u），"
           "校准常数 C 在 dev 上选，限制 null_cal 的误报（两个口径，见下）；它不走后验阈值 τ，不是论文原协议的复现。", "",
           "SPOT 两个口径：`spot` 用 null_cal 文档级 any-FP ≤ α 选 C；`spot_fpr` 用 null_cal 逐位置误报"
           "（null 文章被标记位置的比例）≤ α 选 C。" + "；".join(f"{n}: C = {v['C']:.4g}" for n, v in extra["spot"]["variants"].items())
           + ("。无可行 C：" + "、".join(extra["spot"]["infeasible"]) if extra["spot"]["infeasible"] else "。"), "", "## IoU", "",
           table(["方法"] + list(BLOCK_ORDER), [[m] + [ci_text(report["iou"][b][m]) for b in BLOCK_ORDER] for m in names]), "",
           "## 配对差值（ΔIoU）", "",
           table(["对比"] + list(BLOCK_ORDER), [[k] + [ci_text(v[b]) for b in BLOCK_ORDER] for k, v in report["deltas"].items()]), "",
           "## null_test any-FP（冻结阈值）", "",
           table(["方法", "误报数/篇数", "any-FP", "Wilson 95%"],
                 [[m, f"{c['count']}/{c['n']}", f"{c['rate']:.3f}", f"[{c['wilson95'][0]:.3f}, {c['wilson95'][1]:.3f}]"]
                  for m, c in report["any_fp"].items()]), ""]
    return "\n".join(out)
