"""Mechanism check on the training side (plan 3.3): does entropy-selective editing change f1 of the survivors?

The predictions M1 and M2 are fixed in the plan before any data exists; this module only evaluates them.
Nothing here reads test data, and nothing here feeds back into a fitted parameter.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..io import new_output, write_json
from ..paths import project_path
from ..progress import tick
from .config import validate_l2
from .data import load_side

TAILS = (0.99, 0.9)
ENTROPY_EDGES = (0., .5, 1., 2., 3., np.inf)


def article_stats(docs):
    """Per article: [truth=1 positions, #(Y >= .99), #(Y >= .9)]."""
    rows = []
    for d in docs:
        y = d["y"][d["truth"]]
        if len(y):
            rows.append([len(y)] + [int((y >= q).sum()) for q in TAILS])
    return np.array(rows, float)


def ratio(stats):
    return stats[:, 1:].sum(axis=0) / stats[:, 0].sum()


def bootstrap_ratio(stats, draws, rng):
    idx = rng.integers(0, len(stats), size=(draws, len(stats)))
    picked = stats[idx]
    return picked[:, :, 1:].sum(axis=1) / picked[:, :, 0].sum(axis=1)[:, None]


def interval(values):
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def group_table(groups, pure_stats, draws, seed):
    rng = np.random.default_rng(seed)
    ref_point, ref_boot = ratio(pure_stats), bootstrap_ratio(pure_stats, draws, rng)
    out = {}
    for name, docs in groups.items():
        stats = article_stats(docs)
        if not len(stats):
            continue
        point, boot = ratio(stats), bootstrap_ratio(stats, draws, rng)
        out[name] = {"articles": int(len(stats)), "positions": int(stats[:, 0].sum()),
                     **{f"tail{int(q * 100)}": {"mean": float(point[i]), "ci95": interval(boot[:, i])}
                        for i, q in enumerate(TAILS)},
                     **{f"diff{int(q * 100)}": {"mean": float(point[i] - ref_point[i]),
                                                "ci95": interval(boot[:, i] - ref_boot[:, i])}
                        for i, q in enumerate(TAILS)}}
    return {"pure": {"articles": int(len(pure_stats)), "positions": int(pure_stats[:, 0].sum()),
                     **{f"tail{int(q * 100)}": {"mean": float(ref_point[i]), "ci95": interval(ref_boot[:, i])}
                        for i, q in enumerate(TAILS)}}, **out}


def entropy_tables(pure_docs, groups):
    """(2) pure Y against entropy bins; (3) entropy of edited positions and of survivors against all positions."""
    ys, hs = [], []
    for d in pure_docs:
        if d["entropy"] is not None:
            ys.append(d["y"])
            hs.append(d["entropy"])
    bins = []
    if ys:
        y, h = np.concatenate(ys), np.concatenate(hs)
        for lo, hi in zip(ENTROPY_EDGES[:-1], ENTROPY_EDGES[1:]):
            keep = (h >= lo) & (h < hi)
            bins.append({"entropy": [float(lo), None if np.isinf(hi) else float(hi)], "positions": int(keep.sum()),
                         "mean_y": float(y[keep].mean()) if keep.any() else None,
                         **{f"tail{int(q * 100)}": float((y[keep] >= q).mean()) if keep.any() else None for q in TAILS}})
    pure_mean = float(np.concatenate(hs).mean()) if hs else None
    edited = {}
    for name, docs in groups.items():
        docs = [d for d in docs if d["entropy"] is not None]
        if not docs:
            continue
        all_h = np.concatenate([d["entropy"] for d in docs])
        edit_h = np.array([e["entropy_nats"] for d in docs for e in d["edits"] if "entropy_nats" in e])
        surv_h = np.concatenate([d["entropy"][d["truth"]] for d in docs])
        edited[name] = {"articles": len(docs), "mean_entropy_all": float(all_h.mean()),
                        "mean_entropy_edited": float(edit_h.mean()) if len(edit_h) else None,
                        "mean_entropy_survivors": float(surv_h.mean()) if len(surv_h) else None,
                        "survivor_positions": int(len(surv_h))}
    return {"pure_by_entropy": bins, "pure_mean_entropy": pure_mean, "edits_and_survivors": edited}


def evaluate_predictions(table, config):
    m = config["l2"]["mechanism"]
    rates = sorted((float(n.split("_", 1)[1]), n) for n in table if n.startswith("r2_"))
    high = [(r, n) for r, n in rates if r in set(m["m1_rates"])]
    cond = {}
    for r, n in high:
        d = table[n]["diff99"]
        cond[n] = {"diff": d["mean"], "ci95": d["ci95"], "holds": bool(d["mean"] <= -m["min_diff"] and d["ci95"][1] < 0)}
    diffs = [table[n]["diff99"]["mean"] for _, n in rates]
    monotone = all(b <= a for a, b in zip(diffs[:-1], diffs[1:]))
    m1 = {"conditions": cond, "monotone": monotone, "diffs_by_rate": {n: table[n]["diff99"]["mean"] for _, n in rates},
          "holds": bool(cond) and all(c["holds"] for c in cond.values()) and monotone}
    m2c = {}
    for n in [k for k in table if k.startswith("r0_") or k == "splice_all"]:
        d = table[n]["diff99"]
        m2c[n] = {"diff": d["mean"], "ci95": d["ci95"],
                  "holds": bool(d["ci95"][0] <= 0 <= d["ci95"][1] and abs(d["mean"]) < m["min_diff"])}
    m2 = {"conditions": m2c, "holds": bool(m2c) and all(c["holds"] for c in m2c.values())}
    return {"M1": m1, "M2": m2}


def run_mechanism(train_dir, output, progress_path=None):
    side = load_side(train_dir)
    config = side["manifest"]["config"]
    validate_l2(config)
    if side["manifest"].get("side") != "train":
        raise ValueError("The mechanism check reads the training side only.")
    from .stages import default_progress_path, install_overall
    output = new_output(project_path(output))
    overall = install_overall(config, progress_path or default_progress_path(output), ("D1",))
    splits = side["splits"]
    groups = defaultdict(list)
    for d in splits["mix_r2"].docs:
        groups[d["scenario"]].append(d)
    for d in splits["mix_r0"].docs:
        groups[d["scenario"]].append(d)
    for d in splits["mix_splice"].docs:
        groups[f"splice_{d['scenario']}"].append(d)
    groups["splice_all"] = list(splits["mix_splice"].docs)
    for d in splits["mix_alt"].docs:
        groups[f"alt_{d['scenario']}"].append(d)
    m = config["l2"]["mechanism"]
    pure_docs = splits["cal_pure"].docs
    table = group_table(groups, article_stats(pure_docs), m["draws"], config["l2"]["global_seed"])
    result = {"groups": table, **entropy_tables(pure_docs, {k: v for k, v in groups.items()
                                                          if k.startswith(("r2_", "r0_"))}),
              **evaluate_predictions(table, config)}
    write_json(Path(output) / "mechanism.json", result)
    (Path(output) / "mechanism.md").write_text(render(result), encoding="utf-8")
    tick("D1")
    overall.finish()
    return {"mechanism": str(Path(output) / "mechanism.json"), "M1": result["M1"]["holds"], "M2": result["M2"]["holds"]}


def render(r):
    def ci(d):
        return f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"

    lines = ["# 机制对照（训练侧）", "", "| 组 | 文章 | 位置 | P(Y≥0.99) | P(Y≥0.9) | 与纯水印之差 (0.99) | 之差 (0.9) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, g in r["groups"].items():
        diff99 = ci(g["diff99"]) if "diff99" in g else ""
        diff90 = ci(g["diff90"]) if "diff90" in g else ""
        lines.append(f"| {name} | {g['articles']} | {g['positions']} | {g['tail99']['mean']:.4f} | "
                     f"{g['tail90']['mean']:.4f} | {diff99} | {diff90} |")
    lines += ["", f"**M1**（resample 熵选择：编辑率≥0.2 时差≥0.03 且区间不含 0，且随编辑率单调）：{'成立' if r['M1']['holds'] else '不成立'}；"
              f"单调 {r['M1']['monotone']}。",
              f"**M2**（R0 与 splice 的差区间含 0 且 |差|<0.03）：{'成立' if r['M2']['holds'] else '不成立'}。", ""]
    if r["pure_by_entropy"]:
        lines += ["## 纯水印的枢轴随熵分档", "", "| 熵区间 | 位置 | 均值 Y | P(Y≥0.99) | P(Y≥0.9) |", "|---|---:|---:|---:|---:|"]
        for b in r["pure_by_entropy"]:
            lines.append(f"| {b['entropy']} | {b['positions']} | {b['mean_y']} | {b['tail99']} | {b['tail90']} |")
        lines.append("")
    if r["edits_and_survivors"]:
        lines += ["## 被编辑位置与存活位置的熵", "", "| 组 | 全部位置均熵 | 被编辑位置均熵 | 存活位置均熵 | 存活位置数 |", "|---|---:|---:|---:|---:|"]
        for n, e in r["edits_and_survivors"].items():
            lines.append(f"| {n} | {e['mean_entropy_all']:.3f} | {e['mean_entropy_edited']} | {e['mean_entropy_survivors']} | {e['survivor_positions']} |")
        lines.append(f"\n纯水印全部位置均熵：{r['pure_mean_entropy']}")
    return "\n".join(lines) + "\n"
