"""Length study report: the frozen-f1 group (B) against the retrained group (A), over text lengths.

Reading order, fixed before any new-length number was seen: IoU first, then the finer position and segment metrics
and the health of the posterior (thresholds, false change points, any-FP), then f1 itself. There is no numeric
pass line. Intervals are paired bootstrap 95% intervals over prompts (the same prompts at every length, so
differences between lengths and between groups are paired). Absolute scores of different lengths are not compared
with each other: change-point counts and the SeedBS minimum segment (2.5 x B) both move with the length.
"""
from pathlib import Path

import numpy as np

from ..io import new_output, read_json, read_jsonl, write_json
from ..l0.metrics import bootstrap_indices
from ..paths import project_path
from .data import load_side
from .emission import BINS, bin_counts, bin_index, hist_density
from .evaluate import BLOCK_ORDER, BLOCKS, Blocks
from .methods import Stage1
from .report import BLOCK_TITLE, ci_text, jsonable, num, table

FE = ("pd_fe", "pe_fe")
BASELINE = "seedbs"
METRICS = (("iou", "IoU"), ("recall", "recall"), ("precision", "precision"), ("f1", "F1"),
           ("segment_recall", "段召回"), ("rand_sb", "Rand(S−B)"), ("rand_paper", "Rand(论文)"))
COUNTS = (("sb_true_changepoints", "真值切点"), ("predicted_changepoints", "预测切点"))
TAILS = (.99, .9)
READING = ("先看 IoU：B（冻结 f1，只在新长度重估 a、π、τ）相对 A（在新长度整体重训）的配对差，以及 fe 相对 seedbs 的配对差；"
           "再看细化指标（recall、precision、F1、段召回、Rand index、切点数）和后验健康指标（τ、Setting 1 的假切点、误报），"
           "弄清差异来自哪里；最后看 f1 本身（hist200、尾部质量、e'、核网格）。不设数值红线。区间是按 prompt 配对的 bootstrap 95% 区间，"
           "* 表示区间不含 0。不同长度的绝对分数不直接比较（切点数与 SeedBS 最小段长都随长度变化），跨长度只读配对差。")


# ---------------------------------------------------------------------- distances between histograms
def prob(counts, pseudocount=.5):
    """Pooled hist200 probabilities of (articles, bins) counts."""
    return hist_density(np.asarray(counts, float).sum(axis=0), pseudocount) / BINS


def js(p, q):
    m = .5 * (p + q)
    return float(.5 * (p * np.log(p / m)).sum() + .5 * (q * np.log(q / m)).sum())


def tv(p, q):
    return float(.5 * np.abs(p - q).sum())


def corrected_js(cx, cy, splits=200, seed=0):
    """JS between two views (X and Y) of the same articles, with the sampling noise taken out.

    Articles are split into independent halves. The cross term compares X on one half with Y on the other; the
    noise term is the average of the X-versus-X and the Y-versus-Y half-against-half distances. Under the null both
    carry the same first-order noise, (1/n_x + 1/n_y) up to the same constant, so their difference estimates the
    divergence of the two population densities (it can be slightly negative by chance).
    """
    rng = np.random.default_rng(seed)
    n = len(cx)
    half = n // 2
    cross, noise = [], []
    for _ in range(splits):
        perm = rng.permutation(n)
        a, b = perm[:half], perm[half:2 * half]
        xa, xb, ya, yb = prob(cx[a]), prob(cx[b]), prob(cy[a]), prob(cy[b])
        cross.append(.5 * (js(xa, yb) + js(xb, ya)))
        noise.append(.5 * (js(xa, xb) + js(ya, yb)))
    cross, noise = np.array(cross), np.array(noise)
    excess = cross - noise
    px, py = prob(cx), prob(cy)
    return {"js_all": js(px, py), "tv_all": tv(px, py), "cross": float(cross.mean()), "noise": float(noise.mean()),
            "excess": float(excess.mean()), "excess_range": np.quantile(excess, [.025, .975]).tolist(),
            "articles": int(n), "splits": int(splits)}


def tail_difference(cx, cy, draws=1000, seed=0):
    """Tail masses P(Y >= t) of two views of the same articles and their paired article-bootstrap difference (Y - X)."""
    idx = bootstrap_indices(len(cx), draws, seed)
    out = {}
    for t in TAILS:
        first = int(round(t * BINS))
        tx, ty, nx, ny = cx[:, first:].sum(1), cy[:, first:].sum(1), cx.sum(1), cy.sum(1)
        diff = ty[idx].sum(1) / ny[idx].sum(1) - tx[idx].sum(1) / nx[idx].sum(1)
        out[f"{t:g}"] = {"x": float(tx.sum() / nx.sum()), "y": float(ty.sum() / ny.sum()),
                         "difference": {"mean": float(ty.sum() / ny.sum() - tx.sum() / nx.sum()),
                                        "ci95": np.quantile(diff, [.025, .975]).tolist()}}
    return out


def compare_views(cx, cy, splits, draws, seed):
    return {**corrected_js(cx, cy, splits, seed), "tails": tail_difference(cx, cy, draws, seed)}


def eprime(frozen, name, counts):
    """e' of pure-watermark articles under a frozen method's stage-1 f1 and cap."""
    entry = frozen["methods"][name]
    stage = Stage1.from_dict(frozen["stage1"][str(frozen["l2"]["arms"][entry["arm"]]["pure_articles"])])
    return stage.opt(entry["cap"]).estimate_many(counts)[0]


def describe(values):
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "mean": float(v.mean()), "p05": float(np.quantile(v, .05)),
            "below_0.3": float((v < .3).mean()), "mean_abs_error": float(np.abs(1. - v).mean())}


# ---------------------------------------------------------------------- loading
def load_entry(root, run, with_train=True):
    """Everything the report needs of one run name; group B is present when its test stage has run."""
    def path(name):
        return Path(root) / f"{run}-{name}"

    def group(frozen_dir, test_dir):
        return {"frozen": read_json(path(frozen_dir) / "frozen.json"), "report": read_json(path(test_dir) / "report.json"),
                "rows": read_jsonl(path(test_dir) / "metrics.jsonl")}

    train_manifest = read_json(path("train-pos") / "manifest.json")
    entry = {"run": run, "tokens": train_manifest["config"]["tokens"], "n_eff": None,
             "train_manifest": train_manifest, "test_manifest": read_json(path("test-pos") / "manifest.json"),
             "a": group("fit", "test")}
    entry["n_eff"] = entry["tokens"] - train_manifest["config"]["context_width"]
    if (path("reuse-test") / "report.json").is_file():
        entry["b"] = {**group("reuse", "reuse-test"), "detail": read_json(path("reuse") / "reuse_report.json")}
    if with_train:
        entry["train"] = load_side(path("train-pos"), validate=False)
    return entry


def tags(entry):
    return ("A", "B") if "b" in entry else ("A",)


def tagged_rows(entry):
    """Per-article rows of fe (both groups) and of seedbs (group A's), the method renamed `name@<group><tokens>`."""
    rows = []
    for kind in tags(entry):
        for r in entry[kind.lower()]["rows"]:
            if r["method"] in FE or (kind == "A" and r["method"] == BASELINE):
                rows.append({**r, "method": f"{r['method']}@{kind}{entry['tokens']}"})
    return rows


def variants(entry):
    """[(label, tagged method name)] in display order."""
    t = entry["tokens"]
    return [(f"{m}@{k}", f"{m}@{k}{t}") for m in FE for k in tags(entry)] + [(BASELINE, f"{BASELINE}@A{t}")]


def health_row(group, method):
    """Thresholds, false positives and the false change points of Setting 1 for one method of one group."""
    set1 = [r for r in group["rows"] if r["method"] == method and r["cls"] == "set1"]
    fp, sec = group["report"]["any_fp"][method], group["report"]["secondary"]
    return {"any_fp_frozen_tau": fp["frozen_tau"], "any_fp_recalibrated": fp["recalibrated"], "C1": fp["C1"],
            "tau": sec["tau"].get(method), "saturated": sec["saturation"].get(method),
            "eps_error": sec["eps_error"].get(method), "set1": group["report"]["set1"][method],
            "set1_no_changepoint": float(np.mean([r["predicted_changepoints"] == 0 for r in set1])) if set1 else None}


# ---------------------------------------------------------------------- the report
def f1_diagnostics(entries, reference, window_edges, splits=200, draws=1000):
    """f1 itself on the training side's pure-watermark articles: lengths against each other, positions within the
    longest text, e', and the kernel grid of fe."""
    by = {e["tokens"]: e for e in entries}
    lengths = sorted(by)
    m = by[reference]["a"]["frozen"]["config"]["context_width"]
    seed = by[reference]["a"]["frozen"]["l2"]["global_seed"]
    pure = {t: by[t]["train"]["splits"]["cal_pure"] for t in lengths}
    rows = {t: {d["group"]: i for i, d in enumerate(pure[t].docs)} for t in lengths}
    common = sorted(set.intersection(*[set(r) for r in rows.values()]))
    pick = {t: [rows[t][g] for g in common] for t in lengths}
    counts = {t: pure[t].counts[pick[t]] for t in lengths}
    out = {"articles": len(common), "lengths": {}, "windows": {}, "eprime": {}, "kernel": {}, "selected": {}}
    for i, a in enumerate(lengths):
        for b in lengths[i + 1:]:
            out["lengths"][f"{a}-{b}"] = compare_views(counts[a], counts[b], splits, draws, seed)
    longest = lengths[-1]
    edges = [m] + [e for e in window_edges if m < e < longest] + [longest]
    y = pure[longest].y[pick[longest]]
    windows = [(a, b) for a, b in zip(edges[:-1], edges[1:])]
    views = {f"{a}-{b}": bin_counts(bin_index(y[:, a - m:b - m])) for a, b in windows}
    names = list(views)
    out["windows"] = {"length": longest, "windows": names,
                      "pairs": {f"{p} vs {q}": compare_views(views[p], views[q], splits, draws, seed)
                                for i, p in enumerate(names) for q in names[i + 1:]}}
    source = by[reference]["a"]["frozen"]
    for name in FE:
        out["eprime"][name], out["kernel"][name], out["selected"][name] = {}, {}, {}
        base = eprime(source, name, counts[reference])
        for t in lengths:
            frozen = by[t]["a"]["frozen"]
            own, with_source = eprime(frozen, name, counts[t]), eprime(source, name, counts[t])
            diff = with_source - base
            out["eprime"][name][str(t)] = {
                "own_f1": describe(own), "reference_f1": describe(with_source),
                "shift_vs_reference_length": {"mean": float(diff.mean()), "sd": float(diff.std()),
                                              "corr": None if t == reference else float(np.corrcoef(with_source, base)[0, 1])}}
            entry = frozen["methods"][name]
            out["selected"][name][str(t)] = {"cap": entry["cap"], "h": entry["h"]}
            if t != reference:
                ka, kb = source["methods"][name]["kernel"], entry["kernel"]
                out["kernel"][name][str(t)] = [
                    {"e": float(e), "js": js(np.array(da) / BINS, np.array(db) / BINS), "neff_reference": float(na),
                     "neff_own": float(nb)}
                    for e, da, db, na, nb in zip(ka["grid"], ka["density"], kb["density"], ka["neff"], kb["neff"])]
    return out


def build_report(entries, reference=500, degenerate=None, window_edges=(150, 300, 500), splits=200, draws=1000):
    by = {e["tokens"]: e for e in entries}
    if reference not in by:
        raise ValueError(f"No run of length {reference} among the runs {sorted(by)}.")
    lengths = sorted(by)
    ref = by[reference]["a"]["frozen"]
    rows = [r for e in entries for r in tagged_rows(e)]
    blocks = Blocks(rows, sorted({r["method"] for r in rows}), ref["config"]["bootstrap"], ref["l2"]["global_seed"])
    all_metrics = METRICS + COUNTS
    report = {"reference": reference, "lengths": lengths, "reading": READING, "data": {}, "metrics": {}, "b_minus_a": {},
              "b_minus_a_scenarios": {}, "length_effect": {}, "against_seedbs": {}, "health": {}, "transition": {}}
    for t in lengths:
        e = by[t]
        report["data"][str(t)] = {
            "run": e["run"], "n_eff": e["n_eff"], "train_articles": e["train_manifest"]["articles"],
            "test_articles": e["test_manifest"]["articles"], "l_top": e["a"]["frozen"].get("l_top"),
            "filtered_train": e["train_manifest"].get("filtered"), "filtered_test": e["test_manifest"].get("filtered")}
        report["metrics"][str(t)] = {block: {label: {key: blocks.summary(name, block, key) for key, _ in all_metrics}
                                             for label, name in variants(e)} for block in BLOCK_ORDER}
        if "b" in e:
            report["b_minus_a"][str(t)] = {mode: {block: {key: blocks.delta(f"{mode}@B{t}", f"{mode}@A{t}", block, key)
                                                          for key, _ in all_metrics} for block in BLOCK_ORDER} for mode in FE}
            report["b_minus_a_scenarios"][str(t)] = {
                mode: {block: {s: blocks.delta(f"{mode}@B{t}", f"{mode}@A{t}", block, "iou", s)
                               for s in blocks.scenarios.get(BLOCKS[block], [])} for block in BLOCK_ORDER} for mode in FE}
        if t != reference:
            report["length_effect"][str(t)] = {
                name: {block: {key: blocks.delta(f"{name}@A{t}", f"{name}@A{reference}", block, key) for key, _ in all_metrics}
                       for block in BLOCK_ORDER} for name in FE + (BASELINE,)}
        report["against_seedbs"][str(t)] = {
            f"{mode}@{k}": {block: {key: blocks.delta(f"{mode}@{k}{t}", f"{BASELINE}@A{t}", block, key)
                                    for key in ("iou", "recall", "precision")} for block in BLOCK_ORDER}
            for mode in FE for k in tags(e)}
        health = {}
        for mode in FE:
            for k in tags(e):
                health[f"{mode}@{k}"] = health_row(e[k.lower()], mode)
        health[BASELINE] = health_row(e["a"], BASELINE)
        report["health"][str(t)] = health
        transition = {}
        for mode in FE:
            entry = {"A": {"params": e["a"]["frozen"]["methods"][mode]["params"],
                           "tau": e["a"]["frozen"]["methods"][mode]["rules"]["M"]["pooled"]["value"],
                           "cap": e["a"]["frozen"]["methods"][mode]["cap"], "h": e["a"]["frozen"]["methods"][mode]["h"]}}
            if "b" in e:
                d = e["b"]["detail"]["methods"][mode]
                entry["B"] = {"params": d["params"], "tau": d["tau"], "cap": d["cap"], "h": d["h"]}
            transition[mode] = entry
        report["transition"][str(t)] = transition
    report["degenerate"] = None if degenerate is None else {
        "rule": degenerate["rule"], "groups": len(degenerate["groups"]), "by_role": degenerate["groups_by_role"],
        "inputs": {k: {"tokens": v["tokens"], "side": v["side"], "articles": v["articles"], "flagged": len(v["degenerate"])}
                   for k, v in degenerate["inputs"].items()}}
    report["f1"] = f1_diagnostics(entries, reference, window_edges, splits, draws)
    return jsonable(report)


# ---------------------------------------------------------------------- rendering
LABEL = {"pd_fe@A": "PD fe（A 重训）", "pd_fe@B": "PD fe（B 冻结）", "pe_fe@A": "PE fe（A 重训）",
         "pe_fe@B": "PE fe（B 冻结）", BASELINE: "seedbs"}


def dtext(d, digits=3):
    if not d or d.get("mean") is None:
        return "NA"
    lo, hi = d["ci95"]
    return ci_text(d, digits) + (" *" if lo is not None and (lo > 0 or hi < 0) else "")


def render_data(report):
    out = ["## 1 数据", "", table(["长度", "n_eff", "训练侧文章", "评价侧文章", "L_top", "剔除（训练/评价）"],
           [[t, d["n_eff"], d["train_articles"], d["test_articles"], (d["l_top"] or {}).get("L_top", "NA"),
             f"{(d['filtered_train'] or {}).get('articles_removed', 0)} / {(d['filtered_test'] or {}).get('articles_removed', 0)}"]
            for t, d in report["data"].items()]), ""]
    g = report["degenerate"]
    if g:
        out += [f"退化规则：非水印位置之外的信号位置中，单个 1/200 箱占比 > {g['rule']['max_share']} 且至少 {g['rule']['min_positions']} 个位置；"
                f"任一长度、任一文章退化，整个 prompt 组在所有长度中剔除。共剔除 {g['groups']} 组，按角色：{g['by_role']}。", "",
                table(["输入", "长度", "侧", "文章", "命中文章"],
                      [[Path(k).name, v["tokens"], v["side"], v["articles"], v["flagged"]] for k, v in g["inputs"].items()]), ""]
    return out


def render_metrics(report):
    out = ["## 2 IoU 与细化指标（各长度，prompt 级均值与 95% 区间）", "",
           "precision 在没有任何标记时按 0 计；recall、F1、IoU 在没有存活信号时不计入。Rand(论文) 只在论文块与密度块有。", ""]
    for t, blocks in report["metrics"].items():
        for block in BLOCK_ORDER:
            cells = blocks[block]
            head = ["方法"] + [name for _, name in METRICS + COUNTS]
            out += [f"### 长度 {t}：{BLOCK_TITLE[block]}", "",
                    table(head, [[LABEL.get(label, label)] + [ci_text(cells[label][key]) for key, _ in METRICS + COUNTS]
                                 for label in cells]), ""]
    return out


def render_deltas(report):
    out = ["## 3 配对差", ""]
    if report["b_minus_a"]:
        out += ["### 3.1 冻结 f1 相对重训（B − A，同长度）", ""]
        for block in BLOCK_ORDER:
            cols = [(t, mode) for t in report["b_minus_a"] for mode in FE]
            out += [f"#### {BLOCK_TITLE[block]}", "",
                    table(["指标"] + [f"{t} {mode[:2].upper()}" for t, mode in cols],
                          [[name] + [dtext(report["b_minus_a"][t][mode][block][key]) for t, mode in cols]
                           for key, name in METRICS + COUNTS]), ""]
        out += ["### 3.2 按场景的 IoU（B − A）", ""]
        for t, modes in report["b_minus_a_scenarios"].items():
            for block in BLOCK_ORDER:
                scenarios = list(modes[FE[0]][block])
                out += [f"#### 长度 {t}，{BLOCK_TITLE[block]}", "",
                        table(["场景"] + [m[:2].upper() for m in FE],
                              [[s] + [dtext(modes[m][block][s]) for m in FE] for s in scenarios]), ""]
    if report["length_effect"]:
        out += [f"### 3.3 长度本身的影响（A(长度) − A({report['reference']})，同一批 prompt）", ""]
        for block in BLOCK_ORDER:
            cols = [(t, name) for t in report["length_effect"] for name in FE + (BASELINE,)]
            out += [f"#### {BLOCK_TITLE[block]}", "",
                    table(["指标"] + [f"{t} {n[:2].upper() if n != BASELINE else n}" for t, n in cols],
                          [[name] + [dtext(report["length_effect"][t][n][block][key]) for t, n in cols]
                           for key, name in METRICS + COUNTS]), ""]
    out += ["### 3.4 fe 相对 seedbs（同长度；跨长度可比的量）", ""]
    for block in BLOCK_ORDER:
        cols = [(t, v) for t in report["against_seedbs"] for v in report["against_seedbs"][t]]
        out += [f"#### {BLOCK_TITLE[block]}", "",
                table(["指标"] + [f"{t} {v[:2].upper()}{v[-2:]}" for t, v in cols],
                      [[key] + [dtext(report["against_seedbs"][t][v][block][key]) for t, v in cols]
                       for key in ("iou", "recall", "precision")]), ""]
    return out


def render_health(report):
    out = ["## 4 后验与误报的健康指标", "",
           "τ 在 A、B 两组中都在各自长度的 null_cal 上重新校准，所以 any-FP 反映新长度上的校准质量，不是 τ 的跨长度迁移。", ""]
    for t, health in report["health"].items():
        out += [f"### 长度 {t}", "",
                table(["方法", "τ", "null any-FP（冻结 τ）", "重校准 p95 / 中位", "C1", "饱和位置", "ε̂ 误差",
                       "Setting 1 位置 recall", "Setting 1 假切点", "Setting 1 无切点文章"],
                      [[LABEL.get(label, label), num(h["tau"], 4),
                        f"{num(h['any_fp_frozen_tau']['rate'])} [{num(h['any_fp_frozen_tau']['wilson95'][0])}, {num(h['any_fp_frozen_tau']['wilson95'][1])}]",
                        f"{num(h['any_fp_recalibrated']['p95'])} / {num(h['any_fp_recalibrated']['median'])}",
                        "通过" if h["C1"] else "未通过", num(h["saturated"]), num(h["eps_error"]),
                        num(h["set1"]["recall"]), num(h["set1"]["false_changepoints"], 2), num(h["set1_no_changepoint"])]
                       for label, h in health.items()]), ""]
    out += ["### 转移参数与阈值", "",
            table(["长度", "方法", "组", "cap", "h", "a00", "a11", "π1", "τ"],
                  [[t, mode, k, "不截断" if v["cap"] is None else v["cap"], "∞" if v["h"] is None else v["h"],
                    num(v["params"]["a00"], 4), num(v["params"]["a11"], 4),
                    num(v["params"]["pi1"], 3) if mode.startswith("pe") else "-", num(v["tau"], 4)]
                   for t, modes in report["transition"].items() for mode, groups in modes.items() for k, v in groups.items()]), ""]
    return out


def setting_text(entry):
    return f"cap={'不截断' if entry['cap'] is None else entry['cap']}, h={'∞' if entry['h'] is None else entry['h']}"


def render_f1(report):
    f = report["f1"]
    out = ["## 5 f1 本身（训练侧纯水印文章）", "",
           f"{f['articles']} 篇（所有长度共有的 prompt 组）。距离是 JS（nat）。同一批文章在两个视图里有重叠的位置时，“JS（全部文章）”偏小，"
           "不能直接和噪声比，以“校正后”为准：它是跨视图距离（视图 X 取一半文章、视图 Y 取另一半文章）减去同视图的二分噪声，近似两个总体密度"
           "之间的真实距离，偶尔为负属正常。尾部差为后者减前者，区间是按文章配对的 bootstrap。", ""]

    def pair_rows(pairs):
        rows = []
        for name, d in pairs.items():
            t99, t90 = d["tails"]["0.99"], d["tails"]["0.9"]
            rows.append([name, num(d["js_all"], 5), num(d["noise"], 5), num(d["excess"], 5),
                         f"[{num(d['excess_range'][0], 5)}, {num(d['excess_range'][1], 5)}]",
                         f"{num(t99['x'])}→{num(t99['y'])}  {ci_text(t99['difference'])}",
                         f"{num(t90['x'])}→{num(t90['y'])}  {ci_text(t90['difference'])}"])
        return rows

    head = ["对比", "JS（全部文章）", "二分噪声", "校正后", "校正后范围（二分）", "P(Y≥0.99)", "P(Y≥0.9)"]
    out += ["### 5.1 不同长度的 hist200（前一个长度对后一个）", "", table(head, pair_rows(f["lengths"])), ""]
    w = f["windows"]
    out += [f"### 5.2 位置窗口（长度 {w['length']}，token 位置）", "", table(head, pair_rows(w["pairs"])), ""]
    out += ["### 5.3 e'（纯水印文章，真值 ε = 1）", "",
            "A 自己的 f1 = 该长度自己重训的 f1；B 冻结的 f1 = 参考长度冻结的 f1。偏移是后者下该长度 e' 减参考长度 e'。", ""]
    rows = []
    for name, per in f["eprime"].items():
        for t, d in per.items():
            for kind, label in (("own_f1", "A 自己的 f1"), ("reference_f1", "B 冻结的 f1")):
                s = d[kind]
                shift = d["shift_vs_reference_length"]
                rows.append([name, t, label, num(s["median"]), num(s["mean"]), num(s["p05"]), num(s["below_0.3"]),
                             num(s["mean_abs_error"]),
                             "-" if kind == "own_f1" else f"{num(shift['mean'])}（sd {num(shift['sd'])}）"])
    out += [table(["方法", "长度", "f1", "中位", "均值", "5% 分位", "e'<0.3 占比", "平均偏离 1", "较参考长度的偏移"], rows), ""]
    out += ["### 5.4 fe 核网格（各 e 格点，参考长度对该长度）", "",
            "JS 未做噪声校正，只在 n_eff 足够处可读；低 e 端 n_eff 由少数文章支撑。", ""]
    for name, per in f["kernel"].items():
        for t, grid in per.items():
            ref, own = f["selected"][name][str(report["reference"])], f["selected"][name][t]
            out += [f"#### {name}，长度 {t}（所选：参考 {setting_text(ref)}；该长度 {setting_text(own)}）", "",
                    table(["e"] + [f"{g['e']:.2f}" for g in grid],
                          [["JS"] + [num(g["js"], 4) for g in grid], ["n_eff（参考）"] + [f"{g['neff_reference']:.0f}" for g in grid],
                           ["n_eff（该长度）"] + [f"{g['neff_own']:.0f}" for g in grid]]), ""]
    return out


def render(report):
    out = ["# L2 文本长度稳健性：冻结 f1（B）与重训 f1（A）", "", "## 0 读法（先于结果写定）", "", report["reading"], ""]
    out += render_data(report) + render_metrics(report) + render_deltas(report) + render_health(report) + render_f1(report)
    out += ["## 6 局限", "",
            "- 不同长度的文本是同一批 prompt、同一生成的前缀，评价侧是同一批文章；500 长度的评价结果此前已被看过，本报告全部是描述性的，"
            "不据此回头修改任何冻结规则。",
            "- B 组沿用的是参考长度的 f1（stage 1、核网格、cap、h），a、π、τ 在新长度重估；SeedBS 的 (B, ζ) 与 c 用 A 组在该长度的结果。",
            "- 剔除退化 prompt 组后 null 文章也被剔除，any-FP 只对非退化文本成立。评价侧只扫描水印文本与 null 文本，U 文本的退化不在规则内。",
            "- 区间假定 prompt 之间独立；同一 prompt 的不同场景共享来源文本，已按 prompt 配对。", ""]
    return "\n".join(out)


def run_lengths(runs, output, reference_tokens=500, degenerate=None, window_edges=(150, 300, 500)):
    root = project_path("runs")
    entries = sorted((load_entry(root, run) for run in runs), key=lambda e: e["tokens"])
    flagged = read_json(project_path(degenerate)) if degenerate else None
    report = build_report(entries, reference_tokens, flagged, tuple(window_edges))
    out = new_output(project_path(output))
    write_json(out / "report.json", report)
    (out / "report.md").write_text(render(report), encoding="utf-8")
    return {"report": str(out / "report.md"), "lengths": report["lengths"]}
