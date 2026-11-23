"""Report of the l2-extra stage: IoU per block, paired differences, the 2 x 2 contrasts and false-positive rates."""
from ..l0.evaluate_v2 import recalibrated_false_positive
from .evaluate import BLOCK_ORDER, Blocks, rate_cell
from .report import BLOCK_TITLE, ci_text, num, table

HMM_DOC = ("pd_fe", "pe_fe", "pd_m400", "pe_m400", "seedbs", "pd_fb", "pe_fb", "pd_bm", "pe_bm")
BASELINES = ("spot_opt", "spot_ini", "win_trgof", "win_ars")
TOKEN_ORDER = ("pd_fe", "pe_fe", "pd_m400", "pe_m400", "pd_fb", "pe_fb") + BASELINES
LABEL = {"fe": "hist200 核自适应", "m400": "固定 hist200", "fb": "Beta 核自适应", "bm": "固定 Beta",
         "seedbs": "SeedBS-NOT（Li 等 2024）", "spot_opt": "SPOT-plugin（ε_opt）", "spot_ini": "SPOT（尾部计数 ε̂）",
         "win_trgof": "窗口 Tr-GoF", "win_ars": "窗口 Aaronson 和"}


def label(name):
    base = name[:-4] if name.endswith("_tok") else name
    arm = base.split("_", 1)[1] if base[:3] in ("pd_", "pe_") else base
    return f"`{name}`（{LABEL.get(arm, arm)}）"


def contrast(blocks, plus, minus, block):
    values = sum(blocks.group_means(m, block, "iou") for m in plus) - sum(blocks.group_means(m, block, "iou") for m in minus)
    return blocks.ci(values, f"{block}|None")


def pairs_for(names):
    out = [(f"{m}_fb", f"{m}_fe") for m in ("pd", "pe")] + [(f"{m}_fb", f"{m}_bm") for m in ("pd", "pe")]
    out += [(f"{m}_bm", f"{m}_m400") for m in ("pd", "pe")] + [(f"{m}_fb", f"{m}_m400") for m in ("pd", "pe")]
    out += [(f"{m}_fe", f"{m}_m400") for m in ("pd", "pe")]
    out += [(a, b) for b in BASELINES for a in ("pd_fe", "pe_fe", "seedbs")]
    out += [(f"{a}_tok", f"{b}_tok") for b in BASELINES for a in ("pd_fe", "pe_fe")]
    return [(a, b) for a, b in out if a in names and b in names]


def build_report(config, rows, null_rows, meta, selection):
    present = {r["method"] for r in rows}
    doc = [m for m in HMM_DOC + BASELINES if m in present]
    token = [f"{m}_tok" for m in TOKEN_ORDER if f"{m}_tok" in present]
    names = doc + token
    blocks = Blocks(rows, names, config["bootstrap"], config["l2"]["global_seed"])
    report = {"doc": doc, "token": token, "meta": meta,
              "iou": {b: {m: blocks.summary(m, b, "iou") for m in names} for b in BLOCK_ORDER},
              "deltas": {f"{a} - {b}": {k: blocks.delta(a, b, k) for k in BLOCK_ORDER} for a, b in pairs_for(names)},
              "factorial": {}, "null": {}, "selection": selection}
    for mode in ("pd", "pe"):
        if all(f"{mode}_{a}" in present for a in ("fe", "m400", "fb", "bm")):
            report["factorial"][mode] = {k: contrast(blocks, (f"{mode}_fe", f"{mode}_bm"), (f"{mode}_m400", f"{mode}_fb"), k)
                                         for k in BLOCK_ORDER}
    alpha, seed = config["alpha"], config["l2"]["global_seed"]
    for m in names:
        rs = [r for r in null_rows if r["method"] == m]
        entry = {"any_fp": rate_cell(sum(r["M"] for r in rs), len(rs)),
                 "position_fpr": sum(r["labels_flagged"] for r in rs) / sum(r["n_eff"] for r in rs)}
        if meta.get(m, {}).get("criterion", "doc") == "doc":
            entry["recalibrated"] = recalibrated_false_positive([r["max_score"] for r in rs], alpha, seed=seed)
            entry["C1"] = bool(entry["recalibrated"]["p95"] <= 2 * alpha and entry["recalibrated"]["median"] <= alpha)
        report["null"][m] = entry
    return report


def config_text(cfg):
    show = {"cap": lambda v: "不截断" if v is None else v, "h": lambda v: "∞" if v is None else v}
    return "，".join(f"{k}={show.get(k, lambda v: v)(v)}" for k, v in (cfg or {}).items()) or "-"


def iou_table(r, names):
    return table(["方法", "配置"] + [BLOCK_TITLE[b] for b in BLOCK_ORDER],
                 [[label(m), config_text(r["meta"].get(m, {}).get("config"))] + [ci_text(r["iou"][b][m]) for b in BLOCK_ORDER]
                  for m in names])


def render_extra(r):
    sel = r["selection"]
    out = ["# L2 补充实验：Beta 自适应 f1(e′) 与文献基线", "",
           "fb、bm 与基线在训练侧拟合并冻结（frozen_extra.json）后才读测试侧，只跑一次。fe、m400、seedbs 的逐篇结果取自"
           "参照运行，所有差值都是按 prompt 配对的 bootstrap（95% 区间）。指标是逐篇 token 集合 IoU。", "",
           "## 1 文档级口径（null_cal any-FP ≤ α，项目主协议）", "", iou_table(r, r["doc"]), ""]
    out += ["### 配对差值 ΔIoU（前者 − 后者）", "",
            table(["对比"] + [BLOCK_TITLE[b] for b in BLOCK_ORDER],
                  [[k] + [ci_text(v[b]) for b in BLOCK_ORDER] for k, v in r["deltas"].items() if "_tok" not in k]), ""]
    if r["factorial"]:
        out += ["### 因子分解：自适应对 hist 与对 Beta 的收益是否一样", "",
                "(fe − m400) − (fb − bm)：为正表示核自适应对 hist200 的收益大于对 Beta 的收益。", "",
                table(["π"] + [BLOCK_TITLE[b] for b in BLOCK_ORDER],
                      [[m] + [ci_text(v[b]) for b in BLOCK_ORDER] for m, v in r["factorial"].items()]), ""]
    out += ["### null_test 误报（冻结 τ）", "",
            table(["方法", "文档级 any-FP", "Wilson 95%", "逐位置误报", "C1"],
                  [[label(m), f"{e['any_fp']['count']}/{e['any_fp']['n']} = {num(e['any_fp']['rate'])}",
                    f"[{num(e['any_fp']['wilson95'][0])}, {num(e['any_fp']['wilson95'][1])}]", num(e["position_fpr"], 4),
                    {True: "通过", False: "未通过"}.get(e.get("C1"), "-")] for m, e in r["null"].items() if m in r["doc"]]), ""]
    if r["token"]:
        out += ["## 2 逐位置口径（null_cal 上被标记位置占比 ≤ α）", "",
                "这是比文档级宽松的准则：每篇 null 文章约 500 个位置，几乎必然有位置被标记，所以文档级 any-FP 会很高。"
                "HMM 方法沿用冻结参数，只重新校准 τ；基线在该口径下重新在 dev 上调参。seedbs 不在此口径内。", "",
                iou_table(r, r["token"]), "",
                "### 配对差值 ΔIoU（前者 − 后者）", "",
                table(["对比"] + [BLOCK_TITLE[b] for b in BLOCK_ORDER],
                      [[k] + [ci_text(v[b]) for b in BLOCK_ORDER] for k, v in r["deltas"].items() if "_tok" in k]), "",
                "### null_test 误报（冻结 τ）", "",
                table(["方法", "逐位置误报", "文档级 any-FP"],
                      [[label(m), num(e["position_fpr"], 4), num(e["any_fp"]["rate"])] for m, e in r["null"].items()
                       if m in r["token"]]), ""]
    out += ["## 3 拟合与选择（训练侧）", ""]
    rows = []
    for name, f in sel["hmm"].items():
        best = f["selection"]["best"] or {}
        p = f["params"]
        rows.append([f"`{name}`", config_text({"cap": best.get("cap"), "h": best.get("h")}), num(p["a00"], 4), num(p["a11"], 4),
                     num(p["pi1"], 3) if name.startswith("pe_") else "-", num(f["tau"], 4), num(best.get("iou"))])
    out += [table(["方法", "cap, h", "a00", "a11", "π1(PE)", "τ", "dev 3 类均值 IoU"], rows), ""]
    for key, tab in sel["baselines"].items():
        usable = [x for x in tab if x["ok"]]
        top = max(usable, key=lambda x: x["iou"]) if usable else None
        out.append(f"- `{key}`：" + (f"{config_text(top['params'])}，τ = {num(top['tau'], 4)}，dev IoU {num(top['iou'])}"
                                     if top else "无可用配置（τ ≥ 1 或 dev 上无标记位置）"))
    out.append("")
    for name, f in sel["hmm"].items():
        if f["beta_vs_hist"] and name.startswith("pd_"):
            out += [f"### Beta 与同权重 hist200 的偏离（`{name}`，总变差，含 hist 的抽样噪声）", "",
                    table(["e", "n_eff", "a", "b", "TV"], [[num(x["e"], 2), f"{x['neff']:.0f}", num(x["a"]), num(x["b"]), num(x["tv"], 4)]
                                                           for x in f["beta_vs_hist"]]), ""]
    out += ["## 4 参照复现", "",
            f"训练侧 stage-1 计数与参照拟合一致：{r['reference_check']['stage1_counts_equal']}；"
            f"fe 核密度最大偏差 {r['reference_check']['fe_kernel_max_abs_diff']:.2g}。", "",
            "基线的定义、与论文协议的差别与未纳入的方法见 docs/l2_extra.md。", ""]
    return "\n".join(out)
