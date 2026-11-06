"""L2 reports: dicts first (JSON-serialisable), markdown rendered from the dicts only."""
import math
from collections import defaultdict

import numpy as np

from .methods import DEV_CLASSES
from .seedbs_run import SEEDBS

BLOCK_TITLE = {"paper": "论文块（Setting 2–4）", "alt": "密度块（等长交替段）", "r2": "R2 块（resample，熵选择性）",
               "r0": "R0 块（resample，位置与熵无关）"}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    return x


def num(v, digits=3):
    return "NA" if v is None else f"{v:.{digits}f}"


def ci_text(d, digits=3):
    if not d or d.get("mean") is None:
        return "NA"
    lo, hi = d["ci95"]
    return f"{num(d['mean'], digits)} [{num(lo, digits)}, {num(hi, digits)}]"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------- fit report
def build_fit_report(config, side, trainer, fitted, seedbs_states, seedbs_table, cov, f0, seconds):
    scen = defaultdict(list)
    for split, stack in side["splits"].items():
        for d in stack.docs:
            scen[(split, d["scenario"])].append(d["eps_true"])
    methods = {}
    for name, f in fitted.items():
        m, em = f["method"], f["em"]
        methods[name] = {
            "arm": m.arm, "mode": m.mode, "cap": m.cap, "h": m.h, "params": m.params,
            "tau": m.rules["M"]["pooled"].value, "selection": f["selection"],
            "em": {k: em.get(k) for k in ("a00", "a11", "pi1", "iterations", "multimodal", "start_spread", "at_bound",
                                          "nonmonotone", "any_nonmonotone", "grid_gap", "lambda", "stationary", "log_likelihood")},
            "uncertainty": f["uncertainty"], "dev_eps_error": f["dev_eps_error"],
            "e_prime_identifiable": f["e_prime_identifiable"], "neff": f["neff"]}
    selected = seedbs_states[SEEDBS]
    return {"roles": {s: len(st) for s, st in side["splits"].items()},
            "eps_true": {f"{s}/{n}": float(np.mean(v)) for (s, n), v in scen.items()},
            "min_neff": config["l2"]["min_neff"], "coverage": cov, "methods": methods, "f0": f0, "l_top": side["manifest"].get("l_top"),
            "seedbs": {"selected": {"B": selected["B"], "zeta": selected["zeta"]}, "table": seedbs_table,
                       "paper": {n: {"zeta": s["zeta"], "c": 1 - s["rules"]["M"]["pooled"].value}
                                 for n, s in seedbs_states.items() if n != SEEDBS}},
            "seconds": seconds}


def render_fit(r):
    out = ["# L2 拟合阶段报告（训练侧，dev 选择与冻结）", "",
           "本报告只用训练侧（密钥 K1）数据；测试侧在冻结之后才读取。", ""]
    out += ["## 角色与真值占比", "", table(["角色/场景", "ε 真值均值"],
            [[k, num(v)] for k, v in sorted(r["eps_true"].items())]), ""]
    if r.get("l_top"):
        lt = r["l_top"]
        out += [f"alt_top：参考 resample_0.1 的 {lt['articles']} 篇，平均真值切换点 {num(lt['mean_changepoints'], 2)}，"
                f"L_top = {lt['L_top']}，K_top = {lt['K_top']}。", ""]
    out += ["## e' 覆盖（truth=1 位置数）", "",
            table(["e' 区间", "文章数", "位置数", "达标"],
                  [[f"[{b['low']:.2f}, {b['high']:.2f})", b["articles"], b["positions"], "是" if b["ok"] else "否"]
                   for b in r["coverage"]["bins"]]), ""]
    out += ["## 方法冻结结果", "",
            table(["方法", "cap", "h", "a00", "a11", "π1(PE)", "τ", "dev 3 类均值 IoU", "回退"],
                  [[n, m["cap"] if m["cap"] is not None else "不截断", m["h"] if m["h"] is not None else "∞",
                    num(m["params"]["a00"], 4), num(m["params"]["a11"], 4),
                    num(m["params"]["pi1"], 3) if m["mode"] == "pe" else "-", num(m["tau"], 4),
                    num((m["selection"]["best"] or {}).get("iou")), "是" if m["selection"]["fallback"] else "否"]
                   for n, m in r["methods"].items()]), ""]
    for n, m in r["methods"].items():
        out += [f"### {n}：dev 候选", "",
                table(["cap", "h", "τ", "IoU(均值)"] + [f"IoU {c}" for c in DEV_CLASSES] + ["标记位置", "可用"],
                      [[c["cap"] if c["cap"] is not None else "不截断", c["h"] if c["h"] is not None else "∞",
                        num(c["tau"], 4), num(c["iou"])] + [num(c["iou_by_class"].get(k)) for k in DEV_CLASSES]
                       + [c["flagged_positions"], "是" if c["ok"] else "否"] for c in m["selection"]["table"]]), ""]
        em, u = m["em"], m["uncertainty"]
        bs = u.get("bootstrap")
        out += [f"EM：迭代 {em['iterations']}，多峰 {em['multimodal']}，触边界 {em['at_bound']}，网格差 {num(em['grid_gap'], 6)}；"
                f"a00 = a11 检验 p = {num(u['equal_retention']['p_value'], 4)}。"
                + (f" bootstrap a00 {ci_text({'mean': bs['a00']['mean'], 'ci95': bs['a00']['ci95']})}，"
                   f"a11 {ci_text({'mean': bs['a11']['mean'], 'ci95': bs['a11']['ci95']})}。" if bs else ""), ""]
        if m["neff"]:
            need = r.get("min_neff", 0)
            out += ["核加权 n_eff(e)（* 表示低于 %s，密度估计不稳，不做局部加宽）：" % need
                    + "，".join(f"{e:.2f}:{n_:.0f}" + ("*" if n_ < need else "")
                                for e, n_ in zip(m["neff"]["grid"], m["neff"]["neff"])), ""]
    s = r["seedbs"]
    out += ["## seedbs", "", f"dev 选择：B = {s['selected']['B']}，ζ = {s['selected']['zeta']}。", "",
            table(["B", "ζ", "c", "dev IoU", "标记位置", "null 平均假切换点", "可用"],
                  [[x["B"], x["zeta"], num(x["c"], 4), num(x["iou"]), x["flagged_positions"],
                    num(x["false_changepoints_per_null"], 2), "是" if x["ok"] else "否"] for x in s["table"]]), "",
            "论文参数（B = paper_B）各档 ζ 的校准：" + "，".join(f"{n}: c={num(v['c'], 4)}" for n, v in s["paper"].items()), ""]
    f0 = r["f0"]
    out += ["## f0 检查（null_cal）", "",
            f"位置 {f0['positions']}；sup|f̂0−1| = {num(f0['observed']['sup_dev'])}（iid 参考均值 "
            f"{num(f0['iid_reference']['sup_dev']['mean'])}，p95 {num(f0['iid_reference']['sup_dev']['p95'])}）；"
            f"χ² = {num(f0['observed']['chi2'], 1)}（自由度 {f0['dof']}，iid 参考 p95 "
            f"{num(f0['iid_reference']['chi2']['p95'], 1)}）。", "",
            f"用时 {num(r['seconds'], 0)} 秒。"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------- test report
def method_row_blocks(report, methods, metric="iou"):
    blocks = [b for b in ("paper", "alt", "r2", "r0") if b in report["blocks"]]
    rows = []
    for m in methods:
        cells = []
        for b in blocks:
            d = report["blocks"][b][m][metric]
            cells.append(ci_text(d) if d else "NA")
        rows.append([m] + cells)
    return blocks, rows


def render_test(r):
    hmm, methods = r["hmm_methods"], r["methods"]
    out = ["# L2 评价报告（一次 test，K2 密钥）", "",
           "本报告的判据与解读规则在生成评价侧数据之前写定（plan_l2_paper_setting_20261106 §5）。"
           "沿用论文的数据构造与评价指标，检测量与算法按其适配，不是对论文方法的复现。", "",
           "## 结论", "", r["interpretation"]["verdict"], "",
           f"C1（误报，重校准 p95 ≤ 2α 且中位 ≤ α）：" + "，".join(f"{m}={'通过' if ok else '未通过'}" for m, ok in r["criteria"]["C1"].items()),
           f"C2a：PD {'成立' if r['interpretation']['C2a']['pd'] else '不成立'}，PE {'成立' if r['interpretation']['C2a']['pe'] else '不成立'}。", ""]
    for metric, title in (("iou", "IoU（按 prompt 配对 bootstrap，95% 区间）"), ("rand_paper", "论文式 Rand index（无缺口真值切换点）")):
        blocks, rows = method_row_blocks(r, methods, metric)
        if metric == "rand_paper":
            blocks = [b for b in blocks if b in ("paper", "alt")]
            rows = [[row[0]] + [ci_text(r["blocks"][b][row[0]]["rand_paper"]) for b in blocks] for row in rows]
        out += [f"## 主终点：{title}", "", table(["方法"] + [BLOCK_TITLE[b] for b in blocks], rows), ""]
    tiers = sorted(r["tiers"].items(), key=lambda kv: (kv[1]["methods"][hmm[0]]["true_changepoints"] is None,
                                                       kv[1]["methods"][hmm[0]]["true_changepoints"] or 0))
    out += ["## 随切换点数 K 的曲线（论文 Setting 2–4 与交替段）", ""]
    for metric, title in (("iou", "IoU"), ("rand_paper", "Rand index"), ("cp", "检出切换点数 / 真值切换点数")):
        rows = []
        for s, t in tiers:
            if t["block"] not in ("paper", "alt"):
                continue
            k = t["methods"][hmm[0]]["true_changepoints"]
            cells = []
            for m in methods:
                c = t["methods"][m]
                if metric == "cp":
                    cells.append("NA" if not k else num(c["predicted_changepoints"] / k, 2))
                else:
                    cells.append(ci_text(c[metric], 3) if c[metric] else "NA")
            rows.append([s, num(k, 1)] + cells)
        out += [f"**{title}**", "", table(["档", "K"] + methods, rows), ""]
    d = r["deltas"]
    out += ["## 判据", "", "### C2a / C2b：fe − m400（ΔIoU）", "",
            table(["π", "块", "ΔIoU [95%]"], [[mode, b, ci_text(d["fe_vs_m400"][mode][b])]
                                            for mode in ("pd", "pe") for b in ("r2", "r0", "paper", "alt")]), "",
            "### C3：m400 − m100（ΔIoU）", "",
            table(["π", "块", "ΔIoU [95%]"], [[mode, b, ci_text(d["m400_vs_m100"][mode][b])]
                                            for mode in ("pd", "pe") for b in ("paper", "alt", "r2", "r0")]), "",
            "### C4：HMM 各臂 − seedbs（只报，不设判据）", ""]
    for ref in [SEEDBS] + [n for n in methods if n.startswith("seedbs_paper")]:
        rows = []
        for m in hmm:
            for b in ("paper", "alt", "r2", "r0"):
                x = d["vs_seedbs"][m][ref][b]
                rows.append([m, b, ci_text(x["iou"]), ci_text(x["rand_paper"]) if x["rand_paper"] else "-",
                             ci_text(x["rand_sb"])])
        out += [f"参照 `{ref}`", "", table(["方法", "块", "ΔIoU", "Δ论文式 Rand", "Δ(S−B) Rand"], rows), ""]
    out += ["## 误报（null_test）", "",
            table(["方法", "冻结 τ 下 any-FP", "重校准 p95", "重校准中位", "C1"],
                  [[m, f"{num(v['frozen_tau']['rate'])} ({num(v['frozen_tau']['wilson95'][0])}–{num(v['frozen_tau']['wilson95'][1])})",
                    num(v["recalibrated"]["p95"]), num(v["recalibrated"]["median"]), "通过" if v["C1"] else "未通过"]
                   for m, v in r["any_fp"].items()]), ""]
    out += ["## 次终点", "",
            table(["方法", "Setting 1 recall", "Setting 1 假切换点", "Setting 1 Rand", "ε̂ 误差", "后验饱和", "τ(均值)"],
                  [[m, num(r["set1"][m]["recall"]), num(r["set1"][m]["false_changepoints"], 2),
                    ci_text(r["set1"][m]["rand_paper"]), num((r["secondary"]["eps_error"] or {}).get(m)),
                    num((r["secondary"]["saturation"] or {}).get(m)), num(r["secondary"]["tau"].get(m), 4)]
                   for m in methods]), ""]
    if r.get("transition"):
        rows = []
        for n, t in r["transition"].items():
            bs = t.get("bootstrap") or {}
            er = t.get("equal_retention") or {}
            rows.append([n, t["cap"] if t["cap"] is not None else "不截断", t["h"] if t["h"] is not None else "∞",
                         num(t["params"]["a00"], 4), ci_text({"mean": (bs.get("a00") or {}).get("mean"), "ci95": (bs.get("a00") or {}).get("ci95", [None, None])}, 4),
                         num(t["params"]["a11"], 4), ci_text({"mean": (bs.get("a11") or {}).get("mean"), "ci95": (bs.get("a11") or {}).get("ci95", [None, None])}, 4),
                         num(er.get("p_value"), 4)])
        out += ["## A 的估计", "", table(["方法", "cap", "h", "a00", "a00 bootstrap", "a11", "a11 bootstrap", "a00=a11 检验 p"], rows), "",
                "`A` 在 mix_r2、mix_splice、mix_alt 上拟合；`test_paper` 与 `test_alt` 对它是域内，"
                "不据此声称对未见结构泛化。", ""]
    out += ["## R2 与 R0 的真值占比（应基本一致）", "",
            table(["场景", "ε 真值均值"], [[k, num(v)] for k, v in r["eps_true"].items()]), ""]
    if r.get("mechanism"):
        mech = r["mechanism"]
        out += ["## 机制对照（训练侧，不读 test）", "",
                f"M1：{'成立' if mech['M1']['holds'] else '不成立'}；M2：{'成立' if mech['M2']['holds'] else '不成立'}。", ""]
    out += ["## 曲线", "", "![按 K 的性能](curves.svg)", "", "## 说明", "",
            "- 非水印文本 U 来自同模型、同 prompt、不加水印的独立生成；论文没有写来源，这是本文的实现选择。",
            "- 每块 100 个 prompt，配对 ΔIoU 区间半宽约 0.02，对 0.01 以下的差异没有分辨力，不作方向性断言。",
            "- 等长交替段是规则的，resample 的存活游程长短不一，密度相当不等于难度相当。",
            "- 训练侧密钥 K1、评价侧密钥 K2：f₁ 与 A 在 K1 下拟合，在 K2 下测试，同时检验密钥迁移。", ""]
    if r.get("usage"):
        out += ["## 进度单位实际用量（下一轮权重初值）", "",
                table(["阶段", "单位数", "缓存", "秒"], [[k, v["units"], v["cached"], num(v["seconds"], 1)]
                                                    for k, v in r["usage"].items()]), ""]
    out += [f"用时 {num(r['seconds'], 0)} 秒。"]
    return "\n".join(out) + "\n"
