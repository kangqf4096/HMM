"""Markdown rendering of the L0 report; every number comes from report.json."""
from .schema import TEST_SCENARIOS

EMISSION_LABEL = {"histogram": "hist", "beta": "beta", "beta_mixture": "beta_mix",
                  "histogram_nocap": "hist_nocap", "beta_nocap": "beta_nocap"}
METHOD_ORDER = ("opt_hmm", "independent", "fixed_e", "fixed_rho", "window")

LIMITATIONS = [
    "calibration 只有 50 篇 → S2 分箱后每箱约 12 篇，置换功效受限。",
    "ε 是边际似然解，不是联合 HMM MLE；端点（0/1）会让 HMM 塌缩，端点计数见各表。",
    "ρ 是 7 点网格选择，不联合、不 EM；A 的平稳分布被强制为 π；f1 估完固定不更新。",
    "null 是未加水印的 OPT 续写，不是人工文本（威胁模型已收窄）。",
    "Y_t 理想独立，实际因 4-gram 复用存在弱相关；HMM 的乘积似然是工作近似。",
    "null 校准只控 any-FP，不提供混合文本的 FDR 或逐位置置信保证。",
    "resample 只在熵 ≥ min_entropy 的位置做，编辑后文本质量不做人工评估，只记录熵与替代词排名。",
]


def num(v, digits=3):
    return "NA" if v is None else f"{v:.{digits}f}"


def rate_cell(r):
    if r is None:
        return "NA"
    lo, hi = r["wilson95"]
    return f"{num(r['rate'])} ({num(lo)}–{num(hi)})"


def ordered_rows(by_emission):
    return [(e, m, by_emission[e][m]) for e in by_emission for m in METHOD_ORDER if m in by_emission[e]]


def scenario_table(scenario, by_emission):
    lines = [f"### {scenario}", "",
             "| 方法 / 发射 | IoU(M) [95% CI] | IoU(P) | P(M) | R(M) | F1(M) | ε误差 | any-FP(M) | 检出(M) | any-FP(P) | 检出(P) |",
             "|---|---|---|---:|---:|---:|---:|---|---|---|---|"]
    for e, m, c in ordered_rows(by_emission):
        lo, hi = c["iou"]["ci95"]
        lines.append(f"| {m} / {EMISSION_LABEL[e]} | {num(c['iou']['mean'])} [{num(lo)}, {num(hi)}] | {num(c['iou_P']['mean'])} | "
                     f"{num(c['precision'])} | {num(c['recall'])} | {num(c['f1'])} | {num(c['eps_error'])} | "
                     f"{rate_cell(c.get('any_fp_M'))} | {rate_cell(c['detect_M'])} | "
                     f"{rate_cell(c.get('any_fp_P'))} | {rate_cell(c['detect_P'])} |")
    return lines + [""]


def segment_table(scenario, by_emission):
    lines = [f"### {scenario}", "", "| 方法 / 发射 | 片段召回(≥50%覆盖) | 匹配片段数 | 边界误差均值 | 边界误差中位 | 有信号文章 |",
             "|---|---:|---:|---:|---:|---:|"]
    for e, m, c in ordered_rows(by_emission):
        b = c["boundary"]
        lines.append(f"| {m} / {EMISSION_LABEL[e]} | {num(c['segment_recall'])} | {b['matched']} | {num(b['mean'], 2)} | "
                     f"{num(b['median'], 2)} | {c['positive_articles']}/{c['articles']} |")
    return lines + [""]


def render(report):
    cfg, out = report["config"], []
    add = out.extend
    toy = report["manifest"]["toy"]
    add(["# L0 实验报告", ""])
    if toy:
        add(["> **玩具语言模型数据：仅验证管线，数字不是 OPT 的任何结论。**", ""])
    add(["探索性结果；判定口径见 `docs` 中的实验计划。所有区间为 token 下标，位置只在 N_eff 内。", "",
         f"- 角色篇数：{report['roles']}",
         f"- 发射：{cfg['emission']}，cap = {cfg.get('cap')}，α = {cfg['alpha']}，fraction_tau = {cfg.get('fraction_tau')}",
         f"- 随机重采样编辑：{cfg['edits']}", ""])
    if report["warnings"]:
        add(["## 警告", ""] + [f"- {w}" for w in report["warnings"]] + [""])
    eq = report["rule_equivalence"]
    add(["## 规则 P 与 M 的关系（偏离计划处）", "",
         "计划写“同一 γ 场、同一 τ”。若 P 的逐位置阈值取 M 用的 τ（null 上 max γ 的 1−α 分位），则 `p̂ > 0` 与 `max γ > τ` 完全等价，"
         "且 q 是一个大部分为 0 的统计量的分位数（= 0），P 退化为 M。因此 P 的逐位置阈值单独取 "
         f"`fraction_tau` = {cfg.get('fraction_tau')}（posterior 0.5，G2 冻结），q 仍在 null_cal 上校准。"
         "P 的标红位置由 `fraction_tau` 单独生成，与 M 的 tau 标红分开评分（见表中 IoU(P) 列）。", "",
         f"- 对照检查（test + null_test，共 {eq['rows']} 行 = 文章 × 方法 × 发射）：同 τ 版本 P_same 与 M 一致的比例 = {num(eq['P_same_equals_M'], 4)}；"
         f"正式 P 与 M 不一致的比例 = {num(eq['P_differs_from_M'], 4)}。", ""])
    add(["## 真值（truth = S \\ B）", "", "| 场景 | 文章 | ε_true 均值 | 最小 | 最大 | 无存活信号文章 |", "|---|---:|---:|---:|---:|---:|"])
    for s, t in report["truth"].items():
        add([f"| {s} | {t['articles']} | {num(t['eps_true_mean'])} | {num(t['eps_true_min'])} | {num(t['eps_true_max'])} | "
             f"{t['articles_without_surviving_signal']} |"])
    add(["", "resample 实际编辑量（请求 = int(rate·n)；可编辑 = 熵 ≥ min_entropy 的位置数之和）：", "",
         f"`{report['manifest']['edit_summary']}`", ""])
    add(["## 主表（test，按场景）", "", "IoU = 位置集合的 Jaccard，仅对有存活信号的文章平均并按文章 bootstrap；"
         "any-FP 来自 null_test（括号为 Wilson 95%）。P(M)/R(M)/F1(M) 是规则 M 的标红位置（gamma > tau）的精确率/召回率，"
         "为 0 表示没有标中任何真值位置；IoU(P) 是规则 P 自己标红（gamma > fraction_tau）的 IoU。"
         "null_test 在四个场景间共用同一批文章，any-FP 不是四次独立测量。", ""])
    for s in TEST_SCENARIOS:
        if s in report["summary"]:
            add(scenario_table(s, report["summary"][s]))
    add(["## 片段级", ""])
    for s in TEST_SCENARIOS:
        if s in report["summary"]:
            add(segment_table(s, report["summary"][s]))
    add(["## 连续性增益判定", "", "口径：opt_hmm 的 IoU 对四个对照（independent / fixed_e / fixed_rho / window，同发射轴）"
         "的配对 bootstrap 差值 95% CI 下界均 ≥ 0，且 any-FP ≤ α。M 与 P 各自使用自己的标红位置"
         "（M 为 gamma > tau，P 为 gamma > fraction_tau），因此两列的 IoU 与 any-FP 都是对应规则自己的。"
         "达不到即“本次数据未显示增益”，不调参。", "",
         "| 场景 | 发射 | " + " | ".join(f"Δ vs {c}" for c in ("independent", "fixed_e", "fixed_rho", "window"))
         + " | IoU 不低于所有对照 | any-FP(M)≤α | any-FP(P)≤α | IoU(P) 不低于所有对照 | 增益(M) | 增益(P) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"])
    for s, by_emission in report["gain"].items():
        for e, g in by_emission.items():
            diffs = " | ".join(f"{num(d['mean'])} [{num(d['ci95'][0])}, {num(d['ci95'][1])}]"
                               for d in (g["paired_iou_diff_vs"].get(c) or {"mean": None, "ci95": [None, None]}
                                         for c in ("independent", "fixed_e", "fixed_rho", "window")))
            add([f"| {s} | {EMISSION_LABEL[e]} | {diffs} | {g['iou_not_below_any_control']} | {g.get('any_fp_M_within_alpha')} | "
                 f"{g.get('any_fp_P_within_alpha')} | {g.get('iou_not_below_any_control_P')} | {g.get('gain_M')} | {g.get('gain_P')} |"])
    add(["", "## 参数诊断（opt_hmm）", "", "| 场景 | 发射 | ε̂ 均值 | ε̂=0 | ε̂=1 | ρ̂ 计数 |", "|---|---|---:|---:|---:|---|"])
    for s, by_emission in report["summary"].items():
        for e, by_method in by_emission.items():
            c = by_method["opt_hmm"]
            add([f"| {s} | {EMISSION_LABEL[e]} | {num(c['eps_hat'])} | {c['eps_endpoint']['at_0']} | {c['eps_endpoint']['at_1']} | "
                 f"{c['rho_counts']} |"])
    d = report["diagnostics"]
    add(["", "## 诊断（不进主表）", "", "### 发射形状与 cap（calibration 拟合）", "",
         "| 族 | f1(0.001) | f1(0.1) | f1(0.5) | f1(0.9) | f1(0.99) | f1(0.9999) | cap 下积分质量 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for family, s in d["emission_shapes"].items():
        f = s["f1_at"]
        add([f"| {family} | {num(f['0.001'])} | {num(f['0.1'])} | {num(f['0.5'])} | {num(f['0.9'])} | {num(f['0.99'])} | "
             f"{num(f['0.9999'])} | {num(s['mass_under_cap'] if s['mass_under_cap'] is not None else s['mass_uncapped'], 4)} |"])
    add(["", "### split-half 稳定性（4.2）", ""])
    sh = d["split_half"]
    if "skipped" in sh:
        add([f"跳过：{sh['skipped']}。"])
    else:
        add([f"calibration {sh['calibration_articles']} 篇，每半 {sh['half_size']} 篇，重复 {sh['repeats']} 次。", "",
             "| 发射 | TV 均值 | TV p95 | KS 均值 | KS p95 |", "|---|---:|---:|---:|---:|"])
        for e in cfg["emission"]:
            add([f"| {EMISSION_LABEL[e]} | {num(sh[e]['tv']['mean'])} | {num(sh[e]['tv']['p95'])} | "
                 f"{num(sh[e]['ks']['mean'])} | {num(sh[e]['ks']['p95'])} |"])
    add(["", "### 有界性 / 爆炸（4.3）", "", "| 发射 | 场景 | max log f1 | p99.9 | p99.99 | 位置数 |", "|---|---|---:|---:|---:|---:|"])
    for e, b in d["boundedness"].items():
        for s, t in b["log_f1_by_scenario"].items():
            add([f"| {EMISSION_LABEL[e]} | {s} | {num(t['max'])} | {num(t['p99.9'])} | {num(t['p99.99'])} | {t['n']} |"])
    add(["", "calibration 上 max f1 与 y 高分位处密度：", "", "| 发射 | max f1 | y p99.9 | f1(y p99.9) | y p99.99 | f1(y p99.99) |", "|---|---:|---:|---:|---:|---:|"])
    for e, b in d["boundedness"].items():
        c = b["calibration"]
        add([f"| {EMISSION_LABEL[e]} | {num(c['max_f1'])} | {num(c['y_p99.9'], 5)} | {num(c['f1_at_y_p99.9'])} | "
             f"{num(c['y_p99.99'], 6)} | {num(c['f1_at_y_p99.99'])} |"])
    add(["", "### 后验饱和率（γ > 0.99 的位置占比，test 平均）", "", "| 场景 | 发射 | " + " | ".join(METHOD_ORDER) + " |",
         "|---|---|" + "---:|" * len(METHOD_ORDER)])
    for s, by_emission in report["summary"].items():
        for e, by_method in by_emission.items():
            add([f"| {s} | {EMISSION_LABEL[e]} | " + " | ".join(num(by_method[m]["posterior_saturation"], 4) for m in METHOD_ORDER) + " |"])
    add(["", "### 文档级 AUC（附录，辅助；有存活信号的 test 文章 vs null_test）", "",
         "| 场景 | 发射 / 方法 | AUC(M: max γ) | AUC(P: 标红占比) |", "|---|---|---:|---:|"])
    for s, by_emission in report["document_auc"].items():
        for e, by_method in by_emission.items():
            for m in METHOD_ORDER:
                if m in by_method:
                    add([f"| {s} | {m} / {EMISSION_LABEL[e]} | {num(by_method[m]['M'])} | {num(by_method[m]['P'])} |"])
    add(["", "## 冻结参数与耗时", "", f"- τ（M，pooled）与 q（P，pooled）：`{report['thresholds_pooled']}`",
         f"- 耗时（秒）：`{report['timings']}`", f"- 冒烟出口检查：`{report['smoke_check']}`", "",
         "## 已知限制", ""] + [f"{i}. {t}" for i, t in enumerate(LIMITATIONS, 1)] + [""])
    return "\n".join(out)
