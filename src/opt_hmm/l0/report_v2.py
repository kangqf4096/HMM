"""L1 report (plan_general_transition 6, 9): build_report -> JSON-serialisable dict, render -> markdown.

Everything in the markdown comes from the dict. A diagnostic whose inputs a stage does not have is skipped.
Only stage s4 has formal criteria (one-shot test); every other stage is exploratory, on dev2 / dev with null_cal.
"""
import math
import re
from collections import defaultdict

import numpy as np

from ..parametric import FAMILIES as PARAM_FAMILIES
from .decide import decide
from .emission import Capped, V2_FAMILIES, describe, fit_base
from .evaluate import rate
from .evaluate_v2 import (SEEDBS, STAGE_EVAL, STAGE_ROLES, Endpoint, main_methods, recalibrated_false_positive,
                          summarize)
from .metrics import bootstrap_indices, boundary_summary
from .schema import ALL_SCENARIOS, LOCALIZATION_SCENARIOS, SEG_SCENARIOS

LABEL = {"hist25": "hist25", "hist200": "hist200", "ecdf400": "ecdf400"}
NONINFERIOR = -0.01
REPORT_NAMES = {"opt_hmm": "pd_rho_hist25", "opt_hmm_li": "pd_rho_hist25_epsopt"}


# ---------------------------------------------------------------------- helpers
def num(v, digits=3):
    return "NA" if v is None else f"{v:.{digits}f}"


def ci_text(d):
    if not d:
        return "NA"
    lo, hi = d["ci95"]
    return f"{num(d['mean'])} [{num(lo)}, {num(hi)}]"


def rate_text(r):
    if not r or r.get("rate") is None:
        return "NA"
    lo, hi = r["wilson95"]
    return f"{num(r['rate'])} ({num(lo)}–{num(hi)})"


def jsonable(x):
    """numpy -> python, non-finite floats -> None, tuples -> lists."""
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


def mean_of(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def family_of(name):
    if name in ("opt_hmm", "independent", "fixed_e", "window", "opt_hmm_li", SEEDBS):
        return None
    return name.removesuffix("_nocap").split("_", 1)[1]


def safe_delta(ep, a, b):
    if ep is None or a not in ep.per or b not in ep.per:
        return None
    return ep.delta(a, b)


def mean_ci(ep, method):
    """95% CI of the primary endpoint for one method, same stratified indices as Endpoint.delta."""
    boot = np.mean([ep.per[method][s][ep.indices[s]].mean(axis=1) for s in ep.scenarios], axis=0)
    return {"mean": ep.mean(method), "ci95": np.quantile(boot, [.025, .975]).tolist()}


def half_life(lam):
    return float(math.log(2) / -math.log(lam)) if 0 < lam < 1 else None


def display_names(text):
    """Use descriptive report labels while keeping saved method identities stable."""
    return re.sub(r"(?<![A-Za-z0-9_])(?:opt_hmm_li|opt_hmm)(?![A-Za-z0-9_])",
                  lambda match: REPORT_NAMES[match.group()], text)


def method_name_guide():
    return [
        "**方法名称说明。** `pd_rho_hist25`（原记录标识 `opt_hmm`）表示“首位先验直接取逐篇占比、转移参数 ρ 逐篇选择、25 箱发射”；"
        "`pd_rho_hist25_epsopt`（原记录标识 `opt_hmm_li`）只将占比估计器换成 ε̂_opt。前者使用原边际复合似然占比估计。"
        "两者的转移矩阵均为 `A_d=ρ_d I+(1−ρ_d)1π_dᵀ`，π_d 也是该矩阵的平稳分布。"
        "主表的 `pd_hist200`、`pd_beta` 则各有一个跨文章共享、用 mixed_cal 池化 EM 估计的双参数 A；"
        "“rho”标记用于区分逐篇转移方案与全局转移方案。原始 JSON/逐篇指标中的方法标识用于对应已有实验记录。",
        "",
    ]


def diagnostic_explanation(report, topic):
    """Explain the recorded statistics without changing scores or evaluation criteria."""
    diag = report["diagnostics"]
    paragraphs = []
    if topic == "emission":
        e = diag["emission"]
        paragraphs = [
            "**这些列衡量什么。** 本表检查 calibration 与 calibration2 合并后的纯水印枢轴分布。"
            "∫f 是有限网格上的数值积分，用于检查归一化；尾部质量是密度在 [0.99,1] 的积分，"
            "应与同一批校准枢轴中 y≥0.99 的经验占比对照。两列比值分别是该尾部质量除以 hist200、hist25 的尾部质量；"
            "例如 2.833 表示质量约为参照的 2.833 倍。这里的 f₁ 是枢轴 Y 的密度，密度值可以大于 1。",
            "**原始密度与实际使用的分数。** 本表使用未加 cap、未做尾部外推的基础密度；"
            "max f₁ 是诊断网格上的最大值，不是数学上的全局上界，也不是后验概率。"
            "端点附近的尖峰能在很窄区间内给出很大的密度而仍有有限积分。"
            "参数族的极大网格值不等于推理时直接用了同样大的密度：实际入选 cap 见配置对应表与 cap 选择表。"
            "∫f 略偏离 1 还受到有限积分范围、端点截取和求积精度影响，应记录为数值积分偏差，"
            "仅凭这一项不能断定解析分布没有归一化。",
            "**两折如何计算。** calibration、calibration2 分别作为两折，一折拟合基础密度，另一折计算各枢轴的自然对数密度，"
            "然后交换两折并取平均。“2 折平均 log 密度”越大，表示对另一折纯水印枢轴的平均密度预测更好；"
            "它不是检验 p 值，也不直接衡量混合文本的定位效果。折间 L1 是两折分别拟出的密度差绝对值的积分，"
            "越小表示对这次分折更稳定；稳定本身不保证拟合准确。两项均只作诊断，选族仍使用 dev 定位 IoU。",
        ]
        f = e["families"]
        if "hist25" in f and "hist200" in f:
            folds = (e.get("folds") or {}).get("families", {})
            heldout = {name: row["heldout_mean_log_density"] for name, row in folds.items()
                       if row.get("heldout_mean_log_density") is not None}
            top = max(heldout, key=heldout.get) if heldout else None
            winner = (report.get("param_star") or {}).get("family")
            paragraphs.append(
                f"**本次结果。** 经验尾部占比为 {e['calibration_tail99']:.4f}，hist25 只给出 {f['hist25']['mass_tail99']:.4f}，"
                f"hist200 给出 {f['hist200']['mass_tail99']:.4f}。25 箱把末端较宽区间内的密度摊平，"
                "因此难以保留最后 1% 区间的尖峰；200 箱在这个尾部区间更接近经验质量。"
                "其他族可按尾部质量与经验占比的差距检查高估或低估。")
            if top and winner:
                paragraphs.append(
                    f"两折平均 log 密度最高的是 {top}，dev 定位选出的是 {winner}；"
                    "两者目标不同，纯水印密度预测与混合文本定位需要分别评价。"
                    "折间 L1 较大的族则提示拟合形状更受校准样本变化影响。")
    elif topic == "f0":
        n = diag["f0"]["pivots"]
        fine = diag["f0"]["hist200"]
        paragraphs = [
            f"**计算口径。** 将同一批 null_cal 的 {n:,} 个枢轴分别等宽分成 25 箱、200 箱，"
            "每箱密度为“该箱个数/总个数×箱数”，sup|f̂₀−1| 取所有箱中与理论均匀密度 1 的最大偏差。"
            "这里的 hist25/hist200 只是零分布诊断的分箱数，不表示两种方法采用了不同的 f₀；HMM 仍统一使用 f₀≡1。",
            "**参考值如何读。** “iid U(0,1) 参考”是 20 组同样大小的独立均匀模拟各自最大偏差的平均值，"
            "不是 95% 上限或显著性阈值。箱数增加后每箱样本减少，同时要从更多箱里取最大值，"
            f"即使真实分布完全均匀，最大偏差也会增大。因此观测值 {fine['sup_abs_dev']:.3f} "
            f"与平均参考 {fine['iid_uniform_reference']:.3f} 的差距不能单独作为拒绝均匀分布的依据。"
            "该诊断检查边缘分布，不能验证跨位置独立性；实际文本中的重复上下文也可能影响独立抽样参考的适用性。",
        ]
    elif topic == "eps_estimators":
        e = diag["eps_estimators"]
        paragraphs = [
            "**比较对象。** 两个逐篇 ρ 方案使用同一份 hist25 和同一 cap，区别是占比估计器："
            "原方案最大化边际复合似然；epsopt 版本使用纯水印校准样本参与的 ε̂_opt 估计。"
            "差值定义为 ε̂_opt−原估计；负值表示 ε̂_opt 略小。绝对差衡量两个估计器彼此的差距，"
            "“对真值误差”则是各自与真实存活占比的平均绝对误差，两者含义不同。",
            f"**本次结果。** 在 {e['articles']} 篇评价文章上，差均值为 {e['mean_diff']:.6f}，"
            f"绝对差均值为 {e['mean_abs_diff']:.6f}；真值误差分别为 {e['error_old']:.5f}、{e['error_opt']:.5f}。"
            "这说明两个估计器在本次数据上非常接近，没有观察到明显的占比估计精度收益。"
            "本节纳入评价角色的全部文章，包括 full_wm；主表 ε 误差使用六个定位场景，所以汇总值可以不同。",
            "**边界计数。** 原估计允许精确取 0 或 1；ε̂_opt 投影到 [0.001,0.999]，因此报告的是是否贴近这两个内缩端点。"
            f"本次原估计有 {e['old_at_endpoint']} 篇取到端点，ε̂_opt 有 {e['opt_at_boundary']} 篇贴近内缩端点；"
            "仅凭计数不能确认逐篇身份是否相同，纯水印 full_wm 文章贴近 1 也不一定是异常。",
        ]
    elif topic == "prior_reach":
        reach = diag["prior_reach"]
        distribution = "；".join(
            f"{family} 总位置差异占比 {row['differing_fraction']:.2%}，其中 {row['by_position_fifth'][0]:.1%} 位于最前五分之一"
            for family, row in reach.items())
        paragraphs = [
            "**表中比例的分母。** “不同位置占比”是在全部评价文章、全部有效位置上，PD 与同族 P2 的二值标签不一致的比例。"
            "“五等分分布”只在这些不一致位置内，将位置按文章从前到后的五段分组，五个数之和约为 1。"
            "不能把“全部不一致位置中有多少在开头”理解成“开头有多少 token 被改判”。",
            f"**可以解释到什么程度。** {distribution}。"
            "应同时看后续各段的差异，不能仅凭第一段占比概括为“只影响开头”。"
            "而且这里 PD 与 P2 各自拟合了 A、选择 cap 并校准 τ，比较同时受到这些变化影响，"
            "并非只替换首位先验的单因素实验。首位先验在固定 A 下的衰减可参照结构表中的 λ 与半衰期；"
            "P2 已知的优化局限也使本表只适合作附录描述。",
        ]
    elif topic == "p1_alignment":
        entries = diag["p1_alignment"]
        amplification = "、".join(f"{name} 约 {1 / row['alpha_bar']:.1f} 倍" for name, row in entries.items())
        gaps = "；".join(f"{name} 的超限比例 {row['pi_clipped']:.1%}、平均绝对差 {row['marginal_gap_mean']:.3f}"
                         for name, row in entries.items())
        paragraphs = [
            "**对齐目标与 ᾱ。** P1 试图通过反解首位概率，使全篇的先验平均水印概率等于逐篇 ε̂_opt。"
            "ᾱ 是首位状态对全篇平均占比的影响系数；反解中的误差会被放大 1/ᾱ。"
            f"按本表参数计算，{amplification}。链很快忘记起点时，仅调整首位概率无法适配跨度很大的文章占比。",
            "**原 clip 字段的准确含义。** 程序统计的实际是“先验平均占比与 ε̂_opt 的绝对差超过 0.02”的文章比例，"
            "并非直接统计首位概率恰好截到 0/1 的比例。“不可识别比例”只表示 ᾱ<0.05、触发退回平稳先验的比例；"
            "为 0 仅表示没有触发这条保护，并不保证对齐可行。末列是对齐绝对偏差的均值与第 95 百分位，数值越小越好。",
            f"**本次结果。** {gaps}。超限比例越高，说明越多文章未实现 P1 设想的先验平均占比对齐。"
            "这也不等于预测后验与真实水印占比一定存在同样偏差，因为对齐对象是先验。"
            "P1 的拟合还须结合结构表中的网格差与非单调诊断检查，本节作为附录描述，不参与主表结论。",
        ]
    elif topic == "seedbs_recovery":
        sb = diag["seedbs"]
        recovery = "；".join(f"{name} 真值 {row['true']:.2f} 个、检出 {row['predicted']:.2f} 个"
                             for name, row in sb["recovery"].items())
        performance = ""
        if sb.get("false_changepoints_per_null") is not None:
            performance = f"本次 null_cal 平均每篇约 {sb['false_changepoints_per_null']:.3f} 个假切换点。"
        paragraphs = [
            "**参数与两层判定。** B 控制局部窗口尺度，同时作为移动块自助的块长；ζ 是筛选候选切换点区间的阈值，"
            "越大通常越容易保留切换点。c 则用于分段后的水印标签：段分数为 1−段检验值，分数>τ 等价于段检验值<c，且 c=1−τ。"
            "ζ 控制是否切段，c 控制切出来的段是否标成水印，二者与文档级 α=0.05 不应混同。"
            "分段本身由同一份数据选择，因此段检验值不能直接作为未经选择的有效 p 值；文档误报依靠 null_cal 的整体阈值校准。",
            "**切换点表如何读。** 各列是每篇文章的切换点数量及距离的场景平均。"
            "距离实际按“每个真值切换点到最近检出切换点”计算，单位是有效序列位置；只在两类切换点都存在时有定义。"
            "它不是“每个检出点到最近真值”的距离，也不是带有漏检惩罚的双向匹配误差。"
            f"本次场景平均为：{recovery}。检出数量明显多于真值时提示额外切分较多。"
            "检出点越多也可能降低最近距离，因此应结合 Rand index、最终标签 IoU 与数量一起看。",
            f"**假切换点与误报不同。** {performance}假切换点统计的是纯 null 文本被切出的额外边界，"
            "不等同于水印假阳性标签数。切段后还要经过 c 的标签判定，一篇文章可以有多个切点但没有任何水印标签。"
            "报告中的每篇秒数来自选中候选在调参阶段的“预计算墙钟时间/文章数”，复用时沿用原记录，受当时并行配置影响，"
            "不是本次 S4 单篇端到端耗时。",
        ]
    elif topic == "seedbs_selection":
        s = diag["seedbs"]
        paragraphs = [
            "**16 组选择表。** 每组 B、ζ 各自在 null_cal 上校准 τ/c，然后在 dev 的三个定位场景上计算平均 IoU；"
            "“可检出位置”是该组阈值下 dev 三个场景全部文章被标成水印的位置总数，包含正确和错误标签。"
            "“合格=True”只表示 τ<1 且可检出位置数>0，并不表示切换点误报少或通过了正式 S4 判据。",
            f"**入选理由。** B={s['B']}、ζ={s['zeta']} 是合格候选中 dev IoU 最高的一组。"
            "更小的 ζ 通常减少额外切点，但也可能漏掉真实变化；而每组都重新校准 c，所以切点数与水印标签数不会简单同向变化。"
            "复用参数时本表沿用原阶段的选择，dev IoU 只用于选择，不能与不同评价文章、不同场景汇总直接比较。"
            "SeedBS 不用纯水印 f₁ 或 mixed_cal 的 A 训练；主表四格另用了 100 篇纯水印及 50 篇混合训练文章，训练信息量并不相同。",
        ]
    elif topic == "truth":
        cfg = report["config"]
        tokens, context = cfg["tokens"], cfg["context_width"]
        n = tokens - context
        shares = "；".join(f"{name} 为 {round(row['eps_true_mean'] * n)}/{n}≈{row['eps_true_mean']:.3f}"
                           for name, row in report["truth"].items()
                           if name in ("switch", "seg2", "seg3", "seg4") and row["eps_true_min"] == row["eps_true_max"])
        resamples = "、".join(f"{name} 为 {row['eps_true_mean']:.3f}"
                              for name, row in report["truth"].items() if name.startswith("resample_"))
        paragraphs = [
            "**ε_true 是什么。** 它是每篇有效枢轴位置中 truth=1 的比例，即仍保留当前密钥水印耦合的位置占比，"
            f"不是原文来自水印生成器的比例，也不是 HMM 估计值。长度 {tokens}、上下文宽度 {context} 时，"
            f"前 {context} 个位置不参与评价，分母为 {n}。"
            "表中均值、最小和最大描述同一场景不同文章的真实存活占比。",
            f"**为什么替换 10%/30% 后不剩 90%/70%。** 本项目将被编辑 token 以及其后最多 {context} 个依赖该上下文的 token 都视为耦合被破坏，"
            f"重叠破坏窗口只计一次。本次平均存活比例：{resamples}，"
            "会明显低于简单按替换比例相减得到的值；编辑位置和窗口重叠不同，也会造成文章间范围变化。",
            "**分段场景的常数。** switch、seg2、seg3、seg4 使用固定的水印区间，full_wm 全部有效位置都为水印，"
            f"所以同一场景的占比最小值与最大值相同。区间的 token 取整及前 {context} 个位置的排除后，{shares}。"
            "文章数相同不表示真实水印位置数相同。",
        ]
        if not resamples:
            paragraphs.pop(1)
    return [line for paragraph in paragraphs for line in (paragraph, "")]


# ---------------------------------------------------------------------- per-method posterior diagnostics
def posterior_details(run, names, docs):
    """Detections of the HMM methods on the evaluation documents: mean gamma vs eps_hat and kept for label comparisons."""
    detections, table = {}, {}
    for name in names:
        state = run.states[name]
        if state["kind"] == "seedbs" or state.get("mode") == "window":
            continue
        dets = run.infer(state, docs)
        detections[name] = dets
        mean_gamma = np.array([d.scores.mean() for d in dets])
        eps = np.array([d.epsilon for d in dets])
        table[name] = {"mean_gamma": float(mean_gamma.mean()), "gap_signed": float((mean_gamma - eps).mean()),
                       "gap_abs": float(np.abs(mean_gamma - eps).mean()),
                       "saturation": float(np.mean([(d.scores > .99).mean() for d in dets]))}
    return detections, table


def prior_reach(run, detections, docs):
    """Where PD and the same-family P2 disagree on the labels, by position fifth (plan 6.4)."""
    out = {}
    for family in ("hist25", "ecdf400"):
        a, b = f"pd_{family}", f"p2_{family}"
        if a not in detections or b not in detections:
            continue
        counts, total, n_pos = np.zeros(5), 0, 0
        for doc, da, db in zip(docs, detections[a], detections[b]):
            la, _ = decide(da.scores, doc["n_eff"], run.states[a]["rules"])
            lb, _ = decide(db.scores, doc["n_eff"], run.states[b]["rules"])
            differ = la != lb
            idx = np.minimum(np.arange(len(differ)) * 5 // len(differ), 4)
            counts += np.bincount(idx[differ], minlength=5)
            total += int(differ.sum())
            n_pos += len(differ)
        out[family] = {"differing_fraction": total / n_pos if n_pos else None,
                       "by_position_fifth": (counts / max(total, 1)).tolist(), "differing_positions": total}
    return out


# ---------------------------------------------------------------------- emission / null diagnostics
S_GRID = np.linspace(-27., 27., 20001)
Y_GRID = 1. / (1. + np.exp(-S_GRID))
W_GRID = Y_GRID * (1. - Y_GRID) * (S_GRID[1] - S_GRID[0])


def _integral(values, mask=None):
    w = W_GRID if mask is None else np.where(mask, W_GRID, 0.)
    return float(np.sum(np.asarray(values, float) * w))


def emission_diagnostics(run):
    """Tail mass in [0.99,1], shape summary, 2-fold held-out log density and fold-to-fold L1 distance (plan 6.4)."""
    if not run.bases:
        return None
    tail = S_GRID >= math.log(.99 / .01)
    out = {"families": {}, "folds": None}
    for family, base in run.bases.items():
        f = np.asarray(base(Y_GRID), float)
        out["families"][family] = {"mass_total": _integral(f), "mass_tail99": _integral(f, tail),
                                   "max_f1": float(f.max()), **describe(Capped(base, None), None)}
    wm = np.concatenate([a["y"] for a in run.docs["calibration"] + run.docs["calibration2"]])
    out["calibration_tail99"] = float((wm >= .99).mean())
    for family in out["families"]:
        ref = out["families"]["hist200"]["mass_tail99"]
        out["families"][family]["tail99_vs_hist200"] = out["families"][family]["mass_tail99"] / ref if ref else None
        ref = out["families"]["hist25"]["mass_tail99"]
        out["families"][family]["tail99_vs_hist25"] = out["families"][family]["mass_tail99"] / ref if ref else None
    first, second = run.docs.get("calibration"), run.docs.get("calibration2")
    if first and second:
        ya, yb = (np.concatenate([a["y"] for a in part]) for part in (first, second))
        folds = {}
        for family in V2_FAMILIES:
            try:
                fa, fb = fit_base(family, ya, run.config), fit_base(family, yb, run.config)
                loglik = (float(np.mean(np.log(np.maximum(fa(yb), 1e-300)))) + float(np.mean(np.log(np.maximum(fb(ya), 1e-300))))) / 2
                l1 = _integral(np.abs(np.asarray(fa(Y_GRID)) - np.asarray(fb(Y_GRID))))
                folds[family] = {"heldout_mean_log_density": loglik, "l1_between_folds": l1}
            except (ValueError, FloatingPointError) as error:
                folds[family] = {"error": str(error)}
        out["folds"] = {"fold_articles": [len(first), len(second)], "families": folds}
    return out


def f0_diagnostic(run):
    docs = run.docs.get("null_cal")
    if not docs:
        return None
    y = np.concatenate([a["y"] for a in docs])
    rng = np.random.default_rng(0)
    out = {"pivots": int(len(y))}
    for bins in (25, 200):
        counts, _ = np.histogram(y, bins=bins, range=(0, 1))
        density = counts / len(y) * bins
        noise = [np.abs(np.histogram(rng.random(len(y)), bins=bins, range=(0, 1))[0] / len(y) * bins - 1).max()
                 for _ in range(20)]
        out[f"hist{bins}"] = {"sup_abs_dev": float(np.abs(density - 1).max()), "iid_uniform_reference": float(np.mean(noise))}
    return out


# ---------------------------------------------------------------------- structure / EM health
def structure_row(name, state):
    em = state.get("em")
    if not em:
        return None
    boot, eq = state.get("bootstrap"), state.get("equal_retention")
    k0 = state.get("em_k0")
    row = {"a00": em["a00"], "a11": em["a11"], "pi1": em.get("pi1"), "lambda": em["lambda"],
           "half_life": half_life(em["lambda"]), "stationary": em["stationary"], "log_likelihood": em["log_likelihood"],
           "iterations": em["iterations"], "at_bound": em["at_bound"], "multimodal": em["multimodal"],
           "start_spread": em["start_spread"], "nonmonotone": em["nonmonotone"], "any_nonmonotone": em["any_nonmonotone"],
           "grid_gap": em.get("grid_gap"), "grid_best": em.get("grid", {}).get("best"),
           "plateau": em.get("grid", {}).get("plateau"),
           "cap": state.get("cap"), "tail": state.get("tail"),
           "shift_from_k0": None if not k0 else {"a00": em["a00"] - k0["a00"], "a11": em["a11"] - k0["a11"]},
           "bootstrap": boot, "equal_retention": eq,
           "equal_retention_rejects": None if not eq else bool(eq["p_value"] < .05),
           "tau_pooled": state["rules"]["M"]["pooled"].value, "fit_seconds": state.get("fit_seconds")}
    return row


def p1_alignment(rows, names):
    out = {}
    for name in names:
        sel = [r for r in rows if r["method"] == name and r.get("alpha_bar") is not None]
        if not sel:
            continue
        out[name] = {"alpha_bar": sel[0]["alpha_bar"], "pi_clipped": mean_of([float(r["pi_clipped"]) for r in sel]),
                     "pi_unidentifiable": mean_of([float(r["pi_unidentifiable"]) for r in sel]),
                     "marginal_gap_mean": mean_of([r["marginal_gap"] for r in sel]),
                     "marginal_gap_p95": float(np.quantile([r["marginal_gap"] for r in sel], .95))}
    return out


def eps_estimators(rows):
    """Old estimator (opt_hmm) vs eps_opt (opt_hmm_li) on the same articles."""
    old = {r["article_id"]: r for r in rows if r["method"] == "opt_hmm"}
    new = {r["article_id"]: r for r in rows if r["method"] == "opt_hmm_li"}
    ids = sorted(set(old) & set(new))
    if not ids:
        return None
    diff = np.array([new[i]["eps_hat"] - old[i]["eps_hat"] for i in ids])
    return {"articles": len(ids), "mean_diff": float(diff.mean()), "mean_abs_diff": float(np.abs(diff).mean()),
            "error_old": mean_of([old[i]["eps_error"] for i in ids]), "error_opt": mean_of([new[i]["eps_error"] for i in ids]),
            "old_at_endpoint": int(sum(old[i]["eps_hat"] in (0., 1.) for i in ids)),
            "opt_at_boundary": int(sum(new[i]["eps_hat"] <= 1.001e-3 or new[i]["eps_hat"] >= 1 - 1.001e-3 for i in ids))}


def seedbs_section(run, summary, rows):
    state = run.states.get(SEEDBS)
    if not state:
        return None
    chosen = next((r for r in state["table"] if r["B"] == state["B"] and r["zeta"] == state["zeta"]), None)
    recovery = {s: summary[s][SEEDBS]["changepoints"] for s in SEG_SCENARIOS if s in summary and SEEDBS in summary[s]}
    return {"B": state["B"], "zeta": state["zeta"], "c": 1 - state["rules"]["M"]["pooled"].value,
            "reused_from": state.get("reused_from"), "table": state["table"], "chosen": chosen, "recovery": recovery,
            "false_changepoints_per_null": None if not chosen else chosen["false_changepoints_per_null"],
            "seconds_per_article": None if not chosen else chosen["seconds_per_article"]}


def nocap_section(run, names, summary):
    out = []
    for name in names:
        if not name.endswith("_nocap"):
            continue
        state, ref_name = run.states[name], name.removesuffix("_nocap")
        ref = run.states.get(ref_name)
        scores = [d.scores for d in state.get("null_cal_detections", [])]
        flat = np.concatenate(scores) if scores else None
        train_g = state.get("train", {}).get("g")
        row = {"method": name, "tau": state["rules"]["M"]["pooled"].value,
               "null_gamma_gt_999": None if flat is None else float((flat > .999).mean()),
               "null_max_gamma_is_one": None if flat is None else bool(flat.max() >= 1 - 1e-12),
               "max_g_train": None if train_g is None else float(train_g.max()),
               "em": structure_row(name, state), "reference": ref_name if ref else None,
               "reference_em": structure_row(ref_name, ref) if ref else None,
               "tau_reference": ref["rules"]["M"]["pooled"].value if ref else None}
        for s in LOCALIZATION_SCENARIOS:
            if s in summary and name in summary[s]:
                row.setdefault("iou", {})[s] = summary[s][name]["iou"]["mean"]
                if ref_name in summary[s]:
                    row.setdefault("iou_reference", {})[s] = summary[s][ref_name]["iou"]["mean"]
                row.setdefault("boundary_mean", {})[s] = summary[s][name]["boundary"]["mean"]
        out.append(row)
    return out


# ---------------------------------------------------------------------- the report
def build_report(run, rows, null_rows, names, manifest, time_total):
    stage, config = run.stage, run.config
    formal = stage == "s4"
    names = list(names)
    param_star = run.param_star["family"] if run.param_star else None
    main = [m for m in main_methods(param_star) if m in names] if param_star or "pd_hist200" in names else []
    nocap = [m for m in names if m.endswith("_nocap")]
    appendix = [m for m in names if m not in main and m not in nocap and (m != SEEDBS or not main)]
    scored = [m for m in names if m not in nocap or stage == "s3b"]
    eval_docs = [a for role in STAGE_EVAL[stage] for a in run.docs[role]]
    summary = summarize(rows, null_rows, config, ALL_SCENARIOS)
    draws = config["bootstrap"]
    present = [m for m in scored if any(r["method"] == m for r in rows)]
    ep = Endpoint(rows, present, draws) if present else None
    if ep is not None and not ep.scenarios:
        ep = None

    by_method, null_by_method = defaultdict(list), defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    for r in null_rows:
        null_by_method[r["method"]].append(r)
    detections, posterior = posterior_details(run, [m for m in names if m in run.states], eval_docs)

    methods = {}
    for m in names:
        sel, nul = by_method[m], null_by_method[m]
        loc = [r for r in sel if r["scenario"] in LOCALIZATION_SCENARIOS]
        seg = [r for r in sel if r["scenario"] in SEG_SCENARIOS]
        entry = {"kind": run.states[m]["kind"], "family": family_of(m), "cap": run.states[m].get("cap"),
                 "tau": run.states[m]["rules"]["M"]["pooled"].value, "fit_seconds": run.states[m].get("fit_seconds"),
                 "primary": None, "by_scenario": {}, "any_fp": rate(sum(r["M"] for r in nul), len(nul)) if nul else None,
                 "recalibrated_fp": recalibrated_false_positive([r["max_score"] for r in nul], config["alpha"]) if len(nul) >= 4 else None,
                 "boundary_seg": boundary_summary([r["boundary_errors"] for r in seg]) if seg else None,
                 "rand_index_seg": mean_of([r["rand_index"] for r in seg]),
                 "eps_error": mean_of([r["eps_error"] for r in loc]) if run.states[m]["kind"] != "seedbs" else None,
                 "posterior": posterior.get(m)}
        if ep is not None and m in ep.per:
            entry["primary"] = mean_ci(ep, m)
            entry["by_scenario"] = ep.by_scenario(m)
        for s in ALL_SCENARIOS:
            if s in summary and m in summary[s]:
                c = summary[s][m]
                entry.setdefault("scenario_cells", {})[s] = {
                    "iou": c["iou"]["mean"], "iou_ci95": c["iou"]["ci95"], "recall": c["recall"], "precision": c["precision"],
                    "rand_index": c["rand_index"], "boundary": c["boundary"], "changepoints": c["changepoints"],
                    "detect_M": c["detect_M"]}
        methods[m] = entry

    # criteria 1-6 (plan 6.3); formal only at s4
    criteria = {"formal": formal, "null_role": "null_test" if formal else "null_cal (optimistic, in-sample for tau)",
                "alpha": config["alpha"]}
    c1, c2, c3, c6 = {}, {}, {}, {}
    for m in main:
        rf = methods[m]["recalibrated_fp"]
        c1[m] = {"any_fp": methods[m]["any_fp"], "recalibrated": rf,
                 "pass": None if rf is None else bool(rf["p95"] <= 2 * config["alpha"] and rf["median"] <= config["alpha"])}
        d2 = safe_delta(ep, m, "opt_hmm")
        c2[m] = {"delta": d2, "pass": None if d2 is None else bool(d2["mean"] >= NONINFERIOR)}
        d3 = safe_delta(ep, m, "independent")
        c3[m] = {"delta": d3, "pass": None if d3 is None else bool(d3["ci95"][0] > 0)}
        c6[m] = safe_delta(ep, m, SEEDBS)
    criteria["1_false_alarm"] = c1
    criteria["2_noninferior_vs_opt_hmm"] = {"threshold": NONINFERIOR, "cells": c2}
    criteria["3_continuity_vs_independent"] = {"cells": c3, "any_pass": any(v["pass"] for v in c3.values() if v["pass"] is not None) if c3 else None}
    axes = {"pi_axis": {}, "f1_axis": {}}
    if param_star:
        for fam in ("hist200", param_star):
            axes["pi_axis"][fam] = safe_delta(ep, f"pd_{fam}", f"pe_{fam}")
        for mode in ("pd", "pe"):
            axes["f1_axis"][mode] = safe_delta(ep, f"{mode}_{param_star}", f"{mode}_hist200")
    criteria["4_axes"] = axes
    criteria["5_structure"] = {m: {"a00_equals_a11_rejected": s["equal_retention_rejects"],
                                   "p_value": s["equal_retention"]["p_value"] if s["equal_retention"] else None,
                                   "bounded": s["at_bound"]}
                               for m in main if (s := structure_row(m, run.states[m]))}
    criteria["6_seedbs_delta_only"] = c6

    structure = {m: s for m in names if run.states[m]["kind"] == "general" and (s := structure_row(m, run.states[m]))}
    selection = {m: {"cap": run.states[m]["selection"]["cap"], "tail": run.states[m]["selection"]["tail"],
                     "fallback": run.states[m]["selection"]["fallback"], "dev_iou": run.states[m]["selection"]["iou"],
                     "table": [{k: r[k] for k in ("cap", "tau", "flagged_positions", "iou", "ok")}
                               for r in run.states[m]["selection"]["table"]]}
                 for m in names if "selection" in run.states[m]}
    param_table = None
    if run.param_star:
        param_table = []
        for r in run.param_star["table"]:
            row = dict(r)
            for mode in ("pd", "pe"):
                row[f"{mode}_eval_iou"] = (methods.get(f"{mode}_{r['family']}", {}).get("primary") or {}).get("mean")
            param_table.append(row)
    truth = {}
    first = next((m for m in names if by_method[m]), None)
    for r in by_method.get(first, []):
        t = truth.setdefault(r["scenario"], [])
        t.append(r["eps_true"])
    truth = {s: {"articles": len(v), "eps_true_mean": float(np.mean(v)), "eps_true_min": float(np.min(v)),
                 "eps_true_max": float(np.max(v))} for s, v in truth.items()}

    warnings = []
    if manifest.get("toy"):
        warnings.append("玩具语言模型数据：仅验证管线，数字不是 OPT 的任何结论。")
    if not formal:
        warnings.append(f"阶段 {stage} 是探索性的：只在 {', '.join(STAGE_EVAL[stage])} 上评价，null 用 null_cal（τ 已在其上校准，误报是乐观值，仅作阶段内排序）；不适用正式判据。")
    if run.param_star and run.param_star["fallback"]:
        warnings.append("7 个参数族都没有资格（τ<1 且可检出位置数>0），param* 退回 beta。")
    for m, s in selection.items():
        if s["fallback"]:
            warnings.append(f"{m}: cap 候选都不合格，保留 K0。")
    for m in names:
        if methods[m]["tau"] >= 1:
            warnings.append(f"{m}: τ = 1，什么都检不出。")
    for m, s in structure.items():
        if s["at_bound"]:
            warnings.append(f"{m}: â 取到边界（0.50 或 0.999），边界受限，不当结论。")
        if s["any_nonmonotone"] and run.states[m]["mode"] in ("pe", "pd"):
            warnings.append(f"{m}: 精确 EM 出现非单调步，检查实现。")
    sb = run.states.get(SEEDBS)
    if sb and sb.get("reused_from"):
        warnings.append(f"seedbs 参数沿用 {sb['reused_from']}。")

    report = {
        "stage": stage, "formal": formal, "toy": bool(manifest.get("toy")), "warnings": warnings,
        "roles": {r: len(run.docs[r]) for r in STAGE_ROLES[stage]}, "eval_roles": list(STAGE_EVAL[stage]),
        "config": {k: config.get(k) for k in ("tokens", "context_width", "alpha", "fraction_tau", "bootstrap", "cap")},
        "param_star": run.param_star, "param_table": param_table,
        "main": main, "appendix": appendix, "nocap": nocap, "names": names,
        "localization_scenarios": list(ep.scenarios) if ep else [],
        "methods": methods, "summary": summary, "criteria": criteria, "truth": truth,
        "structure": structure, "selection": selection,
        "diagnostics": {"emission": emission_diagnostics(run), "f0": f0_diagnostic(run),
                        "eps_estimators": eps_estimators(rows), "p1_alignment": p1_alignment(rows, [m for m in names if m.startswith("p1_")]),
                        "prior_reach": prior_reach(run, detections, eval_docs),
                        "seedbs": seedbs_section(run, summary, rows), "nocap": nocap_section(run, names, summary)},
        "timings": {"total_seconds": time_total, "fit_seconds": {m: methods[m]["fit_seconds"] for m in names},
                    "mean_seconds_per_article": {m: mean_of([r["seconds"] for r in by_method[m]]) for m in names}},
        "manifest": {"articles": manifest.get("articles"), "edit_summary": manifest.get("edit_summary")}}
    return jsonable(report)


# ---------------------------------------------------------------------- markdown
def method_title(name, report):
    star = report["param_star"]["family"] if report["param_star"] else None
    return f"{name}（param*）" if star and name in (f"pd_{star}", f"pe_{star}") else name


def primary_table(report, names, title):
    scen = [s for s in LOCALIZATION_SCENARIOS if s in report["localization_scenarios"]]
    lines = [f"### {title}", "", "| 方法 | cap | 主终点：平均 IoU [95% CI] | " + " | ".join(scen) + " | any-FP | 重校准 FP 中位 / p95 | τ |",
             "|---|---|---|" + "---:|" * len(scen) + "---|---|---:|"]
    for m in names:
        e = report["methods"][m]
        rf = e["recalibrated_fp"]
        lines.append(f"| {method_title(m, report)} | {e['cap'] if e['cap'] is not None else '—'} | {ci_text(e['primary'])} | "
                     + " | ".join(num(e["by_scenario"].get(s)) for s in scen)
                     + f" | {rate_text(e['any_fp'])} | {'NA' if rf is None else num(rf['median']) + ' / ' + num(rf['p95'])} | {num(e['tau'], 4)} |")
    return lines + [""]


def secondary_table(report, names, title):
    lines = [f"### {title}", "", "| 方法 | 边界误差均值 / 中位（seg*）| Rand index（seg*）| ε 误差 | mean γ − ε̂（带符号 / 绝对）| 后验饱和 |",
             "|---|---|---:|---:|---|---:|"]
    for m in names:
        e = report["methods"][m]
        b, p = e["boundary_seg"], e["posterior"]
        lines.append(f"| {m} | {'NA' if not b else num(b['mean'], 2) + ' / ' + num(b['median'], 2) + ' (n=' + str(b['matched']) + ')'} | "
                     f"{num(e['rand_index_seg'])} | {num(e['eps_error'])} | "
                     f"{'NA' if not p else num(p['gap_signed']) + ' / ' + num(p['gap_abs'])} | {num(p['saturation'], 4) if p else 'NA'} |")
    return lines + [""]


def scenario_table(report, names):
    lines = ["| 方法 | " + " | ".join(ALL_SCENARIOS) + " |", "|---|" + "---:|" * len(ALL_SCENARIOS)]
    for m in names:
        cells = report["methods"][m].get("scenario_cells", {})
        lines.append(f"| {m} | " + " | ".join(num(cells[s]["iou"]) if s in cells else "—" for s in ALL_SCENARIOS) + " |")
    return lines + ["", "`full_wm` 的 truth 全为 1，IoU 就是 recall，不作主终点。", ""]


def render(report):
    out, add = [], None
    add = out.extend
    stage = report["stage"]
    star = report["param_star"]["family"] if report["param_star"] else None
    add([f"# L1 报告（阶段 {stage}）", ""])
    if report["warnings"]:
        add(["## 警告", ""] + [f"- {w}" for w in report["warnings"]] + [""])
    add([f"- 角色篇数：{report['roles']}；评价角色：{report['eval_roles']}",
         f"- α = {report['config']['alpha']}，bootstrap = {report['config']['bootstrap']}，主终点场景：{report['localization_scenarios']}", ""])

    if report["main"]:
        add([f"## 主表：2×2 四格 + seedbs（`param*` = **{star}**）", "",
             "| `f₁` \\ `π` | PD | PE |", "|---|---|---|",
             f"| hist200 | pd_hist200 | pe_hist200 |", f"| {star}（param*） | pd_{star} | pe_{star} |", ""])
        add(primary_table(report, report["main"] + ([SEEDBS] if SEEDBS in report["methods"] else []), "主终点与误报"))
        add(secondary_table(report, report["main"] + ([SEEDBS] if SEEDBS in report["methods"] else []), "次终点"))
        add(["### 各场景 IoU", ""] + scenario_table(report, report["main"] + ([SEEDBS] if SEEDBS in report["methods"] else [])))

    if report["param_table"]:
        add([f"### param* 入选表（`dev` 上选，不读 dev2 / test；fallback = {report['param_star']['fallback']}）", "",
             "| 族 | 资格 | PD dev IoU (cap) | PE dev IoU (cap) | 得分 | PD 评价集 IoU | PE 评价集 IoU |", "|---|---|---|---|---:|---:|---:|"])
        for r in report["param_table"]:
            mark = " ←" if r["family"] == star else ""
            add([f"| {r['family']}{mark} | {r['eligible']} | {num(r['pd_iou'])} ({r['pd_cap']}) | {num(r['pe_iou'])} ({r['pe_cap']}) | "
                 f"{num(r['score'])} | {num(r['pd_eval_iou'])} | {num(r['pe_eval_iou'])} |"])
        add(["", "评价集 IoU 只描述，不用于选族。", ""])

    cr = report["criteria"]
    if report["main"]:
        add([f"## 判据（四格；{'正式，一次性 test' if cr['formal'] else '探索性，不作结论'}）", "", f"null：{cr['null_role']}", "",
             "| 格 | 1 误报（p95 ≤ 2α 且中位 ≤ α） | 2 对 opt_hmm ΔIoU ≥ −0.01 | 3 对 independent 的 ΔIoU（下界 > 0） | 6 对 seedbs ΔIoU（只报） |", "|---|---|---|---|---|"])
        for m in report["main"]:
            c1, c2, c3 = cr["1_false_alarm"][m], cr["2_noninferior_vs_opt_hmm"]["cells"][m], cr["3_continuity_vs_independent"]["cells"][m]
            rf = c1["recalibrated"]
            add([f"| {method_title(m, report)} | {'NA' if rf is None else num(rf['median']) + ' / ' + num(rf['p95'])} → {c1['pass']} | "
                 f"{ci_text(c2['delta'])} → {c2['pass']} | {ci_text(c3['delta'])} → {c3['pass']} | {ci_text(cr['6_seedbs_delta_only'][m])} |"])
        add(["", f"3：任一格显著为正 = {cr['3_continuity_vs_independent']['any_pass']}。", "", "### 4 两条轴（只报方向与区间，不设门槛）", "",
             "`param*` 经过 7 选 1，hist200 没有，按“流程”措辞。", "", "| 轴 | 比较 | ΔIoU [95% CI] |", "|---|---|---|"])
        for fam, d in cr["4_axes"]["pi_axis"].items():
            add([f"| π 轴 | pd_{fam} − pe_{fam} | {ci_text(d)} |"])
        for mode, d in cr["4_axes"]["f1_axis"].items():
            add([f"| f₁ 轴 | {mode}_{star} − {mode}_hist200 | {ci_text(d)} |"])
        add(["", "### 5 结构结论（a₀₀ = a₁₁ 似然比，p < 0.05 才声称保持率不同）", "", "| 格 | 拒绝 a₀₀=a₁₁ | p 值 | 贴边 |", "|---|---|---:|---|"])
        for m, s in cr["5_structure"].items():
            add([f"| {m} | {s['a00_equals_a11_rejected']} | {num(s['p_value'], 4)} | {s['bounded']} |"])
        add([""])

    if report["appendix"]:
        add(["## 附录表", "", "与主表同一次运行，τ 校准、终点、bootstrap 下标相同。", ""])
        add(primary_table(report, report["appendix"], "附录：主终点与误报"))
        add(secondary_table(report, report["appendix"], "附录：次终点"))
        add(["### 附录：各场景 IoU", ""] + scenario_table(report, report["appendix"]))

    if report["structure"]:
        add(["## 结构终点与 EM 健康度", "",
             "| 方法 | â₀₀ [CI] | â₁₁ [CI] | π̂₁ [CI] | λ̂ | 半衰期 | LRT p | cap | 迭代 | 多峰 | 贴边 | ℓ_EM − ℓ_网格 | 平台 | 重估位移 |",
             "|---|---|---|---|---:|---:|---:|---|---:|---|---|---:|---|---|"])
        for m, s in report["structure"].items():
            b = s["bootstrap"] or {}
            show = lambda k, v: f"{num(v)} [{num(b[k]['ci95'][0])}, {num(b[k]['ci95'][1])}]" if k in b else num(v)
            sh = s["shift_from_k0"]
            add([f"| {m} | {show('a00', s['a00'])} | {show('a11', s['a11'])} | {show('pi1', s['pi1']) if s['pi1'] is not None else '—'} | "
                 f"{num(s['lambda'])} | {num(s['half_life'], 1)} | {num(s['equal_retention']['p_value'], 4) if s['equal_retention'] else 'NA'} | "
                 f"{s['cap']} | {s['iterations']} | {s['multimodal']} | {s['at_bound']} | {num(s['grid_gap'], 6)} | "
                 f"{s['plateau']['a00_range'] if s['plateau'] else 'NA'}×{s['plateau']['a11_range'] if s['plateau'] else ''} | "
                 f"{'NA' if not sh else num(sh['a00'], 4) + ', ' + num(sh['a11'], 4)} |"])
        add(["", "bootstrap 区间只对 param* 的两格、hist200、hist25、ecdf400 与附录 P1 / P2 报；其余参数族候选不做 bootstrap（plan 5.2）。", ""])

    if report["selection"]:
        add(["## cap 选择（dev）", "", "| 方法 | 入选 cap | tail | dev IoU | fallback | 候选 IoU（cap: IoU，ok） |", "|---|---|---|---:|---|---|"])
        for m, s in report["selection"].items():
            cands = "; ".join(f"{r['cap']}: {num(r['iou'])}{'' if r['ok'] else '✗'}" for r in s["table"])
            add([f"| {m} | {s['cap']} | {s['tail']} | {num(s['dev_iou'])} | {s['fallback']} | {cands} |"])
        add([""])

    d = report["diagnostics"]
    add(["## 诊断", ""])
    if d["emission"]:
        e = d["emission"]
        add(["### 发射族：尾部质量与 2 折拟合优度", "",
             f"calibration ∪ calibration2 上 y ≥ 0.99 的经验占比 = {num(e['calibration_tail99'], 4)}", "",
             "| 族 | ∫f（应≈1） | y∈[0.99,1] 质量 | ÷hist200 | ÷hist25 | max f₁ | 2 折平均 log 密度 | 折间 L1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        folds = (e["folds"] or {}).get("families", {})
        for f, s in e["families"].items():
            fd = folds.get(f, {})
            add([f"| {f} | {num(s['mass_total'], 4)} | {num(s['mass_tail99'], 4)} | {num(s['tail99_vs_hist200'])} | {num(s['tail99_vs_hist25'])} | "
                 f"{num(s['max_f1'], 1)} | {num(fd.get('heldout_mean_log_density'), 4)} | {num(fd.get('l1_between_folds'), 3)} |"])
        add(["", "拟合优度只报，不用于选族。", ""])
        add(diagnostic_explanation(report, "emission"))
    if d["f0"]:
        add(["### f̂₀（null_cal 上同分箱密度）", "", "| 分箱 | sup\\|f̂₀−1\\| | iid U(0,1) 参考 |", "|---|---:|---:|"])
        for k in ("hist25", "hist200"):
            add([f"| {k} | {num(d['f0'][k]['sup_abs_dev'])} | {num(d['f0'][k]['iid_uniform_reference'])} |"])
        add([""])
        add(diagnostic_explanation(report, "f0"))
    if d["eps_estimators"]:
        e = d["eps_estimators"]
        add(["### ε̂ 估计器：旧（opt_hmm）vs ε̂_opt（opt_hmm_li）", "",
             f"文章 {e['articles']}；差均值 {num(e['mean_diff'])}，绝对差 {num(e['mean_abs_diff'])}；对真值误差 旧 {num(e['error_old'])} / opt {num(e['error_opt'])}；"
             f"旧估计取 0 或 1 的篇数 {e['old_at_endpoint']}，ε̂_opt 贴 1e-3 边界的篇数 {e['opt_at_boundary']}。", ""])
        add(diagnostic_explanation(report, "eps_estimators"))
    if d["prior_reach"]:
        add(["### PD 的首位先验管多远（PD vs 同族 P2 标签不同的位置，按位置五等分）", "", "| 族 | 不同位置占比 | 五等分分布 |", "|---|---:|---|"])
        for f, s in d["prior_reach"].items():
            add([f"| {f} | {num(s['differing_fraction'], 4)} | {[round(v, 3) for v in s['by_position_fifth']]} |"])
        add([""])
        add(diagnostic_explanation(report, "prior_reach"))
    if d["p1_alignment"]:
        add(["### P1 对齐质量（附录）", "", "| 方法 | ᾱ | 对齐偏差>0.02比例（原 clip 字段） | 触发 ᾱ<0.05 回退比例 | \\|先验边际−ε̂\\| 均值 / p95 |", "|---|---:|---:|---:|---|"])
        for m, s in d["p1_alignment"].items():
            add([f"| {m} | {num(s['alpha_bar'], 4)} | {num(s['pi_clipped'])} | {num(s['pi_unidentifiable'])} | "
                 f"{num(s['marginal_gap_mean'])} / {num(s['marginal_gap_p95'])} |"])
        add([""])
        add(diagnostic_explanation(report, "p1_alignment"))
    if d["seedbs"]:
        s = d["seedbs"]
        add(["### seedbs", "", f"选出 B = {s['B']}，ζ = {s['zeta']}，c = {num(s['c'], 4)}"
             + (f"；null_cal 每篇假切换点 {num(s['false_changepoints_per_null'])}；每篇 {num(s['seconds_per_article'], 2)} 秒" if s["chosen"] else ""), ""])
        if s["recovery"]:
            add(["| 场景 | 真值切换点 | 检出切换点 | 真值切换点到最近检出点的距离 |", "|---|---:|---:|---:|"])
            for sc, r in s["recovery"].items():
                add([f"| {sc} | {num(r['true'], 2)} | {num(r['predicted'], 2)} | {num(r['nearest_distance'], 2)} |"])
            add([""])
        add(diagnostic_explanation(report, "seedbs_recovery"))
        add(["| B | ζ | τ | c | dev IoU | 可检出位置 | 假切换点/篇 | 合格 |", "|---:|---:|---:|---:|---:|---:|---:|---|"])
        for r in s["table"]:
            add([f"| {r['B']} | {r['zeta']} | {num(r['tau'], 4)} | {num(r['c'], 4)} | {num(r['iou'])} | {r['flagged_positions']} | "
                 f"{num(r['false_changepoints_per_null'], 2)} | {r['ok']} |"])
        add([""])
        add(diagnostic_explanation(report, "seedbs_selection"))
    if d["nocap"]:
        add(["### nocap 旁支（不进主表、不参与选择）", "",
             "| 方法 | τ | 参照 τ | null γ>0.999 占比 | max γ=1 | max g（EM 训练） | â₀₀ / â₁₁ | 各场景 IoU（nocap / 参照） |", "|---|---:|---:|---:|---|---:|---|---|"])
        for r in d["nocap"]:
            em = r["em"] or {}
            ious = "; ".join(f"{s}: {num(v)} / {num(r.get('iou_reference', {}).get(s))}" for s, v in r.get("iou", {}).items())
            add([f"| {r['method']} | {num(r['tau'], 4)} | {num(r['tau_reference'], 4)} | {num(r['null_gamma_gt_999'], 4)} | {r['null_max_gamma_is_one']} | "
                 f"{num(r['max_g_train'], 1)} | {num(em.get('a00'))} / {num(em.get('a11'))} | {ious} |"])
        add([""])

    add(["## 真值", "", "| 场景 | 文章 | ε_true 均值 | 最小 | 最大 |", "|---|---:|---:|---:|---:|"])
    for s, t in report["truth"].items():
        add([f"| {s} | {t['articles']} | {num(t['eps_true_mean'])} | {num(t['eps_true_min'])} | {num(t['eps_true_max'])} |"])
    add([""])
    add(diagnostic_explanation(report, "truth"))
    t = report["timings"]
    add(["", "## 耗时", "", f"- 总计 {num(t['total_seconds'], 1)} 秒；各方法拟合秒数：`{ {k: round(v, 1) for k, v in t['fit_seconds'].items() if v is not None} }`", ""])
    text = display_names("\n".join(out))
    if any(name in report["methods"] for name in REPORT_NAMES):
        title, body = text.split("\n", 1)
        text = title + "\n\n" + "\n".join(method_name_guide()) + body
    return text
