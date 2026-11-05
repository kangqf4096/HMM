"""L1 report (plan_general_transition 6, 9): build_report -> JSON-serialisable dict, render -> markdown.

Everything in the markdown comes from the dict. A diagnostic whose inputs a stage does not have is skipped.
Only stage s4 has formal criteria (one-shot test); every other stage is exploratory, on dev2 / dev with null_cal.
"""
import math
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
             f"calibration 上 y ≥ 0.99 的经验占比 = {num(e['calibration_tail99'], 4)}", "",
             "| 族 | ∫f（应≈1） | y∈[0.99,1] 质量 | ÷hist200 | ÷hist25 | max f₁ | 2 折平均 log 密度 | 折间 L1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        folds = (e["folds"] or {}).get("families", {})
        for f, s in e["families"].items():
            fd = folds.get(f, {})
            add([f"| {f} | {num(s['mass_total'], 4)} | {num(s['mass_tail99'], 4)} | {num(s['tail99_vs_hist200'])} | {num(s['tail99_vs_hist25'])} | "
                 f"{num(s['max_f1'], 1)} | {num(fd.get('heldout_mean_log_density'), 4)} | {num(fd.get('l1_between_folds'), 3)} |"])
        add(["", "拟合优度只报，不用于选族。", ""])
    if d["f0"]:
        add(["### f̂₀（null_cal 上同分箱密度）", "", "| 分箱 | sup\\|f̂₀−1\\| | iid U(0,1) 参考 |", "|---|---:|---:|"])
        for k in ("hist25", "hist200"):
            add([f"| {k} | {num(d['f0'][k]['sup_abs_dev'])} | {num(d['f0'][k]['iid_uniform_reference'])} |"])
        add([""])
    if d["eps_estimators"]:
        e = d["eps_estimators"]
        add(["### ε̂ 估计器：旧（opt_hmm）vs ε̂_opt（opt_hmm_li）", "",
             f"文章 {e['articles']}；差均值 {num(e['mean_diff'])}，绝对差 {num(e['mean_abs_diff'])}；对真值误差 旧 {num(e['error_old'])} / opt {num(e['error_opt'])}；"
             f"旧估计取 0 或 1 的篇数 {e['old_at_endpoint']}，ε̂_opt 贴 1e-3 边界的篇数 {e['opt_at_boundary']}。", ""])
    if d["prior_reach"]:
        add(["### PD 的首位先验管多远（PD vs 同族 P2 标签不同的位置，按位置五等分）", "", "| 族 | 不同位置占比 | 五等分分布 |", "|---|---:|---|"])
        for f, s in d["prior_reach"].items():
            add([f"| {f} | {num(s['differing_fraction'], 4)} | {[round(v, 3) for v in s['by_position_fifth']]} |"])
        add([""])
    if d["p1_alignment"]:
        add(["### P1 对齐质量（附录）", "", "| 方法 | ᾱ | clip 比例 | 不可识别比例 | \\|先验边际−ε̂\\| 均值 / p95 |", "|---|---:|---:|---:|---|"])
        for m, s in d["p1_alignment"].items():
            add([f"| {m} | {num(s['alpha_bar'], 4)} | {num(s['pi_clipped'])} | {num(s['pi_unidentifiable'])} | "
                 f"{num(s['marginal_gap_mean'])} / {num(s['marginal_gap_p95'])} |"])
        add([""])
    if d["seedbs"]:
        s = d["seedbs"]
        add(["### seedbs", "", f"选出 B = {s['B']}，ζ = {s['zeta']}，c = {num(s['c'], 4)}"
             + (f"；null_cal 每篇假切换点 {num(s['false_changepoints_per_null'])}；每篇 {num(s['seconds_per_article'], 2)} 秒" if s["chosen"] else ""), ""])
        if s["recovery"]:
            add(["| 场景 | 真值切换点 | 检出切换点 | 到最近真值的距离 |", "|---|---:|---:|---:|"])
            for sc, r in s["recovery"].items():
                add([f"| {sc} | {num(r['true'], 2)} | {num(r['predicted'], 2)} | {num(r['nearest_distance'], 2)} |"])
            add([""])
        add(["| B | ζ | τ | c | dev IoU | 可检出位置 | 假切换点/篇 | 合格 |", "|---:|---:|---:|---:|---:|---:|---:|---|"])
        for r in s["table"]:
            add([f"| {r['B']} | {r['zeta']} | {num(r['tau'], 4)} | {num(r['c'], 4)} | {num(r['iou'])} | {r['flagged_positions']} | "
                 f"{num(r['false_changepoints_per_null'], 2)} | {r['ok']} |"])
        add([""])
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
    t = report["timings"]
    add(["", "## 耗时", "", f"- 总计 {num(t['total_seconds'], 1)} 秒；各方法拟合秒数：`{ {k: round(v, 1) for k, v in t['fit_seconds'].items() if v is not None} }`", ""])
    return "\n".join(out)
