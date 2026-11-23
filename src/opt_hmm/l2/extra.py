"""Extra arms and literature baselines on a finished L2 run (CPU): train side first, test side read once afterwards.

HMM arms, fitted like l2-fit but without the bootstrap: fb = Beta-shaped f1(e') with fe's kernel weights, bm = one fixed
Beta of the pure articles. Baselines (pivots only, see baselines.py): spot_opt, spot_ini, win_trgof, win_ars.

Two operating points. Document level (the project protocol): tau on null_cal so that any-FP <= alpha (rule M). Token
level (suffix _tok): tau on null_cal so that at most alpha of the pooled null positions are flagged, a looser criterion
reported with both false-positive rates. fe, m400 and the HMM arms enter it with their frozen parameters (only tau is
recalibrated); baselines are tuned on dev under each criterion separately. The reference rows of fe, m400 and seedbs come
from the reference test run, so every difference is a paired bootstrap over the same test articles.
"""
import copy
from pathlib import Path
from time import perf_counter

import numpy as np

from ..calibration import Threshold
from ..detector import Detection
from ..io import new_output, provenance, read_jsonl, write_json, write_jsonl
from ..l0.decide import build_rules, decide, rules_to_dict
from ..paths import project_path
from . import baselines as bl
from .config import validate_l2
from .data import load_side
from .evaluate import TEST_SPLITS, check_gate, load_frozen, rebuild_methods, score_rows
from .fit import check_side, fit_method
from .methods import EXTRA_ARMS, MODES, Trainer

NEW_HMM = tuple(f"{mode}_{arm}" for arm in EXTRA_ARMS for mode in MODES)
REFERENCE = ("pd_m400", "pe_m400", "pd_fe", "pe_fe", "seedbs")
TOKEN_HMM = ("pd_m400", "pe_m400", "pd_fe", "pe_fe", "pd_fb", "pe_fb")
BASELINES = ("spot_opt", "spot_ini", "win_trgof", "win_ars")
WINDOWS = (10, 20, 40, 60)


def memoized(fn):
    cache = {}

    def run(stack):
        if id(stack) not in cache:
            cache[id(stack)] = fn(stack)
        return cache[id(stack)]
    return run


# ------------------------------------------------ scorers: stack -> (scores (D, n), eps (D,) or None, identifiable (D,))
def spot_scorer(kind, stage1, cap):
    def run(stack):
        if kind == "ini":
            eps, ident = bl.tail_eps(stack.y), np.ones(len(stack), bool)
        else:
            value, ident, _ = stage1.opt(cap).estimate_many(stack.counts)
            eps = np.clip(value, 0., 1.)
        return np.where(ident[:, None], bl.SpotCurve(stack.y, eps).scores(), 0.), eps, ident
    return memoized(run)


def window_scorer(family, B, s=None):
    def run(stack):
        score = bl.window_trgof(stack.y, B, s) if family == "win_trgof" else bl.window_aaronson(stack.y, B)
        return score, None, np.ones(len(stack), bool)
    return memoized(run)


def hmm_scorer(method):
    def run(stack):
        gamma, eps, ident, _ = method.scores(stack)
        return gamma, eps, ident
    return memoized(run)


def candidates(stage1, config):
    """Baseline families -> [(params, scorer)]; SPOT-opt's cap is tuned on dev like the HMM arms'."""
    return {"spot_opt": [({"cap": c}, spot_scorer("opt", stage1, c)) for c in config["l2"]["cap_grid"]],
            "spot_ini": [({}, spot_scorer("ini", None, None))],
            "win_trgof": [({"B": B, "s": s}, window_scorer("win_trgof", B, s)) for B in WINDOWS for s in (1, 2)],
            "win_ars": [({"B": B}, window_scorer("win_ars", B)) for B in WINDOWS]}


# ------------------------------------------------ calibration on null_cal and selection on dev
def token_rules(null_scores, alpha):
    """tau with at most alpha of the pooled null positions strictly above it."""
    flat = np.sort(np.asarray(null_scores, float).ravel())
    t = Threshold(float(flat[len(flat) - 1 - int(np.floor(alpha * len(flat)))]), len(flat), len(flat), alpha, True)
    return {"tau_fraction": "same", "group_sizes": {}, "M": {"pooled": t}, "P": {"pooled": t}, "P_same": {"pooled": t}}


def calibrate(scorer, null, config, criterion):
    scores = scorer(null)[0]
    if criterion == "doc":
        return build_rules([(null.n_eff, row) for row in scores], config)
    return token_rules(scores, config["alpha"])


def tau_of(rules):
    return rules["M"]["pooled"].value


def dev_row(scorer, rules, dev):
    labels = [decide(row, dev.n_eff, rules)[0] for row in scorer(dev)[0]]
    classes, value, flagged = Trainer.dev_scores(labels, dev)
    tau = tau_of(rules)
    return {"tau": tau, "iou": value, "iou_by_class": classes, "flagged_positions": flagged,
            "ok": bool(tau < 1 and flagged > 0)}


def select(options, null, dev, config, criterion):
    """Best usable configuration by dev IoU (the first one wins a tie); None when no configuration is usable."""
    table, best = [], None
    for params, scorer in options:
        rules = calibrate(scorer, null, config, criterion)
        row = {"params": params, **dev_row(scorer, rules, dev)}
        table.append(row)
        if row["ok"] and (best is None or round(row["iou"], 12) > round(best["row"]["iou"], 12)):
            best = {"row": row, "scorer": scorer, "rules": rules}
    return best, table


# ------------------------------------------------ reference check, diagnostics, test rows
def with_extra_arms(config):
    out = copy.deepcopy(config)
    for arm in EXTRA_ARMS:
        out["l2"]["arms"][arm] = dict(config["l2"]["arms"]["fe"])
    return out


def check_reference(frozen, trainer):
    """The training side and the estimation code are the ones behind the reference fit: stage 1 and the fe kernel match."""
    key = trainer.l2["arms"]["fe"]["pure_articles"]
    if trainer.stage1[key].counts.tolist() != frozen["stage1"][str(key)]["counts"]:
        raise ValueError("The training side differs from the one behind the reference fit (stage-1 counts).")
    entry = frozen["methods"]["pd_fe"]
    kernel = trainer.kernel("fe", entry["cap"], entry["h"])
    diff = float(np.abs(kernel.density - np.array(entry["kernel"]["density"])).max())
    if diff > 1e-6:
        raise ValueError(f"The fe kernel is not reproduced by the current code (max density difference {diff:.3g}).")
    return {"stage1_counts_equal": True, "fe_kernel_max_abs_diff": diff}


def beta_vs_hist(trainer, method):
    """Per grid point: n_eff, Beta (a, b) and the total-variation distance to the kernel hist200 with the same weights."""
    hist, beta = trainer.kernel("fe", method.cap, method.h), method.kernel
    return [{"e": float(e), "neff": float(n), "a": float(p[0]), "b": float(p[1]), "tv": float(.5 * np.abs(d - f).mean())}
            for e, n, p, d, f in zip(beta.grid, hist.neff, beta.params, hist.density, beta.density)]


def detections(scorer, stack):
    scores, eps, ident = scorer(stack)
    return [Detection(scores[i], float("nan") if eps is None else float(eps[i]), 0., bool(ident[i]), 0.)
            for i in range(len(stack))]


def evaluate(name, scorer, rules, splits):
    """Rows of the four test blocks and of null_test for one calibrated scorer."""
    rows = []
    for split in TEST_SPLITS:
        rows += score_rows(name, splits[split].docs, detections(scorer, splits[split]), rules)
    null = score_rows(name, splits["null_test"].docs, detections(scorer, splits["null_test"]), rules)
    if scorer(splits["null_test"])[1] is None:
        for r in rows + null:
            r["eps_hat"] = r["eps_error"] = None
    return rows, null


def check_same_articles(new_rows, ref_rows):
    """Paired differences need every method on the same test articles."""
    ids = lambda rows: {(r["split"], r["article_id"]) for r in rows}
    differ = ids(new_rows) ^ ids(ref_rows)
    if differ:
        raise ValueError(f"The reference test run covers other articles than this test side ({len(differ)} differ).")


def null_rates(scorer, rules, null):
    """Calibration-set rates of a frozen rule: documents with any flagged position, and flagged share of all positions."""
    scores = scorer(null)[0]
    flagged = scores > tau_of(rules)
    return {"any_fp": float(flagged.any(axis=1).mean()), "position_fpr": float(flagged.mean())}


def final_methods(fitted, reference, chosen_doc, chosen_token, null, config):
    """name -> {scorer, rules, criterion, family, config}: everything that is evaluated on the test side."""
    final = {}
    for name, fit in fitted.items():
        final[name] = {"scorer": hmm_scorer(fit["method"]), "rules": fit["method"].rules, "criterion": "doc",
                       "family": "hmm", "config": {"cap": fit["method"].cap, "h": fit["method"].h}}
    methods = {**{n: f["method"] for n, f in fitted.items()}, **{n: reference[n] for n in TOKEN_HMM if n in reference}}
    for name in TOKEN_HMM:
        scorer = hmm_scorer(methods[name])
        final[f"{name}_tok"] = {"scorer": scorer, "rules": token_rules(scorer(null)[0], config["alpha"]),
                                "criterion": "token", "family": "hmm", "config": {"cap": methods[name].cap, "h": methods[name].h}}
    for suffix, chosen in (("", chosen_doc), ("_tok", chosen_token)):
        for family, c in chosen.items():
            final[family + suffix] = {"scorer": c["scorer"], "rules": c["rules"], "family": family,
                                      "criterion": "token" if suffix else "doc", "config": c["row"]["params"]}
    return final


def run_extra(frozen_path, train_dir, test_dir, reference_dir, output):
    from .extra_report import build_report, render_extra
    from .report import jsonable
    started = perf_counter()
    frozen = load_frozen(frozen_path)
    train = load_side(train_dir)
    config = train["manifest"]["config"]
    validate_l2(config)
    check_side(train, config)
    out = new_output(project_path(output))
    null, dev = train["splits"]["null_cal"], train["splits"]["dev"]
    trainer = Trainer(with_extra_arms(config), train)
    check = check_reference(frozen, trainer)
    fitted = {name: fit_method(trainer, name, uncertainty=False) for name in NEW_HMM}
    reference = rebuild_methods(frozen)
    chosen, tables = {"doc": {}, "token": {}}, {}
    options = candidates(trainer.arm_stage1["fe"], config)
    for criterion in chosen:
        for family, opts in options.items():
            best, table = select(opts, null, dev, config, criterion)
            tables[f"{family}/{criterion}"] = table
            if best:
                chosen[criterion][family] = best
    final = final_methods(fitted, reference, chosen["doc"], chosen["token"], null, config)
    write_json(out / "frozen_extra.json", jsonable({
        "stage": "extra", "reference_fit": str(project_path(frozen_path)), "train_key_id": train["manifest"]["key_id"],
        "reference_check": check,
        "methods": {n: {"criterion": m["criterion"], "family": m["family"], "config": m["config"],
                        "null_cal": null_rates(m["scorer"], m["rules"], null), "rules": rules_to_dict(m["rules"])}
                    for n, m in final.items()},
        "hmm": {n: {**f["method"].to_dict(), "rules": rules_to_dict(f["method"].rules)} for n, f in fitted.items()}}))
    selection = {"hmm": {n: {"selection": f["selection"], "em": f["em"], "dev_eps_error": f["dev_eps_error"],
                            "beta_vs_hist": beta_vs_hist(trainer, f["method"]) if f["method"].arm == "fb" else None,
                            "params": f["method"].params, "tau": tau_of(f["method"].rules)} for n, f in fitted.items()},
                 "baselines": tables, "infeasible": [f"{f}/{c}" for c in chosen for f in options if f not in chosen[c]]}
    write_json(out / "fit_extra.json", jsonable(selection))

    test = load_side(test_dir)
    test_config = test["manifest"]["config"]
    validate_l2(test_config)
    check_gate(frozen, test_config, test["manifest"])
    rows, null_rows = [], []
    for name, item in final.items():
        r, n = evaluate(name, item["scorer"], item["rules"], test["splits"])
        rows, null_rows = rows + r, null_rows + n
    ref = [r for r in read_jsonl(Path(project_path(reference_dir)) / "metrics.jsonl") if r["method"] in REFERENCE]
    check_same_articles(rows + null_rows, ref)
    split_null = lambda rs, keep: [r for r in rs if (r["split"] == "null_test") == keep]
    meta = {n: {"criterion": "doc", "config": {"cap": m["cap"], "h": m["h"]}} for n, m in frozen["methods"].items()}
    meta["seedbs"] = {"criterion": "doc", "config": {k: frozen["seedbs"]["seedbs"][k] for k in ("B", "zeta")}}
    meta.update({n: {k: m[k] for k in ("criterion", "family", "config")} for n, m in final.items()})
    report = build_report(config, rows + split_null(ref, False), null_rows + split_null(ref, True), meta, selection)
    report["reference_check"] = check
    report["seconds"] = perf_counter() - started
    report = jsonable(report)
    write_json(out / "report.json", report)
    (out / "report.md").write_text(render_extra(report), encoding="utf-8")
    write_jsonl(out / "metrics.jsonl", [{k: v for k, v in r.items() if k != "boundary_errors"} for r in rows + null_rows])
    write_json(out / "provenance.json", provenance())
    return {"report": str(out / "report.md"), "seconds": report["seconds"]}
