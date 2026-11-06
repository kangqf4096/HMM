"""L2 fit stage: everything up to the freeze, on the training side only (plan 4.1, steps 1-7).

Test roles are not loadable here: the test side is a different directory that only `l2.test` opens, and only
after `frozen.json` exists.
"""
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ..io import new_output, provenance, write_json
from ..l0.decide import rules_to_dict
from ..paths import project_path
from ..progress import progress, tick
from .config import validate_l2
from .data import load_side
from .methods import DEV_CLASSES, HMM_ARMS, Trainer, cap_key, method_names, params_from, parse
from .seedbs_run import SeedBSRunner
from .stages import default_progress_path, install_overall

TRAIN_NEEDS = ("cal_pure", "mix_r2", "mix_splice", "mix_alt", "mix_r0", "dev", "null_cal")
FIT_STAGES = ("F1", "E1", "S1", "S2", "T1", "BS")


def check_side(side, config):
    manifest = side["manifest"]
    if manifest.get("side") != "train":
        raise ValueError("The fit stage reads the training side only.")
    if manifest["key_id"] != config["l2"]["sides"]["train"]["key_id"]:
        raise ValueError("Training positions were built with a different key.")
    missing = [r for r in TRAIN_NEEDS if r not in side["splits"]]
    if missing:
        raise ValueError(f"Training side lacks roles {missing}.")
    if any(not d["truth"].all() for d in side["splits"]["cal_pure"].docs):
        raise ValueError("cal_pure must be pure surviving watermark.")
    if any(d["truth"].any() for d in side["splits"]["null_cal"].docs):
        raise ValueError("null_cal contains positive truth.")
    dev_classes = {d["cls"] for d in side["splits"]["dev"].docs}
    if not set(DEV_CLASSES) <= dev_classes:
        raise ValueError(f"dev must contain the classes {DEV_CLASSES}.")


def coverage(trainer, config):
    """Truth = 1 positions per 0.1-wide bin of e' (plan 3.2); every bin needs min_positions."""
    cov = config["l2"]["coverage"]
    lo, step, need = cov["lo"], cov["step"], cov["min_positions"]
    n = config["l2"]["arms"]["fe"]["pure_articles"]
    opt = trainer.arm_stage1["fe"].opt(config["l2"]["cap_k0"])
    pure, r2 = trainer.splits["cal_pure"], trainer.splits["mix_r2"]
    eps = np.concatenate([opt.estimate_many(pure.counts[:n])[0], opt.estimate_many(r2.counts)[0]])
    size = np.concatenate([pure.counts[:n].sum(axis=1), r2.truth_counts().sum(axis=1)])
    nb = int(np.floor((1. - lo) / step + 1e-9)) + 1
    which = np.clip(np.floor((eps - lo) / step + 1e-9).astype(int), 0, nb - 1)
    rows = []
    for k in range(nb):
        keep = which == k
        rows.append({"low": lo + k * step, "high": lo + (k + 1) * step, "articles": int(keep.sum()),
                     "positions": int(size[keep].sum()), "ok": bool(size[keep].sum() >= need)})
    return {"cap": config["l2"]["cap_k0"], "min_positions": need, "bins": rows,
            "deficient": [r for r in rows if not r["ok"]]}


def f0_check(null_stack, draws=50, seed=0):
    """Pooled null_cal pivot histogram against the uniform law, next to iid uniform draws of the same size."""
    bins = null_stack.counts.shape[1]
    counts = null_stack.counts.sum(axis=0)
    total = int(counts.sum())

    def stats(c):
        expected = total / bins
        return {"sup_dev": float(np.abs(c / expected - 1.).max()), "chi2": float(((c - expected) ** 2 / expected).sum())}

    rng = np.random.default_rng(seed)
    ref = [stats(np.bincount(np.minimum((rng.random(total) * bins).astype(int), bins - 1), minlength=bins))
           for _ in range(draws)]
    return {"positions": total, "observed": stats(counts), "dof": bins - 1,
            "iid_reference": {k: {"mean": float(np.mean([r[k] for r in ref])),
                                  "p95": float(np.quantile([r[k] for r in ref], .95))} for k in ("sup_dev", "chi2")}}


def fit_method(trainer, name):
    """One method end to end: A at K0 per bandwidth, joint (h, cap) selection on dev, refit, tau, uncertainty."""
    l2 = trainer.l2
    mode, arm = parse(name)
    hs = l2["h_grid"] if arm == "fe" else [None]
    k0, fits, table = l2["cap_k0"], {}, []
    for h in hs:
        base = trainer.method(name, k0, h)
        result = trainer.fit_transition(base)
        tick("E1")
        fits[(cap_key(k0), h)] = result
        params0 = params_from(result["fit"])
        for cap in l2["cap_grid"]:
            table.append(trainer.evaluate_candidate(trainer.method(name, cap, h, params0)))
            tick("S1")
    best = trainer.best(table)
    fallback = best is None
    cap, h = (k0, hs[-1]) if fallback else (best["cap"], best["h"])
    key = (cap_key(cap), h)
    if key in fits:
        result = fits[key]
        tick("E1", cached=True)
    else:
        result = trainer.fit_transition(trainer.method(name, cap, h))
        tick("E1")
    method = trainer.method(name, cap, h, params_from(result["fit"]))
    rules, null = trainer.null_rules(method)
    method.rules = rules
    tick("T1")
    uncertainty = trainer.uncertainty(method, result)
    tick("E1")
    tick("BS", l2["em"]["bootstrap_draws"])
    dev = trainer.splits["dev"]
    eps, ident = method.proportions(dev.counts)[:2]
    by_class = defaultdict(list)
    for doc, e in zip(dev.docs, eps):
        by_class[doc["cls"]].append(abs(e - doc["eps_true"]))
    return {"method": method, "selection": {"table": table, "best": best, "fallback": fallback},
            "em": result["fit"], "uncertainty": uncertainty, "null_cal_max": [float(d.scores.max()) for d in null],
            "dev_eps_error": {c: float(np.mean(v)) for c, v in by_class.items()},
            "e_prime_identifiable": float(np.mean(ident)),
            "neff": None if method.kernel is None else
            {"grid": method.kernel.grid.tolist(), "neff": method.kernel.neff.tolist()}}


def freeze(config, side, trainer, fitted, seedbs_states, cov, l_top):
    return {
        "stage": "fit",
        "config": {k: config[k] for k in ("tokens", "context_width", "bins", "pseudocount", "alpha", "fraction_tau",
                                          "min_group_size", "bootstrap", "seed")},
        "l2": {k: config["l2"][k] for k in ("global_seed", "cap_k0", "cap_grid", "h_grid", "kernel_grid", "arms",
                                           "cap_sweep", "seedbs", "em", "mechanism", "min_neff")},
        "train_key_id": side["manifest"]["key_id"], "l_top": l_top,
        "stage1": {str(k): s.to_dict() for k, s in trainer.stage1.items()},
        "methods": {n: {**f["method"].to_dict(), "rules": rules_to_dict(f["method"].rules)} for n, f in fitted.items()},
        "seedbs": {n: {"B": s["B"], "zeta": s["zeta"], "rules": rules_to_dict(s["rules"])}
                   for n, s in seedbs_states.items()},
        "coverage": cov}


def run_fit(train_dir, output, workers=1, cache=None, progress_path=None, weights=None):
    started = perf_counter()
    side = load_side(train_dir)
    config = side["manifest"]["config"]
    validate_l2(config)
    check_side(side, config)
    output = new_output(project_path(output))
    overall = install_overall(config, progress_path or default_progress_path(output), FIT_STAGES, weights)
    trainer = Trainer(config, side)
    for _ in trainer.stage1:
        tick("F1")
    progress("覆盖检查", 0, 1)
    cov = coverage(trainer, config)
    write_json(output / "coverage.json", cov)
    if cov["deficient"]:
        raise ValueError(f"e' coverage below {cov['min_positions']} positions in bins "
                         f"{[(round(r['low'], 2), r['positions']) for r in cov['deficient']]}; add data, do not lower the bar.")
    fitted = {}
    for number, name in enumerate(method_names(), 1):
        progress("拟合方法", number, len(method_names()), name)
        fitted[name] = fit_method(trainer, name)
    runner = SeedBSRunner(config, cache or output / "cache" / "seedbs", workers)
    seedbs_states, seedbs_table = runner.select(trainer.splits["null_cal"], trainer.splits["dev"])
    tick("S2", len(config["l2"]["seedbs"]["zeta"]))
    frozen = freeze(config, side, trainer, fitted, seedbs_states, cov, side["manifest"].get("l_top"))
    write_json(output / "frozen.json", frozen)
    from .report import build_fit_report, render_fit
    report = build_fit_report(config, side, trainer, fitted, seedbs_states, seedbs_table, cov,
                              f0_check(trainer.splits["null_cal"]), perf_counter() - started)
    write_json(output / "fit_report.json", report)
    (output / "fit_report.md").write_text(render_fit(report), encoding="utf-8")
    write_json(output / "provenance.json", provenance())
    overall.finish()
    return {"frozen": str(output / "frozen.json"), "report": str(output / "fit_report.md"),
            "seconds": perf_counter() - started}
