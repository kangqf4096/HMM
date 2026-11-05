"""L1 evaluation (plan_general_transition): global 2-parameter transition + EM, hist200 / ecdf400 emissions, seedbs.

Stage gates (what each stage may read is fixed by STAGE_ROLES; test roles are only touched by s4):
  s1  old methods + seedbs on dev2        s2  + hist25 global-transition methods (K0 cap)
  s3  all families, cap selection on dev  s3b nocap diagnostic side branch
  s4  freeze, then one run on test + test_seg + null_test
"""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from ..calibration import calibrate
from ..detector import Detection, Detector
from ..emissions import Histogram
from ..io import new_output, provenance, read_json, write_json, write_jsonl
from ..paths import project_path
from ..progress import progress
from ..proportion import OptProportion, estimate_proportion
from . import em as em_module
from . import seedbs as sb
from .config import validate_l0, validate_l1
from .decide import build_rules, decide, rules_from_dict, rules_to_dict
from ..parametric import FAMILIES as PARAM_FAMILIES
from .emission import Capped, ECDFPCHIP, V2_FAMILIES, fit_base, tail_beta
from .evaluate import cell, load_articles, score
from .metrics import (bootstrap_indices, changepoint_recovery, changepoints_from_labels, paired_difference,
                      position_metrics, rand_index)
from .schema import ALL_SCENARIOS, LOCALIZATION_SCENARIOS, SEG_SCENARIOS

LEGACY = ("opt_hmm", "independent", "fixed_e", "window")
MAIN_FIXED = ("pd_hist200", "pe_hist200")                      # the two param* cells are added once param* is chosen
PARAM_METHODS = tuple(f"{mode}_{family}" for family in PARAM_FAMILIES for mode in ("pd", "pe"))
ECDF_METHODS = ("pd_ecdf400", "pe_ecdf400", "p1_ecdf400", "p2_ecdf400")
GENERAL_S2 = ("pd_hist25", "pe_hist25", "p1_hist25", "p2_hist25")
NOCAP = ("pe_hist200_nocap", "pe_ecdf400_nocap")
SEEDBS = "seedbs"
GENERAL_ALL = MAIN_FIXED + PARAM_METHODS + ECDF_METHODS + GENERAL_S2
STAGE_METHODS = {
    "s1": LEGACY + (SEEDBS,),
    "s2": LEGACY + ("opt_hmm_li",) + GENERAL_S2 + (SEEDBS,),
    "s3": LEGACY + ("opt_hmm_li",) + GENERAL_ALL + (SEEDBS,),
    "s3b": ("pe_hist200", "pe_ecdf400") + NOCAP,
    "s4": LEGACY + ("opt_hmm_li",) + GENERAL_ALL + (SEEDBS,),
}
# Roles each stage may load for scoring; everything not listed here is never read after loading.
STAGE_ROLES = {
    "s1": ("calibration", "dev", "null_cal", "dev2"),
    "s2": ("calibration", "calibration2", "mixed_cal", "dev", "null_cal", "dev2"),
    "s3": ("calibration", "calibration2", "mixed_cal", "dev", "null_cal", "dev2"),
    "s3b": ("calibration", "calibration2", "mixed_cal", "dev", "null_cal", "dev2"),
    "s4": ("calibration", "calibration2", "mixed_cal", "dev", "null_cal", "test", "test_seg", "null_test"),
}
STAGE_EVAL = {"s1": ("dev2",), "s2": ("dev2",), "s3": ("dev2",), "s3b": ("dev", "dev2"), "s4": ("test", "test_seg")}
CAP_NONE = float("inf")


def main_methods(param_star):
    """The 2x2 main table: hist200 and the chosen parametric family, each with PD and PE."""
    return MAIN_FIXED + (() if param_star is None else (f"pd_{param_star}", f"pe_{param_star}"))


def parse_method(name):
    if name == SEEDBS:
        return {"name": name, "kind": "seedbs"}
    if name in LEGACY:
        return {"name": name, "kind": "legacy", "mode": name, "family": "hist25", "nocap": False}
    if name == "opt_hmm_li":
        return {"name": name, "kind": "li", "mode": name, "family": "hist25", "nocap": False}
    nocap = name.endswith("_nocap")
    mode, family = name.removesuffix("_nocap").split("_", 1)
    if mode not in em_module.MODES or family not in V2_FAMILIES:
        raise ValueError(f"Unknown method {name!r}.")
    return {"name": name, "kind": "general", "mode": mode, "family": family, "nocap": nocap}


def cap_key(cap):
    return CAP_NONE if cap is None else float(cap)


def _seedbs_worker(args):
    y, B, seed, kw = args
    return sb.seedbs_precompute(y, B, seed, **kw)


class Run:
    """Frozen-order pipeline for one stage."""

    def __init__(self, config, articles, stage, workers=1):
        self.config, self.l1, self.stage, self.workers = config, config["l1"], stage, workers
        self.m = config["context_width"]
        self.docs = defaultdict(list)
        for a in sorted(articles, key=lambda a: a["id"]):
            if a["split"] in STAGE_ROLES[stage]:
                self.docs[a["split"]].append(a)
        missing = [r for r in STAGE_ROLES[stage] if not self.docs[r]]
        if missing:
            raise ValueError(f"Stage {stage} needs roles {missing}.")
        for role in ("calibration", "calibration2"):
            if any(not a["truth"].all() for a in self.docs[role]):
                raise ValueError(f"{role} must be pure surviving watermark.")
        if any(a["truth"].any() for r in ("null_cal", "null_test") for a in self.docs[r]):
            raise ValueError("Null roles contain positive truth.")
        pivots = lambda roles: np.concatenate([a["y"] for r in roles for a in self.docs[r]])
        self.wm100 = pivots(("calibration", "calibration2")) if stage != "s1" else None
        self.wm50 = pivots(("calibration",))
        self.bases = {} if stage == "s1" else {f: fit_base(f, self.wm100, config) for f in V2_FAMILIES}
        self.legacy50 = Histogram.fit(self.wm50, config["bins"], config["pseudocount"])
        self._opt = {}
        self.states = {}
        self.timings = {}
        self.param_star = None

    # ------------------------------------------------------------------ densities and detectors
    def _base(self, state):
        return self.legacy50 if state.get("legacy50") else self.bases[state["family"]]

    def density(self, state, cap="state", tail="state"):
        cap = state["cap"] if cap == "state" else cap
        tail = state["tail"] if tail == "state" else tail
        return Capped(self._base(state), cap, tail)

    def opt(self, state, cap="state", tail="state"):
        cap = state["cap"] if cap == "state" else cap
        tail = state["tail"] if tail == "state" else tail
        key = (state["family"], bool(state.get("legacy50")), cap_key(cap), json.dumps(tail, sort_keys=True))
        if key not in self._opt:
            wm = self.wm50 if state.get("legacy50") else self.wm100
            self._opt[key] = OptProportion(self.density(state, cap, tail), wm)
        return self._opt[key]

    def detector(self, state, cap="state", tail="state", params="state"):
        cfg = self.config
        density = self.density(state, cap, tail)
        if state["kind"] == "legacy":
            return Detector(density, state["mode"], rhos=cfg["rho_grid"], fixed_e=cfg["fixed_e"],
                            fixed_rho=cfg["fixed_rho"], window=cfg["window"])
        proportion = self.opt(state, cap, tail)
        if state["kind"] == "li":
            return Detector(density, "opt_hmm_li", rhos=cfg["rho_grid"], proportion=proportion)
        params = state["params"] if params == "state" else params
        return Detector(density, state["mode"], params=params, proportion=proportion)

    def infer(self, state, docs, **kw):
        return self.detector(state, **kw).infer_many([a["y"] for a in docs])

    # ------------------------------------------------------------------ transition estimation
    def fit_transition(self, state, cap, tail):
        """EM on mixed_cal for one (method, density); returns the fit summary and the data it used."""
        docs = self.docs["mixed_cal"]
        if len({len(a["y"]) for a in docs}) != 1:
            raise ValueError("mixed_cal articles must share one length.")
        density = self.density(state, cap, tail)
        g = np.stack([density(a["y"]) for a in docs])
        eps = None
        if state["mode"] in ("pd", "p1"):
            ctx = self.opt(state, cap, tail)
            eps = np.array([ctx.estimate(row).value for row in g])
        fitted, surface = em_module.fit(g, state["mode"], eps, grid_check=True,
                                        max_iter=self.l1["em"]["max_iter"], tol=self.l1["em"]["tol"])
        return {"fit": fitted, "g": g, "eps": eps, "surface": surface}

    def uncertainty(self, state, result):
        """Bootstrap intervals and the a00 = a11 likelihood-ratio test at the final fit."""
        e = self.l1["em"]
        fitted, g, eps = result["fit"], result["g"], result["eps"]
        constrained = em_module.constrained_fit(g, state["mode"], eps, tol=e["tol"])
        return {"bootstrap": em_module.bootstrap(g, state["mode"], fitted, eps, e["bootstrap_draws"],
                                                 seed=self.l1["global_seed"], tol=e["tol"]),
                "equal_retention": {**constrained, **em_module.equal_retention_test(fitted, constrained)}}

    @staticmethod
    def params_from(fit):
        return {"a00": fit["a00"], "a11": fit["a11"], "pi1": fit["pi1"] if fit["pi1"] is not None else .5}

    # ------------------------------------------------------------------ cap selection (dev only)
    def evaluate_candidate(self, state, cap, tail, params):
        """tau on null_cal and mean per-scenario IoU on the three dev localization scenarios."""
        detector = self.detector(state, cap=cap, tail=tail, params=params)
        null = detector.infer_many([a["y"] for a in self.docs["null_cal"]])
        rules = build_rules([(a["n_eff"], d.scores) for a, d in zip(self.docs["null_cal"], null)], self.config)
        dev = [a for a in self.docs["dev"] if a["scenario"] in LOCALIZATION_SCENARIOS]
        results = detector.infer_many([a["y"] for a in dev])
        by_scenario, flagged = defaultdict(list), 0
        for a, d in zip(dev, results):
            labels, _ = decide(d.scores, a["n_eff"], rules)
            flagged += int(labels.sum())
            iou = position_metrics(a["truth"], labels)["iou"]
            if iou is not None:
                by_scenario[a["scenario"]].append(iou)
        tau = rules["M"]["pooled"].value
        iou = float(np.mean([np.mean(v) for v in by_scenario.values()])) if by_scenario else 0.
        return {"cap": cap, "tail": tail, "tau": tau, "flagged_positions": flagged, "iou": iou,
                "ok": bool(tau < 1 and flagged > 0)}

    def select_cap(self, state, params):
        grid = self.l1["cap_grid"]["param" if state["family"] in PARAM_FAMILIES else state["family"]]
        table = [self.evaluate_candidate(state, cap, None, params) for cap in grid]
        best = self._best(table)
        tail_table = []
        if best is not None and state["family"] == "ecdf400" and best["cap"] == max(k for k in grid if k is not None):
            base = self.bases["ecdf400"]
            for y0 in self.l1["tail_y0"]:
                tail = {"y0": y0, "beta": tail_beta(base, y0)}
                tail_table += [self.evaluate_candidate(state, cap, tail, params) for cap in grid]
            best = self._best(table + tail_table, prefer_plain=True)
        return {"cap": None if best is None else best["cap"], "tail": None if best is None else best["tail"],
                "iou": None if best is None else best["iou"], "fallback": best is None,
                "table": table, "tail_table": tail_table}

    @staticmethod
    def _best(table, prefer_plain=False):
        ok = [r for r in table if r["ok"]]
        if not ok:
            return None
        # highest IoU; ties: no tail, then less truncation (larger cap; None = no cap)
        return max(ok, key=lambda r: (round(r["iou"], 12), r["tail"] is None, cap_key(r["cap"])))

    # ------------------------------------------------------------------ one method, end to end
    def fit_method(self, name):
        spec = parse_method(name)
        k0 = None if spec["nocap"] else self.l1["cap_k0"]
        state = {**spec, "legacy50": self.stage == "s1" and spec["kind"] == "legacy", "tail": None, "cap": k0}
        start = perf_counter()
        if spec["kind"] == "general":
            result = self.fit_transition(state, k0, None)
            state["em_k0"] = result["fit"]
            state["params"] = self.params_from(result["fit"])
            if self.stage in ("s3", "s3b", "s4") and not spec["nocap"]:
                choice = self.select_cap(state, state["params"])
                state["selection"] = choice
                if choice["fallback"]:
                    choice["note"] = "no candidate had tau < 1 and flagged positions; kept K0"
                else:
                    state["cap"], state["tail"] = choice["cap"], choice["tail"]
                    if (state["cap"], state["tail"]) != (k0, None):
                        result = self.fit_transition(state, state["cap"], state["tail"])
                        state["params"] = self.params_from(result["fit"])
            state["em"] = result["fit"]
            candidate = spec["family"] in PARAM_FAMILIES and not spec["nocap"]
            if not spec["nocap"] and not candidate and self.l1["em"]["bootstrap_draws"] > 0:
                state.update(self.uncertainty(state, result))
            state["surface"] = result["surface"]
            state["train"] = {"g": result["g"], "eps": result["eps"]}
        detector = self.detector(state)
        null = detector.infer_many([a["y"] for a in self.docs["null_cal"]])
        state["rules"] = build_rules([(a["n_eff"], d.scores) for a, d in zip(self.docs["null_cal"], null)], self.config)
        state["null_cal_max"] = [float(d.scores.max()) for d in null]
        state["null_cal_detections"] = null
        state["fit_seconds"] = perf_counter() - start
        self.states[name] = state
        return state

    # ------------------------------------------------------------------ param* (dev only)
    def choose_param_star(self):
        """Pick the parametric family with the best mean PD/PE dev IoU (plan 3.1); bootstrap only the winner."""
        names = {f: (f"pd_{f}", f"pe_{f}") for f in PARAM_FAMILIES}
        if not all(n in self.states for pair in names.values() for n in pair):
            return None
        table = []
        for family in PARAM_FAMILIES:
            picks = [self.states[n]["selection"] for n in names[family]]
            eligible = not any(p["fallback"] for p in picks)
            table.append({"family": family, "eligible": eligible,
                          "pd_iou": picks[0]["iou"], "pe_iou": picks[1]["iou"],
                          "pd_cap": picks[0]["cap"], "pe_cap": picks[1]["cap"],
                          "score": float(np.mean([p["iou"] for p in picks])) if eligible else None})
        ok = [r for r in table if r["eligible"]]
        if ok:
            top = max(round(r["score"], 12) for r in ok)
            family = next(r["family"] for r in ok if round(r["score"], 12) == top)    # ties: candidate order
            fallback = False
        else:
            family, fallback = "beta", True
        self.param_star = {"family": family, "fallback": fallback, "table": table}
        if self.l1["em"]["bootstrap_draws"] > 0:
            for name in names[family]:
                state = self.states[name]
                result = {"fit": state["em"], "g": state["train"]["g"], "eps": state["train"]["eps"]}
                state.update(self.uncertainty(state, result))
        return self.param_star

    # ------------------------------------------------------------------ seedbs
    def seedbs_kw(self):
        s = self.l1["seedbs"]
        return {"T": s["T"], "Tp": s["Tp"], "a": s["a"], "min_factor": s["min_factor"]}

    def precompute_seedbs(self, docs, B):
        jobs = [(a["y"], B, sb.article_seed(self.l1["global_seed"], a["id"]), self.seedbs_kw()) for a in docs]
        if self.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                return list(pool.map(_seedbs_worker, jobs, chunksize=max(1, len(jobs) // (4 * self.workers))))
        return [_seedbs_worker(j) for j in jobs]

    def select_seedbs(self, reuse=None):
        if reuse:
            saved = read_json(Path(project_path(reuse)) / "seedbs_params.json")
            state = {"name": SEEDBS, "kind": "seedbs", "B": saved["B"], "zeta": saved["zeta"],
                     "rules": rules_from_dict(saved["rules"]), "table": saved["table"], "reused_from": str(reuse)}
            self.states[SEEDBS] = state
            return state
        s = self.l1["seedbs"]
        null_docs = self.docs["null_cal"]
        dev = [a for a in self.docs["dev"] if a["scenario"] in LOCALIZATION_SCENARIOS]
        table, best, null_pres = [], None, {}
        for B in s["B"]:
            progress("seedbs 调参", len(table), len(s["B"]) * len(s["zeta"]), f"B={B} 预计算")
            started = perf_counter()
            pre_null, pre_dev = self.precompute_seedbs(null_docs, B), self.precompute_seedbs(dev, B)
            null_pres[B] = pre_null
            seconds = (perf_counter() - started) / (len(null_docs) + len(dev))
            for zeta in s["zeta"]:
                cuts_null = [sb.seedbs_select(p, zeta) for p in pre_null]
                scores = [sb.segment_scores(a["y"], c) for a, c in zip(null_docs, cuts_null)]
                rules = build_rules([(a["n_eff"], x) for a, x in zip(null_docs, scores)], self.config)
                tau = rules["M"]["pooled"].value
                by_scenario, flagged = defaultdict(list), 0
                for a, p in zip(dev, pre_dev):
                    x = sb.segment_scores(a["y"], sb.seedbs_select(p, zeta))
                    labels, _ = decide(x, a["n_eff"], rules)
                    flagged += int(labels.sum())
                    iou = position_metrics(a["truth"], labels)["iou"]
                    if iou is not None:
                        by_scenario[a["scenario"]].append(iou)
                iou = float(np.mean([np.mean(v) for v in by_scenario.values()])) if by_scenario else 0.
                row = {"B": B, "zeta": zeta, "tau": tau, "c": 1 - tau, "flagged_positions": flagged, "iou": iou,
                       "ok": bool(tau < 1 and flagged > 0), "seconds_per_article": seconds,
                       "false_changepoints_per_null": float(np.mean([len(c) for c in cuts_null]))}
                table.append(row)
                if row["ok"]:
                    key = (round(iou, 12), B, -zeta)            # ties: larger B, smaller zeta
                    if best is None or key > best[0]:
                        best = (key, row, rules)
        if best is None:
            raise RuntimeError("No seedbs setting has tau < 1 and flagged positions on dev.")
        _, row, rules = best
        state = {"name": SEEDBS, "kind": "seedbs", "B": row["B"], "zeta": row["zeta"], "rules": rules, "table": table,
                 "null_pre": null_pres[row["B"]]}
        self.states[SEEDBS] = state
        return state

    def infer_seedbs(self, docs, state=None, pres=None):
        state = state or self.states[SEEDBS]
        pres = pres or self.precompute_seedbs(docs, state["B"])
        out = []
        for a, p in zip(docs, pres):
            cuts = sb.seedbs_select(p, state["zeta"])
            out.append(Detection(sb.segment_scores(a["y"], cuts), 0., 0., True, p["seconds"],
                                 info={"changepoints": cuts}))
        return out


# ---------------------------------------------------------------------- scoring
def score_method(run, name, docs, detections=None):
    """Rows for one method on `docs` (M-rule labels), with Rand index and change-point recovery added."""
    state = run.states[name]
    if detections is None:
        detections = run.infer_seedbs(docs) if state["kind"] == "seedbs" else run.infer(state, docs)
    rows = score(docs, {("m", name): detections}, {("m", name): state["rules"]}, run.m)
    for row, a, d in zip(rows, docs, detections):
        labels, _ = decide(d.scores, a["n_eff"], state["rules"])
        truth_cp = changepoints_from_labels(a["truth"])
        pred_cp = d.info["changepoints"] if state["kind"] == "seedbs" else changepoints_from_labels(labels)
        row.update({"method": name, "emission": state.get("family", "seedbs"),
                    "rand_index": rand_index(truth_cp, pred_cp, a["n_eff"]),
                    **changepoint_recovery(truth_cp, pred_cp), "labels_flagged": int(labels.sum())})
        if state["kind"] == "seedbs":
            row.update({"eps_hat": None, "eps_error": None, "rho": None})
        else:
            row["pi1"] = d.info.get("pi1")
            for key in ("alpha_bar", "pi_clipped", "pi_unidentifiable", "marginal_gap"):
                if key in d.info:
                    row[key] = d.info[key]
    return rows


def summarize(rows, null_rows, config, scenarios=ALL_SCENARIOS, seed=42):
    index, nulls = defaultdict(list), defaultdict(list)
    for r in rows:
        index[(r["scenario"], r["method"])].append(r)
    for r in null_rows or []:
        nulls[r["method"]].append(r)
    for bucket in list(index.values()) + list(nulls.values()):
        bucket.sort(key=lambda r: r["article_id"])
    summary = {}
    for scenario in scenarios:
        keys = [k for k in index if k[0] == scenario]
        if not keys:
            continue
        positive = [r for r in index[keys[0]] if r["signal_count"] > 0]
        indices = bootstrap_indices(max(len(positive), 1), config["bootstrap"], seed)
        summary[scenario] = {}
        for _, method in keys:
            c = cell(index[(scenario, method)], nulls.get(method) if null_rows is not None else None, indices)
            sel = index[(scenario, method)]
            c["rand_index"] = _mean([r["rand_index"] for r in sel])
            c["changepoints"] = {"true": _mean([r["true_changepoints"] for r in sel]),
                                 "predicted": _mean([r["predicted_changepoints"] for r in sel]),
                                 "nearest_distance": _mean([r["nearest_distance"] for r in sel])}
            summary[scenario][method] = c
    return summary


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def per_scenario_iou(rows, method, scenarios=LOCALIZATION_SCENARIOS):
    by = defaultdict(list)
    for r in sorted((r for r in rows if r["method"] == method and r["scenario"] in scenarios), key=lambda r: r["article_id"]):
        if r["signal_count"] > 0:
            by[r["scenario"]].append(r["iou"])
    return {s: np.array(v) for s, v in by.items()}


class Endpoint:
    """Main endpoint = mean over localization scenarios of the scenario-mean IoU, stratified paired bootstrap."""

    def __init__(self, rows, methods, draws, seed=7):
        self.per = {m: per_scenario_iou(rows, m) for m in methods}
        self.scenarios = [s for s in LOCALIZATION_SCENARIOS if all(s in self.per[m] for m in methods)]
        self.indices = {s: bootstrap_indices(len(self.per[methods[0]][s]), draws, seed + i)
                        for i, s in enumerate(self.scenarios)}

    def mean(self, m):
        return float(np.mean([self.per[m][s].mean() for s in self.scenarios]))

    def by_scenario(self, m):
        return {s: float(self.per[m][s].mean()) for s in self.scenarios}

    def delta(self, a, b):
        diffs = [self.per[a][s] - self.per[b][s] for s in self.scenarios]
        boot = np.mean([d[self.indices[s]].mean(axis=1) for d, s in zip(diffs, self.scenarios)], axis=0)
        return {"mean": float(np.mean([d.mean() for d in diffs])), "ci95": np.quantile(boot, [.025, .975]).tolist()}


def recalibrated_false_positive(null_scores, alpha, draws=200, seed=0):
    """R1: split the pooled null max-scores in halves, calibrate on one, count false alarms on the other."""
    scores = np.asarray(null_scores, float)
    rng, half, rates = np.random.default_rng(seed), len(scores) // 2, []
    for _ in range(draws):
        order = rng.permutation(len(scores))
        tau = calibrate(scores[order[:half]], alpha).value
        rates.append(float((scores[order[half:]] > tau).mean()))
    return {"median": float(np.median(rates)), "p95": float(np.quantile(rates, .95)), "draws": draws}


# ---------------------------------------------------------------------- freezing
def freeze(run):
    methods = {}
    for name, s in run.states.items():
        entry = {"kind": s["kind"], "rules": rules_to_dict(s["rules"])}
        if s["kind"] == "seedbs":
            entry.update({**run.l1["seedbs"], "B_grid": run.l1["seedbs"]["B"], "zeta_grid": run.l1["seedbs"]["zeta"],
                          "B": s["B"], "zeta": s["zeta"], "c": 1 - s["rules"]["M"]["pooled"].value,
                          "seed": run.l1["global_seed"]})
        else:
            entry.update({"mode": s["mode"], "family": s["family"], "legacy50": bool(s.get("legacy50")),
                          "cap": s["cap"], "tail": s["tail"], "params": s.get("params")})
        methods[name] = entry
    emissions = {f: b.to_dict() for f, b in run.bases.items()}
    emissions["hist25_calibration50"] = run.legacy50.to_dict()
    return {"stage": run.stage, "param_star": run.param_star, "config": {k: run.config[k] for k in ("tokens", "context_width", "key_id", "roles",
                                                                       "alpha", "fraction_tau", "bootstrap", "l1")},
            "emissions": emissions, "methods": methods}


def thaw(run, frozen):
    """Rebuild every method state from a frozen.json dict (the only source used after the gate)."""
    run.param_star = frozen.get("param_star")
    for name, entry in frozen["methods"].items():
        rules = rules_from_dict(entry["rules"])
        if entry["kind"] == "seedbs":
            run.states[name] = {"name": name, "kind": "seedbs", "B": entry["B"], "zeta": entry["zeta"], "rules": rules}
            continue
        run.states[name] = {**parse_method(name), "legacy50": entry["legacy50"], "cap": entry["cap"],
                            "tail": entry["tail"], "params": entry["params"], "rules": rules}


# ---------------------------------------------------------------------- driver
def run_stage(positions_dir, output, stage, override=None, reuse_seedbs=None, workers=1, methods=None, param_star=None):
    if stage not in STAGE_METHODS:
        raise ValueError(f"stage must be one of {tuple(STAGE_METHODS)}.")
    started = perf_counter()
    directory = Path(project_path(positions_dir))
    manifest = read_json(directory / "manifest.json")
    config = dict(manifest["config"])
    config.update(override or {})
    validate_l0(config)
    validate_l1(config)
    output = new_output(project_path(output))
    progress("加载并校验 positions")
    articles = [a for a in load_articles(directory, manifest["key"], config["context_width"])
                if a["split"] in STAGE_ROLES[stage]]
    run = Run(config, articles, stage, workers)
    names = tuple(methods or STAGE_METHODS[stage])
    if stage == "s3b" and param_star and not methods:      # nocap of the family chosen at s3 (plan 3.5)
        if param_star not in PARAM_FAMILIES:
            raise ValueError(f"param_star must be one of {PARAM_FAMILIES}.")
        names += (f"pe_{param_star}", f"pe_{param_star}_nocap")
    for number, name in enumerate(names, 1):
        progress("拟合方法", number, len(names), name)
        if name == SEEDBS:
            run.select_seedbs(reuse_seedbs)
        else:
            run.fit_method(name)
    run.choose_param_star()
    frozen = freeze(run)
    write_json(output / "frozen.json", frozen)
    if stage == "s4":                                     # gate: only the reloaded objects are used from here on
        frozen = json.loads((output / "frozen.json").read_text(encoding="utf-8"))
        saved_null = {n: (s["null_cal_max"], s["null_cal_detections"]) for n, s in run.states.items() if "null_cal_max" in s}
        extras = {n: {k: s[k] for k in ("em", "em_k0", "bootstrap", "equal_retention", "selection", "surface", "train", "fit_seconds")
                      if k in s} for n, s in run.states.items()}
        table = {n: s.get("table") for n, s in run.states.items() if s["kind"] == "seedbs"}
        thaw(run, frozen)
        for n, e in extras.items():
            run.states[n].update(e)
        for n, (mx, det) in saved_null.items():
            run.states[n]["null_cal_max"], run.states[n]["null_cal_detections"] = mx, det
        for n, t in table.items():
            run.states[n]["table"] = t
    rows, null_rows = [], []
    eval_docs = [a for role in STAGE_EVAL[stage] for a in run.docs[role]]
    null_role = "null_test" if stage == "s4" else "null_cal"
    for number, name in enumerate(names, 1):
        progress("评价", number, len(names), name)
        rows += score_method(run, name, eval_docs)
        if stage == "s4":
            null_rows += score_method(run, name, run.docs[null_role])
        else:
            state = run.states[name]
            if name == SEEDBS:
                detections = run.infer_seedbs(run.docs[null_role], pres=state.get("null_pre"))
            else:
                detections = state["null_cal_detections"]
            null_rows += score_method(run, name, run.docs[null_role], detections)
    from .report_v2 import build_report, render
    report = build_report(run, rows, null_rows, names, manifest, time_total=perf_counter() - started)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render(report), encoding="utf-8")
    slim = lambda rs: [{k: v for k, v in r.items() if k != "boundary_errors"} for r in rs]
    write_jsonl(output / "metrics.jsonl", slim(rows) + slim(null_rows))
    surfaces = {n: np.round(s["surface"], 4).tolist() for n, s in run.states.items() if s.get("surface") is not None}
    if surfaces:
        write_json(output / "em_surfaces.json", {"grid": em_module.GRID.tolist(), "log_likelihood": surfaces})
    if SEEDBS in run.states:
        s = run.states[SEEDBS]
        write_json(output / "seedbs_params.json", {"B": s["B"], "zeta": s["zeta"], "rules": rules_to_dict(s["rules"]),
                                                   "table": s["table"]})
    return {"report": str(output / "report.md"), "stage": stage, "seconds": report["timings"]["total_seconds"]}
