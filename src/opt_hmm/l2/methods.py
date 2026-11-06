"""HMM methods of L2: {pd, pe} x {m100, m400, fe}. Fitting (EM, selection on dev) and frozen inference."""
from collections import defaultdict

import numpy as np

from ..detector import Detection
from ..hmm import forward_backward_two_state
from ..l0 import em as em_module
from ..l0.decide import build_rules, decide
from ..l0.metrics import position_metrics
from .emission import HistOpt, KernelGrid, capped, hist_density, ratios

MODES = ("pd", "pe")
HMM_ARMS = ("m100", "m400", "fe")
DEV_CLASSES = ("paper", "alt", "r2")
TRAIN_A_SPLITS = ("mix_r2", "mix_splice", "mix_alt")


def cap_key(cap):
    return float("inf") if cap is None else float(cap)


def method_names():
    return tuple(f"{mode}_{arm}" for arm in HMM_ARMS for mode in MODES)


def parse(name):
    mode, arm = name.split("_", 1)
    if mode not in MODES or arm not in HMM_ARMS:
        raise ValueError(f"Unknown method {name!r}.")
    return mode, arm


class Stage1:
    """Pure-watermark hist200 and the bin counts of the pivots that fitted it."""

    def __init__(self, counts, pseudocount=.5):
        self.counts = np.asarray(counts, np.int64)
        self.density = hist_density(self.counts, pseudocount)
        self._opt = {}

    def opt(self, cap):
        key = cap_key(cap)
        if key not in self._opt:
            self._opt[key] = HistOpt(self.density, self.counts, cap)
        return self._opt[key]

    def to_dict(self):
        return {"counts": self.counts.tolist(), "density": self.density.tolist()}

    @classmethod
    def from_dict(cls, data):
        out = cls(data["counts"])
        if not np.allclose(out.density, data["density"]):
            raise ValueError("Frozen stage-1 density does not match its counts.")
        return out


class HistMethod:
    """One frozen (or being fitted) method: arm, mode, cap, bandwidth, transition parameters."""

    def __init__(self, name, stage1, cap, kernel=None, h=None, params=None, rules=None):
        self.name, (self.mode, self.arm) = name, parse(name)
        self.stage1, self.cap, self.kernel, self.h = stage1, cap, kernel, h
        self.params, self.rules = params, rules
        if (self.arm == "fe") != (kernel is not None):
            raise ValueError("Only the fe arm carries a kernel grid.")

    def proportions(self, counts):
        return self.stage1.opt(self.cap).estimate_many(counts)

    def ratios(self, index, counts):
        """Per-article density ratios, e', identifiability: the emission each article is read with."""
        value, ident, _ = self.proportions(counts)
        if self.kernel is None:
            g = ratios(index, self.stage1.density, self.cap)
        else:
            g = ratios(index, self.kernel.densities(value), self.cap)
        return g, value, ident

    def scores(self, stack, params=None):
        """Posterior P(z_t = 1) for every article of the stack, with e' and the identifiable flag."""
        params = params or self.params
        g, eps, ident = self.ratios(stack.index, stack.counts)
        pi1 = eps if self.mode == "pd" else np.full(len(eps), params["pi1"])
        gamma = forward_backward_two_state(g, np.clip(pi1, 0., 1.), params["a00"], params["a11"], want_gamma=True).gamma
        gamma = np.where(ident[:, None], gamma, 0.)
        return gamma, eps, ident, pi1

    def detections(self, stack, params=None):
        gamma, eps, ident, pi1 = self.scores(stack, params)
        p = params or self.params
        lam = p["a00"] + p["a11"] - 1
        return [Detection(gamma[i], float(eps[i]), lam, bool(ident[i]), 0., info={"pi1": float(pi1[i])})
                for i in range(len(stack))]

    def to_dict(self):
        return {"name": self.name, "arm": self.arm, "mode": self.mode, "cap": self.cap, "h": self.h,
                "params": self.params, "kernel": None if self.kernel is None else self.kernel.to_dict()}


def params_from(fit):
    return {"a00": fit["a00"], "a11": fit["a11"], "pi1": fit["pi1"] if fit["pi1"] is not None else .5}


class Trainer:
    """Everything the fit stage needs from the training side: stacked roles, stage-1 densities, e' caches."""

    def __init__(self, config, side):
        self.config, self.l2 = config, config["l2"]
        self.splits = side["splits"]
        self.alpha = config["alpha"]
        pure = self.splits["cal_pure"]
        self.stage1 = {}
        for arm, spec in self.l2["arms"].items():
            k = spec["pure_articles"]
            self.stage1.setdefault(k, Stage1(pure.counts[:k].sum(axis=0), config["pseudocount"]))
        self.arm_stage1 = {arm: self.stage1[spec["pure_articles"]] for arm, spec in self.l2["arms"].items()}
        self._kernel = {}
        self._mix = None

    # ---------------------------------------------------------------- e' and the kernel grid
    def kernel(self, arm, cap, h):
        """f1(.; e) fitted on cal_pure and mix_r2 truth=1 positions, weighted by each article's e'."""
        key = (arm, cap_key(cap), h)
        if key not in self._kernel:
            stage1 = self.arm_stage1[arm]
            pure, r2 = self.splits["cal_pure"], self.splits["mix_r2"]
            n = self.l2["arms"][arm]["pure_articles"]
            counts = np.vstack([pure.counts[:n], r2.truth_counts()])
            eps = np.concatenate([stage1.opt(cap).estimate_many(pure.counts[:n])[0],
                                  stage1.opt(cap).estimate_many(r2.counts)[0]])
            self._kernel[key] = KernelGrid.fit(counts, eps, self.l2["kernel_grid"], h, self.config["pseudocount"])
        return self._kernel[key]

    def method(self, name, cap, h=None, params=None, rules=None):
        mode, arm = parse(name)
        kernel = self.kernel(arm, cap, h) if arm == "fe" else None
        return HistMethod(name, self.arm_stage1[arm], cap, kernel, h, params, rules)

    # ---------------------------------------------------------------- EM on the A-training split
    def mix_stack(self):
        if self._mix is None:
            from .data import Stack
            self._mix = Stack([d for s in TRAIN_A_SPLITS for d in self.splits[s].docs])
        return self._mix

    def fit_transition(self, method, grid_check=True):
        stack = self.mix_stack()
        g, eps, _ = method.ratios(stack.index, stack.counts)
        em = self.l2["em"]
        fitted, surface = em_module.fit(g, method.mode, eps if method.mode == "pd" else None, grid_check=grid_check,
                                        max_iter=em["max_iter"], tol=em["tol"])
        return {"fit": fitted, "g": g, "eps": eps, "surface": surface}

    def uncertainty(self, method, result):
        em = self.l2["em"]
        fitted, g, eps = result["fit"], result["g"], result["eps"]
        eps = eps if method.mode == "pd" else None
        constrained = em_module.constrained_fit(g, method.mode, eps, tol=em["tol"])
        out = {"equal_retention": {**constrained, **em_module.equal_retention_test(fitted, constrained)}}
        if em["bootstrap_draws"] > 0:
            out["bootstrap"] = em_module.bootstrap(g, method.mode, fitted, eps, em["bootstrap_draws"],
                                                   seed=self.l2["global_seed"], tol=em["tol"])
        return out

    # ---------------------------------------------------------------- dev candidates
    def null_rules(self, method):
        null = self.splits["null_cal"]
        det = method.detections(null)
        return build_rules([(null.n_eff, d.scores) for d in det], self.config), det

    def dev_scores(self, labels_by_doc, stack):
        by_class = defaultdict(list)
        flagged = 0
        for doc, labels in zip(stack.docs, labels_by_doc):
            flagged += int(labels.sum())
            iou = position_metrics(doc["truth"], labels)["iou"]
            if iou is not None and doc["cls"] in DEV_CLASSES:
                by_class[doc["cls"]].append(iou)
        classes = {c: float(np.mean(v)) for c, v in by_class.items()}
        value = float(np.mean([classes[c] for c in DEV_CLASSES])) if len(classes) == len(DEV_CLASSES) else 0.
        return classes, value, flagged

    def evaluate_candidate(self, method):
        """tau on null_cal, then the equal-weight mean of the class IoUs on dev."""
        rules, _ = self.null_rules(method)
        dev = self.splits["dev"]
        gamma = method.scores(dev)[0]
        labels = [decide(row, dev.n_eff, rules)[0] for row in gamma]
        classes, value, flagged = self.dev_scores(labels, dev)
        tau = rules["M"]["pooled"].value
        return {"cap": method.cap, "h": method.h, "tau": tau, "iou": value, "iou_by_class": classes,
                "flagged_positions": flagged, "ok": bool(tau < 1 and flagged > 0)}

    @staticmethod
    def best(table):
        ok = [r for r in table if r["ok"]]
        if not ok:
            return None
        # highest IoU; ties: less truncation (larger cap, None largest), then larger h (None = infinite)
        return max(ok, key=lambda r: (round(r["iou"], 12), cap_key(r["cap"]), cap_key(r["h"])))
