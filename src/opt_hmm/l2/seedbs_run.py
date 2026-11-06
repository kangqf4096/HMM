"""SeedBS-NOT for L2 (plan 4.3): disk-cached precompute, dev selection, and the paper-parameter runs."""
import hashlib
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..detector import Detection
from ..l0 import seedbs as sb
from ..l0.decide import build_rules, decide
from ..l0.metrics import position_metrics
from ..progress import progress, tick
from .methods import DEV_CLASSES

SEEDBS = "seedbs"
PAPER_PREFIX = "seedbs_paper_z"


def paper_name(zeta):
    return f"{PAPER_PREFIX}{zeta:g}"


def seedbs_names(config):
    return (SEEDBS,) + tuple(paper_name(z) for z in config["l2"]["seedbs"]["zeta"])


def _digest(*parts):
    return hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"), digest_size=10).hexdigest()


def _job(args):
    y, B, seed, kw, path = args
    if path and Path(path).is_file():
        with open(path, "rb") as handle:
            return pickle.load(handle), True
    pre = sb.seedbs_precompute(y, B, seed, **kw)
    if path:
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as handle:
            pickle.dump(pre, handle)
        Path(tmp).replace(path)
    return pre, False


class SeedBSRunner:
    def __init__(self, config, cache_dir=None, workers=1):
        self.config, self.s = config, config["l2"]["seedbs"]
        self.seed = config["l2"]["global_seed"]
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.kw = {"T": self.s["T"], "Tp": self.s["Tp"], "a": self.s["a"], "min_factor": self.s["min_factor"]}

    def _path(self, doc, B):
        if not self.cache:
            return None
        key = _digest(doc["id"], B, self.s["T"], self.s["Tp"], self.s["a"], self.s["min_factor"], self.seed,
                      hashlib.blake2b(np.ascontiguousarray(doc["y"]).tobytes(), digest_size=8).hexdigest())
        return str(self.cache / f"B{B}_{key}.pkl")

    def precompute(self, docs, B, stage=None):
        """pre-dicts for `docs`; ticks `stage` once per article, marking cache hits as cached."""
        jobs = [(d["y"], B, sb.article_seed(self.seed, d["id"]), self.kw, self._path(d, B)) for d in docs]
        out = []
        if self.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                for pre, hit in pool.map(_job, jobs, chunksize=max(1, len(jobs) // (8 * self.workers))):
                    out.append(pre)
                    if stage:
                        tick(stage, 1, cached=hit)
        else:
            for job in jobs:
                pre, hit = _job(job)
                out.append(pre)
                if stage:
                    tick(stage, 1, cached=hit)
        return out

    def detections(self, docs, pres, zeta):
        out = []
        for doc, pre in zip(docs, pres):
            cuts = sb.seedbs_select(pre, zeta)
            out.append(Detection(sb.segment_scores(doc["y"], cuts), 0., 0., True, pre["seconds"],
                                 info={"changepoints": cuts}))
        return out

    # ------------------------------------------------------------------ fit stage
    def select(self, null_stack, dev_stack):
        """Dev selection over (B, zeta); also the null-calibrated rules of the four paper-parameter runs."""
        cfg = self.config
        table, best, states = [], None, {}
        total = len(self.s["B"]) * len(self.s["zeta"])
        for B in self.s["B"]:
            progress("seedbs 调参", len(table), total, f"B={B} 预计算")
            pre_null = self.precompute(null_stack.docs, B)
            pre_dev = self.precompute(dev_stack.docs, B)
            for zeta in self.s["zeta"]:
                cuts_null = [sb.seedbs_select(p, zeta) for p in pre_null]
                scores = [sb.segment_scores(d["y"], c) for d, c in zip(null_stack.docs, cuts_null)]
                rules = build_rules([(null_stack.n_eff, x) for x in scores], cfg)
                tau = rules["M"]["pooled"].value
                by_class, flagged = defaultdict(list), 0
                for doc, pre in zip(dev_stack.docs, pre_dev):
                    x = sb.segment_scores(doc["y"], sb.seedbs_select(pre, zeta))
                    labels, _ = decide(x, dev_stack.n_eff, rules)
                    flagged += int(labels.sum())
                    iou = position_metrics(doc["truth"], labels)["iou"]
                    if iou is not None and doc["cls"] in DEV_CLASSES:
                        by_class[doc["cls"]].append(iou)
                classes = {c: float(np.mean(v)) for c, v in by_class.items()}
                iou = float(np.mean([classes[c] for c in DEV_CLASSES])) if len(classes) == len(DEV_CLASSES) else 0.
                row = {"B": B, "zeta": zeta, "tau": tau, "c": 1 - tau, "iou": iou, "iou_by_class": classes,
                       "flagged_positions": flagged, "ok": bool(tau < 1 and flagged > 0),
                       "false_changepoints_per_null": float(np.mean([len(c) for c in cuts_null]))}
                table.append(row)
                tick("S2")
                if row["ok"]:
                    key = (round(iou, 12), B, -zeta)            # ties: larger B, smaller zeta
                    if best is None or key > best[0]:
                        best = (key, row, rules)
                if B == self.s["paper_B"]:
                    states[paper_name(zeta)] = {"name": paper_name(zeta), "B": B, "zeta": zeta, "rules": rules}
        if best is None:
            raise RuntimeError("No seedbs setting has tau < 1 and flagged positions on dev.")
        _, row, rules = best
        states[SEEDBS] = {"name": SEEDBS, "B": row["B"], "zeta": row["zeta"], "rules": rules}
        for name in states:
            tick("T1")
        return states, table
