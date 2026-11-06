"""Optional flushed progress events for the notebook subprocess reader, plus the overall experiment bar."""
import json
import os
import sys
import time
from collections import deque

PREFIX = "@@OPT_HMM_PROGRESS "
TOTAL_PREFIX = "@@OPT_HMM_TOTAL "
WINDOW = 50


def progress(stage, current=None, total=None, detail=""):
    if os.environ.get("OPT_HMM_PROGRESS") == "1":
        event = {"stage": stage, "current": current, "total": total, "detail": detail}
        print(PREFIX + json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)


def clock_text(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Overall:
    """Counts finished units over a fixed stage table and reports `done/total` plus a weighted percentage.

    stages: ordered {name: {"total": int, "unit": str}}. weights: {name: seconds per unit} or None
    (then every unit weighs the same and the bar says so). Cached units are counted as done but never timed.
    """

    def __init__(self, stages, path=None, weights=None, stream=None, clock=time.monotonic, elapsed_offset=0., interval=.5):
        self.stages = {k: {"total": int(v["total"]), "unit": v.get("unit", "")} for k, v in stages.items()}
        self.weights = {k: float(weights[k]) for k in self.stages if weights and weights.get(k)} if weights else {}
        self.calibrated = bool(weights) and all(k in self.weights for k in self.stages)
        self.done = dict.fromkeys(self.stages, 0)
        self.cached = dict.fromkeys(self.stages, 0)
        self.seconds = dict.fromkeys(self.stages, 0.)
        self.recent = {k: deque(maxlen=WINDOW) for k in self.stages}
        self.path, self.stream, self.clock = path, stream if stream is not None else sys.stderr, clock
        self.started, self.last, self.offset = clock(), clock(), float(elapsed_offset)
        self.current = None
        self.events = 0
        self.interval, self.last_emit = interval, None

    def restore(self, path):
        """Continue the counts of an earlier command (same stage table); the clock keeps adding up."""
        try:
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            return False
        if any(saved["stages"].get(k, {}).get("total") != v["total"] for k, v in self.stages.items()):
            return False
        for k in self.stages:
            self.done[k] = saved["stages"][k]["done"]
            self.cached[k] = saved["stages"][k].get("cached", 0)
            self.seconds[k] = saved["stages"][k].get("seconds", 0.)
        self.offset = float(saved.get("elapsed_seconds", 0.))
        return True

    def reset(self, *names):
        """Start these stages over (a rerun of the command that owns them)."""
        for k in names:
            self.done[k], self.cached[k], self.seconds[k] = 0, 0, 0.
            self.recent[k].clear()

    def weight(self, name):
        return self.weights.get(name, 1.) if self.calibrated else 1.

    def advance(self, stage, n=1, cached=False):
        if stage not in self.stages:
            raise KeyError(f"Unknown progress stage {stage!r}.")
        now = self.clock()
        n = int(n)
        room = self.stages[stage]["total"] - self.done[stage]
        n = max(0, min(n, room))
        self.done[stage] += n
        if cached:
            self.cached[stage] += n
        elif n:
            dt = now - self.last
            self.seconds[stage] += dt
            self.recent[stage].extend([dt / n] * min(n, WINDOW))
        self.last = now
        if n:
            self.current = stage
            self.events += 1
            if (self.last_emit is None or now - self.last_emit >= self.interval
                    or self.done[stage] == self.stages[stage]["total"]):
                self.last_emit = now
                self.emit()

    def percent(self):
        total = sum(s["total"] * self.weight(k) for k, s in self.stages.items())
        done = sum(self.done[k] * self.weight(k) for k in self.stages)
        return 100. * done / total if total else 100.

    def eta(self):
        """[low, high] seconds, or None when the later stages cannot be priced."""
        stage = self.current
        if stage is None or not self.recent[stage]:
            return None
        rate = sum(self.recent[stage]) / len(self.recent[stage])
        remaining = rate * (self.stages[stage]["total"] - self.done[stage])
        if self.calibrated:
            later = list(self.stages)[list(self.stages).index(stage) + 1:]
            remaining += sum((self.stages[k]["total"] - self.done[k]) * self.weights[k] for k in later)
        return [.7 * remaining, 1.5 * remaining]

    def snapshot(self):
        return {"percent": self.percent(), "calibrated": self.calibrated, "current": self.current,
                "elapsed_seconds": self.offset + self.clock() - self.started, "eta_seconds": self.eta(),
                "stages": {k: {"done": self.done[k], "total": s["total"], "unit": s["unit"], "cached": self.cached[k],
                               "weight": self.weights.get(k), "seconds": self.seconds[k]}
                           for k, s in self.stages.items()}}

    def line(self):
        snap = self.snapshot()
        parts = [f"总进度 {snap['percent']:.1f}%"]
        if self.current:
            parts.append(f"{self.current} {self.done[self.current]}/{self.stages[self.current]['total']}")
        parts.append(f"已用 {clock_text(snap['elapsed_seconds'])}")
        if snap["eta_seconds"]:
            parts.append(f"剩余 ~{clock_text(snap['eta_seconds'][0])}–{clock_text(snap['eta_seconds'][1])}")
        if not self.calibrated:
            parts.append("权重未校准")
        return "[" + " | ".join(parts) + "]"

    def emit(self):
        snap = self.snapshot()
        if self.path:
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(snap, handle, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        if self.stream:
            print("\r" + self.line(), end="", file=self.stream, flush=True)
        if os.environ.get("OPT_HMM_PROGRESS") == "1":
            print(TOTAL_PREFIX + json.dumps(snap, ensure_ascii=False), file=sys.stderr, flush=True)

    def finish(self):
        self.emit()
        if self.stream:
            print(file=self.stream, flush=True)

    def usage(self):
        """Actual units and seconds per stage: the starting weights of the next run."""
        return {k: {"units": self.done[k], "cached": self.cached[k], "seconds": self.seconds[k]}
                for k in self.stages}


_OVERALL = None


def set_overall(overall):
    global _OVERALL
    _OVERALL = overall


def get_overall():
    return _OVERALL


def tick(stage, n=1, cached=False):
    """Advance the overall bar when one is installed; a no-op otherwise."""
    if _OVERALL is not None:
        _OVERALL.advance(stage, n, cached)
