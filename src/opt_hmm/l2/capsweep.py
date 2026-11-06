"""cap sensitivity of m100 (plan 5.3): refit A and recalibrate tau at every cap, then read the test curve.

The curve is a sensitivity analysis. The cap frozen for m100 was chosen on dev and is not changed by anything here.
"""
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ..io import new_output, write_json
from ..l0.decide import decide
from ..l0.metrics import boundary_summary, position_metrics, segment_report
from ..paths import project_path
from ..progress import progress, tick
from .data import load_side
from .evaluate import SATURATION, TEST_SPLITS, load_frozen
from .fit import check_side
from .methods import MODES, Trainer, params_from
from .report import jsonable, num, table
from .stages import default_progress_path, install_overall

TERTILES = ((0., 1. / 3), (1. / 3, 2. / 3), (2. / 3, 1. + 1e-9))


def evaluate_cap(trainer, method, test_splits):
    rules, _ = trainer.null_rules(method)
    method.rules = rules
    per_scenario, tertile = defaultdict(list), defaultdict(list)
    saturation, boundary = defaultdict(list), defaultdict(list)
    for split in TEST_SPLITS:
        stack = test_splits[split]
        gamma, eps, ident, _ = method.scores(stack)
        for doc, row, e in zip(stack.docs, gamma, eps):
            labels, _ = decide(row, stack.n_eff, rules)
            pm = position_metrics(doc["truth"], labels)
            key = doc["scenario"]
            saturation[key].append(float((row > SATURATION).mean()))
            boundary[key].append(segment_report(doc["truth"], labels)["boundary_errors"])
            if pm["iou"] is not None:
                per_scenario[key].append(pm["iou"])
                tertile[next(i for i, (lo, hi) in enumerate(TERTILES) if lo <= e < hi)].append(pm["iou"])
    null = test_splits["null_test"]
    gamma = method.scores(null)[0]
    flags = [decide(row, null.n_eff, rules)[1]["M"] for row in gamma]
    return {"tau": rules["M"]["pooled"].value, "any_fp": float(np.mean(flags)),
            "iou": {k: float(np.mean(v)) for k, v in per_scenario.items()},
            "saturation": {k: float(np.mean(v)) for k, v in saturation.items()},
            "boundary": {k: boundary_summary(v)["mean"] for k, v in boundary.items()},
            "iou_by_eps": {f"{lo:.2f}-{min(hi, 1.):.2f}": (float(np.mean(tertile[i])) if tertile[i] else None, len(tertile[i]))
                           for i, (lo, hi) in enumerate(TERTILES)}}


def run_cap_sweep(frozen_path, train_dir, test_dir, output, progress_path=None, weights=None):
    started = perf_counter()
    frozen = load_frozen(frozen_path)
    train, test = load_side(train_dir), load_side(test_dir)
    config = train["manifest"]["config"]
    check_side(train, config)
    if test["manifest"].get("side") != "test":
        raise ValueError("Second side must be the test build.")
    output = new_output(project_path(output))
    overall = install_overall(config, progress_path or default_progress_path(output), ("C1",), weights)
    trainer = Trainer(config, train)
    caps = config["l2"]["cap_sweep"]
    rows = []
    for mode in MODES:
        name = f"{mode}_m100"
        for number, cap in enumerate(caps, 1):
            progress("cap 细化", number, len(caps), f"{name} cap={cap}")
            method = trainer.method(name, cap)
            result = trainer.fit_transition(method, grid_check=False)
            method.params = params_from(result["fit"])
            rows.append({"method": name, "cap": cap, "a00": method.params["a00"], "a11": method.params["a11"],
                         **evaluate_cap(trainer, method, test["splits"])})
            tick("C1")
    frozen_caps = {n: frozen["methods"][n]["cap"] for n in (f"{m}_m100" for m in MODES)}
    report = jsonable({"rows": rows, "frozen_cap": frozen_caps, "seconds": perf_counter() - started,
                       "note": "敏感性曲线；冻结的 cap 由 dev 选定，曲线上的 test 最优点不用来修改它。"})
    write_json(Path(output) / "cap_sweep.json", report)
    (Path(output) / "cap_sweep.md").write_text(render(report), encoding="utf-8")
    overall.finish()
    return {"report": str(Path(output) / "cap_sweep.md")}


def render(r):
    scenarios = sorted({s for row in r["rows"] for s in row["iou"]})
    out = ["# m100 的 cap 细化（敏感性，不重新选）", "", r["note"], "",
           "冻结的 cap：" + "，".join(f"{n}: {c if c is not None else '不截断'}" for n, c in r["frozen_cap"].items()), ""]
    for name in sorted({row["method"] for row in r["rows"]}):
        sel = [row for row in r["rows"] if row["method"] == name]
        out += [f"## {name}：按场景", "",
                table(["cap", "τ", "any-FP", "a00", "a11"] + scenarios,
                      [[row["cap"] if row["cap"] is not None else "不截断", num(row["tau"], 4), num(row["any_fp"]),
                        num(row["a00"], 3), num(row["a11"], 3)] + [num(row["iou"].get(s)) for s in scenarios] for row in sel]), "",
                f"## {name}：按 e' 三档的 IoU（括号为文章数）", "",
                table(["cap"] + list(sel[0]["iou_by_eps"]),
                      [[row["cap"] if row["cap"] is not None else "不截断"] +
                       [f"{num(v[0])} ({v[1]})" for v in row["iou_by_eps"].values()] for row in sel]), "",
                f"## {name}：后验饱和比例与边界误差", "",
                table(["cap"] + [f"饱和 {s}" for s in scenarios[:4]] + [f"边界 {s}" for s in scenarios[:4]],
                      [[row["cap"] if row["cap"] is not None else "不截断"] + [num(row["saturation"].get(s)) for s in scenarios[:4]]
                       + [num(row["boundary"].get(s), 2) for s in scenarios[:4]] for row in sel]), ""]
    return "\n".join(out) + "\n"
