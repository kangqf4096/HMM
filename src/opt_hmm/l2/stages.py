"""Stage table of the overall progress bar (plan 8.1), recomputed from the configuration."""
import sys

from .methods import HMM_ARMS, MODES
from .plan import assign_recipes, recipe_sources

STAGE_ORDER = ("G1", "B1", "D1", "F1", "E1", "S1", "S2", "T1", "G2", "B2", "R1", "BS", "C1", "REP")
UNITS = {"G1": "生成请求(训练侧)", "B1": "文章构建(训练侧)", "D1": "机制对照", "F1": "f1 拟合", "E1": "EM 运行",
         "S1": "dev 候选评价", "S2": "seedbs 选择", "T1": "τ/c 校准", "G2": "生成请求(评价侧)",
         "B2": "文章构建(评价侧)", "R1": "(方法,文章)推理", "BS": "A 的 bootstrap 重抽样", "C1": "cap 细化",
         "REP": "报告"}


def role_counts(config, side):
    """(generation requests, articles) of the roles on one side."""
    l2 = config["l2"]
    requests = articles = 0
    for name in l2["sides"][side]["roles"]:
        for recipes in assign_recipes(l2["roles"][name]):
            requests += len({s for r in recipes for s in recipe_sources(r)})
            articles += len(recipes)
    return requests, articles


def stage_table(config):
    l2 = config["l2"]
    g1, b1 = role_counts(config, "train")
    g2, b2 = role_counts(config, "test")
    n_caps, n_h, draws = len(l2["cap_grid"]), len(l2["h_grid"]), l2["em"]["bootstrap_draws"]
    methods = len(HMM_ARMS) * len(MODES)
    n_zeta, n_B = len(l2["seedbs"]["zeta"]), len(l2["seedbs"]["B"])
    seedbs_methods = 1 + n_zeta
    m_arms = len(HMM_ARMS) - 1
    k0_fits = len(MODES) * (m_arms + n_h)
    counts = {
        "G1": g1, "B1": b1, "D1": 1,
        "F1": len({a["pure_articles"] for a in l2["arms"].values()}),
        "E1": k0_fits + 2 * methods,                       # K0 fits, final fits, equal-retention fits
        "S1": len(MODES) * n_caps * (m_arms + n_h),
        "S2": n_B * n_zeta + n_zeta,
        "T1": methods + seedbs_methods,
        "G2": g2, "B2": b2,
        "R1": (methods + seedbs_methods) * b2,
        "BS": draws * methods,
        "C1": len(MODES) * len(l2["cap_sweep"]),
        "REP": 1,
    }
    return {k: {"total": counts[k], "unit": UNITS[k]} for k in STAGE_ORDER}


def compare_to_plan(config, stream=None):
    """Log the stages whose unit count differs from the plan's table (config l2.plan_units)."""
    expected = config["l2"].get("plan_units")
    if not expected:
        return {}
    table = stage_table(config)
    diff = {k: {"plan": expected[k], "program": table[k]["total"]} for k in table
            if k in expected and expected[k] != table[k]["total"]}
    for k, v in diff.items():
        print(f"进度表差异 {k}: 计划 {v['plan']}，程序 {v['program']}（以程序为准）", file=stream or sys.stderr)
    return diff


def install_overall(config, path, owns, weights_path=None, stream=None):
    """Create the overall bar for one command: continue the counts saved at `path`, restart the stages it owns."""
    import json
    from pathlib import Path

    from ..progress import Overall, set_overall
    compare_to_plan(config)
    weights = None
    weights_path = weights_path or config["l2"].get("progress", {}).get("weights")
    if weights_path and Path(weights_path).is_file():
        weights = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    overall = Overall(stage_table(config), path=str(path) if path else None, weights=weights, stream=stream)
    if path:
        overall.restore(str(path))
    overall.reset(*owns)
    set_overall(overall)
    return overall


def default_progress_path(output):
    """One shared progress file beside the stage output directories, so every command adds to the same bar."""
    from pathlib import Path
    return Path(output).parent / "l2_progress.json"


def weights_from_progress(progress_path, output):
    """progress_weights.json: seconds per timed unit of every stage that has run (cached units are not timed)."""
    import json
    from pathlib import Path
    saved = json.loads(Path(progress_path).read_text(encoding="utf-8"))
    weights = {}
    for name, stage in saved["stages"].items():
        timed = stage["done"] - stage.get("cached", 0)
        if timed > 0 and stage["seconds"] > 0:
            weights[name] = stage["seconds"] / timed
    Path(output).write_text(json.dumps(weights, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"weights": weights, "missing": [k for k in saved["stages"] if k not in weights]}
