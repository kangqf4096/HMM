"""Group B of the length study: f1 stays frozen from another text length; only what the data length changes is refitted.

Kept from the frozen source (nothing is selected again): the pure-watermark hist200 (stage 1), the e'-kernel grid of
`fe`, the cap and the bandwidth of every method. Refitted on the new length's training side: the transition
parameters (a00, a11, pi1) by the same EM as the fit stage, and tau on `null_cal`. SeedBS, the key and L_top come
from the new length's own fit (group A), so the same test stage can run: `l2-test --frozen <output>`.
"""
from time import perf_counter

from ..io import new_output, provenance, write_json
from ..l0.decide import rules_to_dict
from ..paths import project_path
from .config import validate_l2
from .data import load_side
from .emission import KernelGrid
from .evaluate import load_frozen
from .fit import check_side
from .methods import HistMethod, Stage1, Trainer, method_names, params_from

SAME_CONFIG = ("context_width", "bins", "pseudocount", "alpha", "fraction_tau", "min_group_size")
SAME_L2 = ("arms", "kernel_grid", "cap_grid", "h_grid", "cap_k0", "em")
EM_KEYS = ("iterations", "multimodal", "start_spread", "at_bound", "grid_gap", "lambda", "stationary", "log_likelihood")


def check_transferable(source, config):
    """The frozen f1 can only be reused under the same estimation settings (the text length aside)."""
    for key in SAME_CONFIG:
        if source["config"].get(key) != config.get(key):
            raise ValueError(f"Frozen source and new configuration differ in {key}.")
    for key in SAME_L2:
        if source["l2"].get(key) != config["l2"].get(key):
            raise ValueError(f"Frozen source and new configuration differ in l2.{key}.")


def tau_of(entry):
    return entry["rules"]["M"]["pooled"]["value"]


def run_reuse(frozen_path, train_dir, base_dir, output):
    started = perf_counter()
    source, base = load_frozen(frozen_path), load_frozen(base_dir)
    side = load_side(train_dir)
    config = side["manifest"]["config"]
    validate_l2(config)
    check_side(side, config)
    check_transferable(source, config)
    if base["config"]["tokens"] != config["tokens"] or base["train_key_id"] != side["manifest"]["key_id"]:
        raise ValueError("The base fit belongs to another text length or key than this training side.")
    if set(source["methods"]) != set(method_names()):
        raise ValueError("The frozen source does not carry the six L2 methods.")
    trainer = Trainer(config, side)
    stage1 = {k: Stage1.from_dict(v) for k, v in source["stage1"].items()}
    trainer.stage1 = {int(k): v for k, v in stage1.items()}
    trainer.arm_stage1 = {arm: stage1[str(spec["pure_articles"])] for arm, spec in config["l2"]["arms"].items()}
    methods, detail = {}, {}
    for name in method_names():
        entry = source["methods"][name]
        kernel = None if entry["kernel"] is None else KernelGrid.from_dict(entry["kernel"])
        method = HistMethod(name, trainer.arm_stage1[entry["arm"]], entry["cap"], kernel, entry["h"])
        result = trainer.fit_transition(method)
        method.params = params_from(result["fit"])
        method.rules, _ = trainer.null_rules(method)
        methods[name] = method
        detail[name] = {"cap": method.cap, "h": method.h, "params": method.params,
                        "tau": method.rules["M"]["pooled"].value,
                        "params_group_a": base["methods"][name]["params"], "tau_group_a": tau_of(base["methods"][name]),
                        "params_f1_source": entry["params"], "tau_f1_source": tau_of(entry),
                        "em": {k: result["fit"].get(k) for k in EM_KEYS}}
    frozen = {**base, "stage1": source["stage1"],
              "methods": {n: {**m.to_dict(), "rules": rules_to_dict(m.rules)} for n, m in methods.items()},
              "reuse": {"f1_from": str(project_path(frozen_path)), "f1_tokens": source["config"]["tokens"],
                        "base_fit": str(project_path(base_dir)), "tokens": config["tokens"]}}
    out = new_output(project_path(output))
    write_json(out / "frozen.json", frozen)
    write_json(out / "reuse_report.json", {"methods": detail, "seconds": perf_counter() - started,
                                           "provenance": provenance(), "reuse": frozen["reuse"]})
    return {"frozen": str(out / "frozen.json"), "seconds": perf_counter() - started}
