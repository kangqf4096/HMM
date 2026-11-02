"""Reject L0 settings that break the frozen protocol before any expensive step."""
from ..config import validate_text
from .schema import TEST_SCENARIOS

ROLE_NAMES = ("calibration", "dev", "test", "null_cal", "null_test")
EMISSIONS = ("histogram", "beta", "beta_mixture", "histogram_nocap", "beta_nocap")
DECISIONS = ("max_score", "fraction")


def validate_l0(config):
    validate_text({**config, "edits": []})
    roles = config.get("roles", {})
    for name in ROLE_NAMES:
        if type(roles.get(name)) is not int or roles[name] < 1:
            raise ValueError(f"roles.{name} must be a positive integer.")
    if roles["test"] % len(TEST_SCENARIOS):
        raise ValueError("roles.test must be a multiple of 4 so every scenario gets the same number of articles.")
    rates = set()
    for edit in config.get("edits", []):
        if edit.get("kind") != "resample" or not 0 < edit.get("rate", 0) < 1 or edit.get("min_entropy", -1) < 0:
            raise ValueError("L0 edits must be resample with rate in (0,1) and min_entropy >= 0.")
        rates.add(f"resample_{edit['rate']:g}")
    if rates != {s for s in TEST_SCENARIOS if s.startswith("resample_")}:
        raise ValueError("edits must define exactly the resample_0.1 and resample_0.3 scenarios.")
    if not config.get("emission") or not set(config["emission"]) <= set(EMISSIONS):
        raise ValueError(f"emission must be a nonempty subset of {EMISSIONS}.")
    if set(config.get("decision", [])) != set(DECISIONS):
        raise ValueError(f"decision must be exactly {DECISIONS}.")
    if config.get("bins", 0) < 1 or config.get("pseudocount", 0) <= 0:
        raise ValueError("bins >= 1 and pseudocount > 0 are required.")
    if not 0 < config.get("alpha", 0) < 1:
        raise ValueError("alpha must lie in (0,1).")
    if config.get("cap") is not None and config["cap"] <= 1:
        raise ValueError("cap must exceed 1 (the null density) or be null.")
    if not config.get("rho_grid") or any(not 0 <= r < 1 for r in config["rho_grid"]):
        raise ValueError("rho_grid must be nonempty with values in [0,1).")
    if not 0 <= config.get("fixed_e", -1) <= 1 or not 0 <= config.get("fixed_rho", -1) < 1:
        raise ValueError("Invalid fixed_e / fixed_rho.")
    if config.get("window", 0) < 1 or config.get("bootstrap", 0) < 1:
        raise ValueError("window and bootstrap must be positive.")
    if config.get("merge_gap") != 0:
        raise ValueError("Only strict adjacency (merge_gap = 0) is implemented.")
    tau = config.get("fraction_tau", 0.5)
    if tau != "same" and not 0 < tau < 1:
        raise ValueError('fraction_tau must be "same" or a number in (0,1).')
    if config.get("min_group_size", 100) < 1:
        raise ValueError("min_group_size must be positive.")
