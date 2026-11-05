"""Reject L0 settings that break the frozen protocol before any expensive step."""
from ..config import validate_text
from .schema import SEG_SCENARIOS, TEST_SCENARIOS

ROLE_NAMES = ("calibration", "dev", "test", "null_cal", "null_test")
NEW_ROLE_NAMES = ("calibration2", "mixed_cal", "dev2", "test_seg")
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
    new_roles = [name for name in NEW_ROLE_NAMES if name in roles]
    if new_roles:
        for name in NEW_ROLE_NAMES:
            if type(roles.get(name)) is not int or roles[name] < 1:
                raise ValueError(f"roles.{name} must be a positive integer when any of {NEW_ROLE_NAMES} is set.")
        for name in ("dev2", "test_seg"):
            if roles[name] % len(SEG_SCENARIOS):
                raise ValueError(f"roles.{name} must be a multiple of 3 so seg2/seg3/seg4 get equal counts.")
        validate_segment_ranges(config)
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


def validate_segment_ranges(config):
    ranges = config.get("segment_ranges", {})
    if set(ranges) != set(SEG_SCENARIOS):
        raise ValueError(f"segment_ranges must define exactly {SEG_SCENARIOS}.")
    n = config["tokens"]
    for name, bounds_list in ranges.items():
        previous = 0
        for bounds in bounds_list:
            if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= 1:
                raise ValueError(f"segment_ranges.{name}: intervals must lie in [0,1].")
            start, stop = round(bounds[0] * n), round(bounds[1] * n)
            if start < previous or start >= stop:
                raise ValueError(f"segment_ranges.{name}: intervals must be nonempty, ordered and disjoint in tokens.")
            previous = stop + 1


def validate_l1(config):
    """Settings of the L1 evaluation (plan_general_transition); required for l0-evaluate --stage."""
    l1 = config.get("l1")
    if not l1:
        raise ValueError("Config has no l1 section.")
    for name in NEW_ROLE_NAMES:
        if name not in config.get("roles", {}):
            raise ValueError(f"L1 needs roles.{name}.")
    if type(l1.get("global_seed")) is not int or not l1.get("cap_k0", 0) > 1:
        raise ValueError("l1.global_seed must be an int and l1.cap_k0 > 1.")
    grid = l1.get("cap_grid", {})
    if set(grid) != {"hist25", "hist200", "ecdf400", "param"}:
        raise ValueError("l1.cap_grid needs hist25, hist200, ecdf400 and param.")
    for family, caps in grid.items():
        finite = [k for k in caps if k is not None]
        if not caps or any(k <= 1 for k in finite):
            raise ValueError(f"l1.cap_grid.{family}: caps must exceed the null density 1.")
        if family in ("ecdf400", "param"):
            if family == "param" and None in caps:
                raise ValueError("l1.cap_grid.param: parametric densities may be unbounded, so no null (no cap) entry.")
        elif caps[-1] is not None or None in caps[:-1]:
            raise ValueError(f"l1.cap_grid.{family}: a histogram grid ends with null (no cap), once.")
    if not l1.get("tail_y0") or any(not 0 < y < 1 for y in l1["tail_y0"]):
        raise ValueError("l1.tail_y0 must be values in (0,1).")
    e = l1.get("em", {})
    if e.get("max_iter", 0) < 1 or e.get("tol", 0) <= 0 or type(e.get("bootstrap_draws")) is not int or e["bootstrap_draws"] < 0:
        raise ValueError("l1.em needs max_iter, tol and bootstrap_draws (0 disables the bootstrap).")
    s = l1.get("seedbs", {})
    if not s.get("B") or any(b % 2 or b < 2 for b in s["B"]) or not s.get("zeta"):
        raise ValueError("l1.seedbs needs even B values and a zeta grid.")
    if s.get("T", 0) < 1 or s.get("Tp", 0) < 1 or not 0 < s.get("a", 0) < 1 or s.get("min_factor", 0) <= 0:
        raise ValueError("l1.seedbs needs T, Tp, a in (0,1) and min_factor.")
