"""Reject L2 settings that break the plan before any expensive step."""
from ..config import validate_text
from .plan import SIDES, assign_recipes

KINDS = ("pure", "resample", "paper", "alt", "null")
TRAIN_ROLES = ("cal_pure", "mix_r2", "mix_splice", "mix_alt", "mix_r0", "dev", "null_cal")
TEST_SPLITS = ("test_paper", "test_alt", "test_r2", "test_r0", "null_test")
ARMS = ("m100", "m400", "fe")


def validate_recipe(recipe, where):
    kind = recipe.get("kind")
    if kind not in KINDS or not recipe.get("name"):
        raise ValueError(f"{where}: recipe needs a name and a kind in {KINDS}.")
    if kind == "resample" and (not 0 < recipe.get("rate", 0) < 1 or recipe.get("min_entropy", -1) < 0):
        raise ValueError(f"{where}: resample needs rate in (0,1) and min_entropy >= 0.")
    if kind == "paper" and recipe.get("setting") not in (1, 2, 3, 4):
        raise ValueError(f"{where}: paper setting must be 1..4.")
    if kind == "alt" and not (recipe.get("L") == "top" or (type(recipe.get("L")) is int and recipe["L"] >= 2)):
        raise ValueError(f"{where}: alt needs L >= 2 or 'top'.")


def validate_l2(config):
    validate_text({**config, "edits": []})
    l2 = config.get("l2")
    if not l2 or type(l2.get("global_seed")) is not int:
        raise ValueError("Config needs an l2 section with an integer global_seed.")
    if config["tokens"] < 64:
        raise ValueError("tokens too small for the L2 settings.")
    sides = l2.get("sides", {})
    if set(sides) != set(SIDES):
        raise ValueError(f"l2.sides must define exactly {SIDES}.")
    keys = []
    for name, side in sides.items():
        key = side.get("key", "")
        if not key or len(key.encode("utf-8")) > 64 or not side.get("key_id"):
            raise ValueError(f"l2.sides.{name}: key (1..64 bytes) and key_id are required.")
        keys.append((key, side["key_id"]))
    if len({k for k, _ in keys}) != len(keys) or len({i for _, i in keys}) != len(keys):
        raise ValueError("The train and test sides must use different keys and key_ids.")
    roles = l2.get("roles", {})
    listed = [r for side in sides.values() for r in side.get("roles", [])]
    if sorted(listed) != sorted(roles) or len(set(listed)) != len(listed):
        raise ValueError("Every L2 role must belong to exactly one side.")
    splits = {s: set() for s in SIDES}
    for side_name, side in sides.items():
        for role_name in side["roles"]:
            role = roles[role_name]
            if type(role.get("prompts")) is not int or role["prompts"] < 1 or not role.get("recipes"):
                raise ValueError(f"l2.roles.{role_name}: prompts >= 1 and recipes are required.")
            names = [r.get("name") for r in role["recipes"]]
            if len(set(names)) != len(names):
                raise ValueError(f"l2.roles.{role_name}: duplicate recipe names.")
            for recipe in role["recipes"]:
                validate_recipe(recipe, f"l2.roles.{role_name}")
                splits[side_name].add(recipe.get("split", role_name))
            counted = [("count" in r) for r in role["recipes"]]
            if any(counted) and not all(counted):
                raise ValueError(f"l2.roles.{role_name}: give every recipe a count, or none.")
            assign_recipes(role)
    if splits["train"] & splits["test"]:
        raise ValueError("Train and test sides share article roles.")
    alts = [r for role in roles.values() for r in role["recipes"] if r["kind"] == "alt" and r["L"] == "top"]
    if alts and not l2.get("l_top_from", {}).get("role") in roles:
        raise ValueError("alt 'top' recipes need l2.l_top_from = {role, recipe} on the train side.")
    if alts:
        ref = l2["l_top_from"]
        if ref["role"] not in sides["train"]["roles"] or ref["recipe"] not in [r["name"] for r in roles[ref["role"]]["recipes"]]:
            raise ValueError("l2.l_top_from must name a train-side role and one of its resample recipes.")
    grid = l2.get("cap_grid", [])
    if not grid or grid[-1] is not None or any(k is None or k <= 1 for k in grid[:-1]) or grid != sorted(
            grid[:-1]) + [None]:
        raise ValueError("l2.cap_grid: increasing caps above 1, ending with null (no cap).")
    sweep = l2.get("cap_sweep", [])
    if not sweep or sweep[-1] is not None or any(k is None or k <= 1 for k in sweep[:-1]):
        raise ValueError("l2.cap_sweep: caps above 1, ending with null.")
    if l2.get("cap_k0", 0) <= 1:
        raise ValueError("l2.cap_k0 must exceed 1.")
    h = l2.get("h_grid", [])
    if not h or h[-1] is not None or any(v is None or v <= 0 for v in h[:-1]):
        raise ValueError("l2.h_grid: positive bandwidths, ending with null (infinite bandwidth).")
    kernel = l2.get("kernel_grid", [])
    if len(kernel) < 2 or kernel != sorted(kernel) or kernel[0] <= 0 or kernel[-1] > 1:
        raise ValueError("l2.kernel_grid must be an increasing grid in (0,1].")
    arms = l2.get("arms", {})
    if set(arms) != set(ARMS):
        raise ValueError(f"l2.arms must define exactly {ARMS}.")
    pure = roles.get("cal_pure", {}).get("prompts", 0)
    for name, arm in arms.items():
        if not 1 <= arm.get("pure_articles", 0) <= pure:
            raise ValueError(f"l2.arms.{name}.pure_articles must lie in 1..{pure}.")
    e = l2.get("em", {})
    if e.get("max_iter", 0) < 1 or e.get("tol", 0) <= 0 or type(e.get("bootstrap_draws")) is not int:
        raise ValueError("l2.em needs max_iter, tol and bootstrap_draws.")
    s = l2.get("seedbs", {})
    if not s.get("B") or any(b % 2 or b < 2 for b in s["B"]) or not s.get("zeta") or s.get("paper_B") not in s["B"]:
        raise ValueError("l2.seedbs needs even B values including paper_B, and a zeta grid.")
    if s.get("T", 0) < 1 or s.get("Tp", 0) < 1 or not 0 < s.get("a", 0) < 1 or s.get("min_factor", 0) <= 0:
        raise ValueError("l2.seedbs needs T, Tp, a in (0,1) and min_factor.")
    if config.get("alpha", 0) <= 0 or config.get("bootstrap", 0) < 1 or config.get("bins") != 200:
        raise ValueError("alpha, bootstrap and bins = 200 (hist200 only) are required.")
