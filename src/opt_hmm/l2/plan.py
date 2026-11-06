"""L2 roles, recipes and article composition (plan_l2_paper_setting). No GPU, no model."""
import numpy as np

from ..io import read_jsonl
from ..paths import project_path

SIDES = ("train", "test")
SOURCES = ("W", "U")            # W: watermarked generation, U: unwatermarked generation, same prompt
PAPER_CUTS = {                  # paper Setting 2-4 as fractions of n: pieces (source, start, stop) in final coordinates
    2: (("W", 0, .5), ("U", .5, 1)),
    3: (("W", 0, .4), ("U", .4, .6), ("W", .6, 1)),
    4: (("W", 0, .2), ("U", .2, .4), ("W", .4, .6), ("U", .6, .8), ("W", .6, .8)),
}
# In Setting 4 the last piece is the watermark text of source positions [.6n, .8n) moved to [.8n, n).


def recipe_sources(recipe):
    kind = recipe["kind"]
    if kind == "null":
        return ("U",)
    if kind in ("pure", "resample") or (kind == "paper" and recipe["setting"] == 1):
        return ("W",)
    return SOURCES


def recipe_class(recipe):
    """Evaluation block of a recipe: pure, null, set1, paper (settings 2-4), alt, r2 or r0."""
    kind = recipe["kind"]
    if kind == "paper":
        return "set1" if recipe["setting"] == 1 else "paper"
    if kind == "resample":
        return "r0" if recipe["min_entropy"] == 0 else "r2"
    return kind


def assign_recipes(role):
    """List (one entry per prompt of the role) of the recipes applied to that prompt."""
    recipes, n = role["recipes"], role["prompts"]
    if not any("count" in r for r in recipes):
        return [list(recipes) for _ in range(n)]
    out = []
    for recipe in recipes:
        out += [[recipe]] * recipe["count"]
    if len(out) != n:
        raise ValueError(f"Recipe counts add up to {len(out)}, role has {n} prompts.")
    return out


def article_split(role_name, recipe):
    return recipe.get("split", role_name)


def plan_jobs(prompts, config):
    """One generation job per (prompt, needed source); order follows prompts.jsonl."""
    roles, n = config["l2"]["roles"], config["tokens"]
    seen, jobs = {}, []
    for doc in prompts:
        role = doc["role"]
        index = seen.get(role, 0)
        seen[role] = index + 1
        recipes = assign_recipes(roles[role])[index]
        for source in SOURCES:
            if any(source in recipe_sources(r) for r in recipes):
                jobs.append({"doc": doc, "scenario": f"src_{source}", "ranges": [[0, n]] if source == "W" else [],
                             "article_id": f"{doc['group']}#{source}", "extra": {"source": source, "group": doc["group"]}})
    return jobs


def pieces_for(recipe, n, l_top=None):
    """Pieces [(source, start, stop)] of the final article; `start:stop` are the final coordinates,
    and the source coordinates equal them except for the moved last piece of Setting 4."""
    kind = recipe["kind"]
    if kind in ("pure", "resample"):
        return [("W", 0, n, 0)]
    if kind == "null":
        return [("U", 0, n, 0)]
    if kind == "paper":
        if recipe["setting"] == 1:
            return [("W", 0, n, 0)]
        cuts = PAPER_CUTS[recipe["setting"]]
        out, moved = [], recipe["setting"] == 4
        for i, (source, a, b) in enumerate(cuts):
            fa, fb = round(a * n), round(b * n)
            if moved and i == len(cuts) - 1:
                out.append((source, round(.8 * n), n, round(.6 * n)))     # (source, final start, final stop, source start)
            else:
                out.append((source, fa, fb, fa))
        return out
    if kind == "alt":
        length = l_top if recipe["L"] == "top" else int(recipe["L"])
        if length is None:
            raise ValueError("alt_top needs L_top; build the training side first.")
        starts = range(0, n, length)
        return [("W" if i % 2 == 0 else "U", s, min(n, s + length), s) for i, s in enumerate(starts)]
    raise ValueError(f"No pieces for recipe kind {kind!r}.")


def compose(recipe, sources, n, l_top=None):
    """Final tokens, W ranges (final coords), edit positions (every spliced-in non-watermark position)
    and the paper-style change points (token coords, no seed-breaking gap)."""
    tokens, ranges, edits, kinds = [], [], [], []
    for source, a, b, s in pieces_for(recipe, n, l_top):
        text = sources[source]["token_ids"]
        if len(text) != n or s + (b - a) > n:
            raise ValueError("Source shorter than the composition needs.")
        tokens += [int(t) for t in text[s:s + (b - a)]]
        kinds.append((source, a, b))
        if source == "W":
            ranges.append([a, b])
        elif recipe["kind"] != "null":
            edits += [{"t": t, "kind": "splice"} for t in range(a, b)]
    cps = [a for (s0, _, _), (s1, a, _) in zip(kinds[:-1], kinds[1:]) if s0 != s1]
    return tokens, ranges, edits, cps


def paper_changepoints(cps, context_width, n):
    """Token-coordinate boundaries -> pivot coordinates (the first `context_width` tokens have no pivot)."""
    return [c - context_width for c in cps if 0 < c - context_width < n - context_width]


def l_top_from(truth_by_article, n):
    """Alternating-segment length matching the mean truth change-point count of the reference articles."""
    from ..l0.metrics import changepoints_from_labels
    counts = [len(changepoints_from_labels(t)) for t in truth_by_article]
    mean = float(np.mean(counts))
    length = max(2, round(n / (mean + 1.)))
    return {"mean_changepoints": mean, "L_top": int(length), "K_top": int(-(-n // length) - 1),
            "articles": len(counts)}


def exclude_groups(config):
    """Groups of earlier prompts files that L2 must not reuse (every role of those runs)."""
    groups = set()
    for path in config["l2"].get("exclude_prompts", []):
        path = project_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"exclude_prompts file is missing: {path}")
        groups.update(r["group"] for r in read_jsonl(path))
    return groups


def select_l2(raw, tokenizer, config):
    """Prompts of every L2 role, drawn from rows no earlier run used; roles are disjoint."""
    from ..data import collect_candidates
    roles = config["l2"]["roles"]
    required = sum(r["prompts"] for r in roles.values())
    exclude = exclude_groups(config)
    candidates, total, examined, rng = collect_candidates(raw, tokenizer, config, required, frozenset(exclude))
    rng.shuffle(candidates)
    records, cursor = [], 0
    for name, role in roles.items():
        for article in candidates[cursor:cursor + role["prompts"]]:
            article["role"] = name
            article.pop("pool_ids")
            records.append(article)
        cursor += role["prompts"]
    return records, {"source_articles": total, "examined_articles": examined, "selected_articles": required,
                     "excluded_groups": len(exclude)}
