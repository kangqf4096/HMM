"""Turn L2 sources into positions.jsonl: compose W/U per recipe, resample edits, pivots, truth = S minus B."""
from pathlib import Path
from time import perf_counter

import numpy as np

from ..io import new_output, provenance, read_json, read_jsonl, write_json, write_jsonl
from ..l0.build import EDIT_SEED_TAG
from ..l0.resample import resample
from ..l0.schema import build_article, validate_article
from ..paths import project_path
from ..progress import progress, tick
from .config import validate_l2
from .stages import default_progress_path, install_overall
from .plan import article_split, assign_recipes, compose, l_top_from, paper_changepoints

ENTROPY_DECIMALS = 4


def entropy_matrix(probs):
    p = np.asarray(probs, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(p), 0.).sum(axis=1)


def group_sources(sources):
    """Prompts in generation order: [{role, group, prompt_ids, W, U}]."""
    order, by_group = [], {}
    for s in sources:
        if s["group"] not in by_group:
            by_group[s["group"]] = {"role": s["split"], "group": s["group"], "prompt_ids": s["prompt_ids"],
                                    "source_split": s.get("source_split"), "source_row": s.get("source_row")}
            order.append(s["group"])
        by_group[s["group"]][s["source"]] = s
    return [by_group[g] for g in order]


def make_article(recipe, role_name, prompt, key, m, n, decode, rng=None, probs=None, entropy=None, l_top=None):
    kind = recipe["kind"]
    stats = None
    if kind == "resample":
        tokens, edits, stats = resample(prompt["W"]["token_ids"], probs, recipe["rate"], recipe["min_entropy"], rng)
        ranges, cps = [[0, n]], []
    else:
        tokens, ranges, edits, cps = compose(recipe, prompt, n, l_top)
    source = {"article_id": f"{prompt['group']}:{recipe['name']}", "split": article_split(role_name, recipe),
              "scenario": recipe["name"], "ranges": ranges}
    rows, article = build_article(source, tokens, edits, key, m, decode)
    article.update({"recipe": {k: v for k, v in recipe.items() if k != "count"}, "group": prompt["group"],
                    "prompt_role": role_name, "paper_changepoints": paper_changepoints(cps, m, n),
                    "source_row": prompt.get("source_row"), "source_split": prompt.get("source_split"),
                    "edit_stats": stats})
    if entropy is not None and kind in ("pure", "resample"):
        article["entropy"] = [round(float(h), ENTROPY_DECIMALS) for h in entropy[m:]]
    validate_article(rows, article, key, m)
    return rows, article


def build(sources_dir, output, model=None, replay=None, decode=None, device="cuda", train_dir=None,
          progress_path=None):
    sources_dir = Path(project_path(sources_dir))
    metadata = read_json(sources_dir / "metadata.json")
    config, key, side = metadata["config"], metadata["key"], metadata["side"]
    validate_l2(config)
    m, n, roles = config["context_width"], config["tokens"], config["l2"]["roles"]
    prompts = group_sources(read_jsonl(sources_dir / "sources.jsonl"))
    plans, needs_replay = [], False
    seen = {}
    for prompt in prompts:
        role = prompt["role"]
        index = seen.get(role, 0)
        seen[role] = index + 1
        recipes = assign_recipes(roles[role])[index]
        plans.append((prompt, index, recipes))
        needs_replay |= bool(roles[role].get("entropy")) or any(r["kind"] == "resample" for r in recipes)
    model_path = str(project_path(model)) if model else metadata.get("model")
    if replay is None and needs_replay:
        from ..pivot_experiment import ModelReplay
        replay = ModelReplay(model_path, metadata, device)
    if decode is None and model_path and Path(model_path).is_dir():
        from transformers import AutoTokenizer
        decode = AutoTokenizer.from_pretrained(model_path, local_files_only=True).convert_ids_to_tokens
    output = new_output(project_path(output))
    stage = "B1" if side == "train" else "B2"
    install_overall(config, progress_path or default_progress_path(output), (stage,))
    started = perf_counter()
    rows, articles, deferred, edit_stats, role_ids = [], [], [], {}, {r: i for i, r in enumerate(roles)}
    total = sum(len(recipes) for _, _, recipes in plans)
    progress("构建 positions", 0, total)
    reference = config["l2"].get("l_top_from")
    reference_truth = []
    for prompt, index, recipes in plans:
        role = prompt["role"]
        replayed = bool(roles[role].get("entropy")) or any(r["kind"] == "resample" for r in recipes)
        probs = entropy = None
        if replayed:
            probs = replay(prompt["prompt_ids"], prompt["W"]["token_ids"])
            entropy = entropy_matrix(probs)
        for number, recipe in enumerate(recipes):
            if recipe["kind"] == "alt" and recipe["L"] == "top":
                deferred.append((prompt, role, recipe))
                continue
            rng = np.random.default_rng([config["seed"], EDIT_SEED_TAG + 1, role_ids[role], index, number])
            article_rows, article = make_article(recipe, role, prompt, key, m, n, decode, rng, probs, entropy)
            if recipe["kind"] == "resample":
                edit_stats.setdefault(article["scenario"], []).append(article["edit_stats"])
            if side == "train" and reference and role == reference["role"] and recipe["name"] == reference["recipe"]:
                reference_truth.append(np.array([r["truth"] for r in article_rows], bool))
            rows.extend(article_rows)
            articles.append(article)
            tick(stage)
            progress("构建 positions", len(articles), total)
        del probs
    l_top = None
    if deferred:
        if side == "train":
            if not reference_truth:
                raise ValueError("l_top_from names no article of this build; cannot size alt_top.")
            l_top = l_top_from(reference_truth, n)
        else:
            if not train_dir:
                raise ValueError("The test side needs --train DIR to read L_top from the training build.")
            l_top = read_json(Path(project_path(train_dir)) / "manifest.json").get("l_top")
            if not l_top:
                raise ValueError("The training build has no L_top.")
        for prompt, role, recipe in deferred:
            article_rows, article = make_article(recipe, role, prompt, key, m, n, decode, l_top=l_top["L_top"])
            rows.extend(article_rows)
            articles.append(article)
            tick(stage)
            progress("构建 positions", len(articles), total)
    write_jsonl(output / "positions.jsonl", rows)
    write_jsonl(output / "articles.jsonl", articles)
    summary = {name: {"articles": len(v), "requested": sum(s["requested"] for s in v),
                      "eligible": sum(s["eligible"] for s in v), "edited": sum(s["edited"] for s in v),
                      "shortfall_articles": sum(s["shortfall"] > 0 for s in v)} for name, v in edit_stats.items()}
    write_json(output / "manifest.json", {
        "config": config, "key": key, "key_id": metadata.get("key_id"), "side": side, "protocol": metadata.get("protocol"),
        "sources": str(sources_dir), "articles": len(articles), "positions": len(rows), "l_top": l_top,
        "edit_summary": summary, "build_seconds": perf_counter() - started, "provenance": provenance(),
        "toy": bool(metadata.get("toy", False))})
    return {"positions": len(rows), "articles": len(articles), "l_top": l_top, "edit_summary": summary}
