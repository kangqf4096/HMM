"""Drop prompt groups from a built L2 side (positions.jsonl, articles.jsonl, manifest.json) without rebuilding it."""
import json
from collections import Counter
from pathlib import Path

from ..io import new_output, read_json, read_jsonl, write_json, write_jsonl
from ..paths import project_path


def edit_summary(articles):
    """The manifest's per-recipe edit summary, recomputed over the kept articles (as l2.build writes it)."""
    stats = {}
    for a in articles:
        if a["recipe"]["kind"] == "resample" and a.get("edit_stats") is not None:
            stats.setdefault(a["scenario"], []).append(a["edit_stats"])
    return {name: {"articles": len(v), "requested": sum(s["requested"] for s in v),
                   "eligible": sum(s["eligible"] for s in v), "edited": sum(s["edited"] for s in v),
                   "shortfall_articles": sum(s["shortfall"] > 0 for s in v)} for name, v in stats.items()}


def filter_side(input_dir, output_dir, groups, config=None):
    """Copy a side without the articles of `groups`. `config` replaces the manifest's configuration (both sides must
    get the same one). L_top, keys and every other manifest entry stay as built."""
    input_dir = Path(project_path(input_dir))
    manifest = read_json(input_dir / "manifest.json")
    drop = set(groups)
    articles = read_jsonl(input_dir / "articles.jsonl")
    kept = [a for a in articles if a["group"] not in drop]
    keep_ids = {a["article_id"] for a in kept}
    output = new_output(project_path(output_dir))
    rows = 0
    with (input_dir / "positions.jsonl").open(encoding="utf-8") as source, \
            (output / "positions.jsonl").open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            if line.strip() and json.loads(line)["article_id"] in keep_ids:
                target.write(line if line.endswith("\n") else line + "\n")
                rows += 1
    write_jsonl(output / "articles.jsonl", kept)
    removed = Counter(a.get("prompt_role") for a in articles if a["group"] in drop)
    write_json(output / "manifest.json", {
        **manifest, "config": config if config is not None else manifest["config"], "articles": len(kept),
        "positions": rows, "edit_summary": edit_summary(kept),
        "filtered": {"from": str(input_dir), "groups_requested": len(drop),
                     "groups_present": len({a["group"] for a in articles if a["group"] in drop}),
                     "articles_removed": len(articles) - len(kept), "positions_removed": manifest["positions"] - rows,
                     "articles_removed_by_role": dict(removed)}})
    return {"articles": len(kept), "positions": rows, "articles_removed": len(articles) - len(kept)}
