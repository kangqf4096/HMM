"""Turn generated sources into positions.jsonl: resample edits, final-token pivots, truth."""
from pathlib import Path
from time import perf_counter

import numpy as np

from ..io import new_output, provenance, read_json, read_jsonl, write_json, write_jsonl
from ..paths import project_path
from ..progress import progress
from .config import validate_l0
from .resample import resample
from .schema import build_article, resample_rate, validate_article

EDIT_SEED_TAG = 7000


def edit_rng(config, index):
    return np.random.default_rng([config.get("seed", 20261015), EDIT_SEED_TAG, index])


def build(sources_dir, output, model=None, replay=None, decode=None, device="cuda"):
    sources_dir = Path(project_path(sources_dir))
    metadata = read_json(sources_dir / "metadata.json")
    config, key = metadata["config"], metadata["key"]
    validate_l0(config)
    m = config["context_width"]
    sources = read_jsonl(sources_dir / "sources.jsonl")
    edits_by_scenario = {f"resample_{e['rate']:g}": e for e in config["edits"]}
    model_path = str(project_path(model)) if model else metadata.get("model")
    if replay is None and any(resample_rate(s["scenario"]) is not None for s in sources):
        from ..pivot_experiment import ModelReplay
        replay = ModelReplay(model_path, metadata, device)
    if decode is None and model_path and Path(model_path).is_dir():
        from transformers import AutoTokenizer
        decode = AutoTokenizer.from_pretrained(model_path, local_files_only=True).convert_ids_to_tokens
    output = new_output(project_path(output))
    started = perf_counter()
    rows, articles, edit_stats = [], [], {}
    progress("构建 positions", 0, len(sources))
    for index, source in enumerate(sources):
        tokens, edits, stats = source["token_ids"], [], None
        rate = resample_rate(source["scenario"])
        if rate is not None:
            cfg = edits_by_scenario[source["scenario"]]
            probs = replay(source["prompt_ids"], tokens)
            tokens, edits, stats = resample(tokens, probs, cfg["rate"], cfg["min_entropy"], edit_rng(config, index))
            edit_stats.setdefault(source["scenario"], []).append(stats)
            del probs
        article_rows, article = build_article(source, tokens, edits, key, m, decode)
        article.update({"source_row": source.get("source_row"), "source_split": source.get("source_split"),
                        "edit_stats": stats})
        validate_article(article_rows, article, key, m)
        rows.extend(article_rows)
        articles.append(article)
        progress("构建 positions", index + 1, len(sources))
    write_jsonl(output / "positions.jsonl", rows)
    write_jsonl(output / "articles.jsonl", articles)
    summary = {name: {"articles": len(v), "requested": sum(s["requested"] for s in v),
                      "eligible": sum(s["eligible"] for s in v), "edited": sum(s["edited"] for s in v),
                      "shortfall_articles": sum(s["shortfall"] > 0 for s in v)}
               for name, v in edit_stats.items()}
    write_json(output / "manifest.json", {"config": config, "key": key, "key_id": metadata.get("key_id"),
                                          "protocol": metadata.get("protocol"), "sources": str(sources_dir),
                                          "articles": len(articles), "positions": len(rows),
                                          "edit_summary": summary, "build_seconds": perf_counter() - started,
                                          "provenance": provenance(), "toy": bool(metadata.get("toy", False))})
    return {"positions": len(rows), "articles": len(articles), "edit_summary": summary}
