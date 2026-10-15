"""WikiText-2 raw, reconstructed and split by complete original article."""
import re
from pathlib import Path

import numpy as np

from .config import validate_text
from .io import new_output, provenance, write_json, write_jsonl

TITLE = re.compile(r"^\s*=\s*([^=]+?)\s*=\s*$")
ROLES = ("calibration", "dev", "test", "null_cal", "null_test", "pool")


def articles(rows, split):
    title, parts, ordinal = None, [], 0
    for row in rows:
        text = row["text"]
        heading = TITLE.fullmatch(text)
        if heading:
            if title is not None:
                yield {"group": f"{split}-{ordinal}", "title": title,
                       "source_split": split, "text": "\n".join(parts)}
                ordinal += 1
            title, parts = heading.group(1).strip(), [text.strip()]
        elif title is not None:
            parts.append(text.rstrip())
    if title is not None:
        yield {"group": f"{split}-{ordinal}", "title": title,
               "source_split": split, "text": "\n".join(parts)}


def prepare(config, model, output):
    validate_text(config)
    if not Path(model).is_dir():
        raise ValueError(f"Local model directory does not exist: {model}")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    output = new_output(output)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    dataset = config.get("dataset", "Salesforce/wikitext")
    subset = config.get("subset", "wikitext-2-raw-v1")
    if dataset != "Salesforce/wikitext":
        raise ValueError("v0 implements article-aware WikiText preparation only.")
    raw = load_dataset(dataset, subset, cache_dir=str(Path(".cache/hf/datasets")))
    candidates, seen = [], set()
    prompt_length = config.get("prompt_tokens", 50)
    minimum = prompt_length + max(config.get("context_width", 4), 64)
    for split in ("train", "validation", "test"):
        for article in articles(raw[split], split):
            normalized = article["title"].casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            tokens = tokenizer.encode(article["text"], add_special_tokens=False)
            if len(tokens) < minimum:
                continue
            article.pop("text")
            article["prompt_ids"] = tokens[:prompt_length]
            article["pool_ids"] = tokens[prompt_length:prompt_length + 2048]
            candidates.append(article)
    counts = config["roles"]
    if any(int(counts.get(role, 0)) < 1 for role in ROLES):
        raise ValueError(f"Need positive counts for each role: {ROLES}")
    required = sum(counts[role] for role in ROLES)
    if required > len(candidates):
        raise ValueError(f"Need {required} distinct articles, only {len(candidates)} qualify. "
                         "Reduce counts or explicitly use wikitext-103-raw-v1; do not split one article across roles.")
    rng = np.random.default_rng(config.get("seed", 20261015))
    rng.shuffle(candidates)
    records, cursor = [], 0
    for role in ROLES:
        for article in candidates[cursor:cursor + counts[role]]:
            article["role"] = role
            if role != "pool":
                article.pop("pool_ids")
            records.append(article)
        cursor += counts[role]
    write_jsonl(output / "prompts.jsonl", records)
    write_json(output / "manifest.json", {"model": str(Path(model).resolve()), "config": config,
                                        "provenance": provenance(),
                                        "eligible_articles": len(candidates),
                                        "selected_articles": required,
                                        "grouping": "one article per role/prompt; titles deduplicated across official splits",
                                        "dataset_url": "https://huggingface.co/datasets/Salesforce/wikitext"})
    return {"eligible_articles": len(candidates), "selected_articles": required}
