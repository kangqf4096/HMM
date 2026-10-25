"""Scientific arXiv paper bodies, sampled and split by complete document."""
from pathlib import Path

import numpy as np

from .config import validate_text
from .io import new_output, provenance, read_json, write_json, write_jsonl

DATASET = "ccdv/arxiv-summarization"
SUBSET = "document"
REVISION = "240aaf1a969b3f8cd0ade6986bfad0cd730ee288"
SPLITS = ("train", "validation", "test")
ROLES = ("calibration", "dev", "test", "null_cal", "null_test", "pool")


def select_articles(raw, tokenizer, config):
    """Visit a seeded permutation of source rows; tokenize only until roles fill.

    The Parquet release has article/abstract columns, without paper IDs or titles.
    Compare complete bodies with normalized whitespace/case to keep duplicate
    documents out of different roles. This is content deduplication, not a file
    integrity check or a guarantee against different versions of the same paper.
    """
    counts = config["roles"]
    if any(type(counts.get(role)) is not int or counts[role] < 1 for role in ROLES):
        raise ValueError(f"Need positive integer counts for each role: {ROLES}")
    required = sum(counts[role] for role in ROLES)
    ends = np.cumsum([len(raw[split]) for split in SPLITS])
    total = int(ends[-1])
    if required > total:
        raise ValueError(f"Need {required} distinct articles, only {total} source rows exist.")
    rng = np.random.default_rng(config.get("seed", 20261015))
    prompt_length = config.get("prompt_tokens", 50)
    minimum = prompt_length + max(config.get("context_width", 4), 64)
    limit = max(prompt_length + 2048, minimum)
    candidates, seen = [], set()
    examined = 0
    for index in rng.permutation(total):
        split_index = int(np.searchsorted(ends, index, side="right"))
        split = SPLITS[split_index]
        ordinal = int(index - (ends[split_index - 1] if split_index else 0))
        text = raw[split][ordinal]["article"]
        examined += 1
        if not isinstance(text, str):
            continue
        normalized = " ".join(text.split()).casefold()
        if not normalized or normalized in seen:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False,
                                  truncation=True, max_length=limit)
        if len(tokens) < minimum:
            continue
        seen.add(normalized)
        candidates.append({"group": f"arxiv-{split}-{ordinal}",
                           "source_split": split, "source_row": ordinal,
                           "prompt_ids": tokens[:prompt_length],
                           "pool_ids": tokens[prompt_length:prompt_length + 2048]})
        if len(candidates) == required:
            break
    if len(candidates) < required:
        raise ValueError(f"Need {required} distinct articles, only {len(candidates)} qualify. "
                         "Reduce role counts; do not split one paper across roles.")
    rng.shuffle(candidates)
    records, cursor = [], 0
    for role in ROLES:
        for article in candidates[cursor:cursor + counts[role]]:
            article["role"] = role
            if role != "pool":
                article.pop("pool_ids")
            records.append(article)
        cursor += counts[role]
    return records, {"source_articles": total, "examined_articles": examined,
                     "selected_articles": required}


def prepare(config, model, output):
    validate_text(config)
    dataset = config.get("dataset", DATASET)
    subset = config.get("subset", SUBSET)
    revision = config.get("dataset_revision", REVISION)
    if dataset != DATASET or subset != SUBSET:
        raise ValueError(f"Use {DATASET}, subset={SUBSET}, for scientific arXiv paper bodies.")
    if not Path(model).is_dir():
        raise ValueError(f"Local model directory does not exist: {model}")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    output = new_output(output)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    raw = load_dataset(dataset, subset, revision=revision,
                       cache_dir=str(Path(".cache/hf/datasets")))
    records, stats = select_articles(raw, tokenizer, config)
    model_source = Path(model) / "opt_hmm_source.json"
    write_jsonl(output / "prompts.jsonl", records)
    write_json(output / "manifest.json", {"model": str(Path(model).resolve()), "config": config,
                                        "model_source": read_json(model_source) if model_source.is_file() else None,
                                        "provenance": provenance(), **stats,
                                        "grouping": "one complete paper body per role/prompt; "
                                                    "whitespace/case-normalized bodies deduplicated across official splits",
                                        "sampling": "seeded random permutation of all official split rows; "
                                                    "stop once enough distinct eligible papers are selected",
                                        "dataset": dataset, "subset": subset,
                                        "dataset_revision": revision, "text_field": "article",
                                        "dataset_url": f"https://huggingface.co/datasets/{dataset}",
                                        "original_dataset_url": "https://huggingface.co/datasets/armanc/scientific_papers"})
    return stats
