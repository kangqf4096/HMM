"""Scientific arXiv paper bodies, sampled and split by complete document."""
from pathlib import Path

import numpy as np

from .config import validate_text
from .io import new_output, provenance, read_json, write_json, write_jsonl
from .progress import progress
from .paths import project_path

DATASET = "ccdv/arxiv-summarization"
SUBSET = "document"
REVISION = "240aaf1a969b3f8cd0ade6986bfad0cd730ee288"
SPLITS = ("train", "validation", "test")
ROLES = ("calibration", "dev", "test", "null_cal", "null_test", "pool")
# Roles added after L0. Optional (default 0) and taken after the L0 roles, so the L0 assignment never moves.
EXTRA_ROLES = ("calibration2", "mixed_cal", "dev2", "test_seg")


def collect_candidates(raw, tokenizer, config, required, exclude=frozenset()):
    """Visit a seeded permutation of source rows; return `required` distinct eligible papers in visit order.

    Also returns the generator, whose state after the permutation the caller may keep using.

    `exclude` holds group names (arxiv-<split>-<row>) that must not be selected, e.g. rows used by earlier runs.
    """
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
    progress("抽取独立论文", 0, required)
    for index in rng.permutation(total):
        split_index = int(np.searchsorted(ends, index, side="right"))
        split = SPLITS[split_index]
        ordinal = int(index - (ends[split_index - 1] if split_index else 0))
        group = f"arxiv-{split}-{ordinal}"
        if group in exclude:
            continue
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
        candidates.append({"group": group,
                           "source_split": split, "source_row": ordinal,
                           "prompt_ids": tokens[:prompt_length],
                           "pool_ids": tokens[prompt_length:prompt_length + 2048]})
        progress("抽取独立论文", len(candidates), required, f"已检查 {examined} 篇")
        if len(candidates) == required:
            break
    if len(candidates) < required:
        raise ValueError(f"Need {required} distinct articles, only {len(candidates)} qualify. "
                         "Reduce role counts; do not split one paper across roles.")
    return candidates, total, examined, rng


def select_articles(raw, tokenizer, config):
    """Visit a seeded permutation of source rows; tokenize only until roles fill.

    The Parquet release has article/abstract columns, without paper IDs or titles.
    Compare complete bodies with normalized whitespace/case to keep duplicate
    documents out of different roles. This is content deduplication, not a file
    integrity check or a guarantee against different versions of the same paper.
    """
    counts = {"pool": 0, **{role: 0 for role in EXTRA_ROLES}, **config["roles"]}
    if any(type(counts.get(role)) is not int or counts[role] < (0 if role == "pool" or role in EXTRA_ROLES else 1)
           for role in ROLES + EXTRA_ROLES):
        raise ValueError(f"Need positive integer counts for each role (pool and {EXTRA_ROLES} may be 0): {ROLES}")
    old_required = sum(counts[role] for role in ROLES)
    required = old_required + sum(counts[role] for role in EXTRA_ROLES)
    candidates, total, examined, rng = collect_candidates(raw, tokenizer, config, required)
    # numpy's shuffle result depends on the list length, so only the first old_required candidates
    # (the L0 roles) are shuffled; extra roles take the remaining candidates in collection order.
    head = candidates[:old_required]
    rng.shuffle(head)
    candidates = head + candidates[old_required:]
    records, cursor = [], 0
    for role in ROLES + EXTRA_ROLES:
        for article in candidates[cursor:cursor + counts[role]]:
            article["role"] = role
            if role != "pool":
                article.pop("pool_ids")
            records.append(article)
        cursor += counts[role]
    return records, {"source_articles": total, "examined_articles": examined,
                     "selected_articles": required}


def prepare(config, model, output):
    model, output = project_path(model), project_path(output)
    validate_text(config)
    dataset = config.get("dataset", DATASET)
    subset = config.get("subset", SUBSET)
    revision = config.get("dataset_revision", REVISION)
    if dataset != DATASET or subset != SUBSET:
        raise ValueError(f"Use {DATASET}, subset={SUBSET}, for scientific arXiv paper bodies.")
    loader = config.get("dataset_loader", "datasets")
    if loader not in ("datasets", "parquet_file"):
        raise ValueError("dataset_loader must be datasets or parquet_file.")
    if not Path(model).is_dir():
        raise ValueError(f"Local model directory does not exist: {model}")
    progress("加载数据准备依赖")
    from transformers import AutoTokenizer
    output = new_output(output)
    progress("加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if loader == "parquet_file":
        from .arxiv_parquet import load_arxiv_parquet
        raw = load_arxiv_parquet(dataset, subset, revision)
    else:
        from .arxiv_datasets import load_arxiv_datasets
        raw = load_arxiv_datasets(dataset, subset, revision)
    if config.get("l2"):
        from .l2.plan import select_l2
        records, stats = select_l2(raw, tokenizer, config)
    else:
        records, stats = select_articles(raw, tokenizer, config)
    progress("写入实验输入")
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
                                        "dataset_loader": loader,
                                        "dataset_url": f"https://huggingface.co/datasets/{dataset}",
                                        "original_dataset_url": "https://huggingface.co/datasets/armanc/scientific_papers"})
    progress("arXiv 准备完成", stats["selected_articles"], stats["selected_articles"])
    return stats
