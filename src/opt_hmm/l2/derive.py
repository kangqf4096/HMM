"""Derive an L2 run at another text length from an existing one: same prompts, same sources, first n tokens.

Generation is causal and its randomness is keyed by (seed + request number, context), so the first n tokens of a
longer generation are the tokens a shorter generation would have produced. `check_prefix` tests that claim on data.
"""
from collections import Counter
from pathlib import Path

from ..io import new_output, read_json, read_jsonl, write_json, write_jsonl
from ..paths import project_path
from .config import validate_l2

# Settings that may differ between a run and its derived run without touching the prompts or the sources.
FREE = ("tokens", "model", "generation_storage")


def _strip(config):
    out = {k: v for k, v in config.items() if k not in FREE}
    out["l2"] = {k: v for k, v in config["l2"].items() if k != "coverage"}
    return out


def check_compatible(old, new):
    """The configurations must agree on everything that shapes prompts and sources (length and analysis switches aside)."""
    a, b = _strip(old), _strip(new)
    if a != b:
        raise ValueError("Configurations differ beyond tokens and l2.coverage (keys: "
                         f"{sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))}).")


def derive_prompts(source_dir, target_dir, config):
    """Copy a prepared prompt directory; the manifest carries the new configuration."""
    validate_l2(config)
    source_dir = Path(project_path(source_dir))
    manifest = read_json(source_dir / "manifest.json")
    check_compatible(manifest["config"], config)
    target = new_output(project_path(target_dir))
    prompts = read_jsonl(source_dir / "prompts.jsonl")
    write_jsonl(target / "prompts.jsonl", prompts)
    write_json(target / "manifest.json", {**manifest, "config": config, "derived_from": str(source_dir)})
    return {"prompts": len(prompts), "tokens": config["tokens"]}


def cut_source(source, tokens):
    """One source record cut to its first `tokens` tokens; watermark ranges are clipped to the kept part."""
    ids = source["token_ids"]
    if len(ids) < tokens:
        raise ValueError(f"{source['article_id']}: source has {len(ids)} tokens, cannot cut to {tokens}.")
    ranges = [[a, min(b, tokens)] for a, b in source["ranges"] if a < tokens]
    return {**source, "token_ids": [int(t) for t in ids[:tokens]], "ranges": ranges}


def derive_sources(source_dir, target_dir, config):
    """Cut every source of a finished generation run to config['tokens'] tokens; metadata keeps everything else."""
    validate_l2(config)
    source_dir = Path(project_path(source_dir))
    metadata = read_json(source_dir / "metadata.json")
    old = metadata["config"]["tokens"]
    if config["tokens"] > old:
        raise ValueError(f"Can only shorten sources ({old} -> {config['tokens']}).")
    check_compatible(metadata["config"], config)
    cut = [cut_source(s, config["tokens"]) for s in read_jsonl(source_dir / "sources.jsonl")]
    target = new_output(project_path(target_dir))
    write_jsonl(target / "sources.jsonl", cut)
    write_json(target / "metadata.json", {**metadata, "config": config,
                                          "derived_from": {"run": str(source_dir), "tokens": old}})
    return {"sources": len(cut), "tokens": config["tokens"], "from_tokens": old}


def check_prefix(long_dir, short_dir, output=None):
    """Do the long run's sources, cut to the short run's length, equal the short run's sources token by token?"""
    long_dir, short_dir = Path(project_path(long_dir)), Path(project_path(short_dir))
    short = {s["article_id"]: s for s in read_jsonl(short_dir / "sources.jsonl")}
    compared, identical, first_mismatch, by_source = 0, 0, [], {}
    matched_tokens = total_tokens = 0
    for source in read_jsonl(long_dir / "sources.jsonl"):
        other = short.get(source["article_id"])
        if other is None:
            continue
        n = len(other["token_ids"])
        a, b = source["token_ids"][:n], other["token_ids"]
        same = sum(x == y for x, y in zip(a, b))
        compared += 1
        matched_tokens, total_tokens = matched_tokens + same, total_tokens + n
        kind = source["article_id"].rsplit("#", 1)[-1]
        counts = by_source.setdefault(kind, Counter())
        counts["articles"] += 1
        if same == n and len(a) == n:
            identical += 1
            counts["identical"] += 1
        elif len(first_mismatch) < 20:
            at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), n))
            first_mismatch.append({"article_id": source["article_id"], "first_difference_at": at, "length": n})
    report = {"long": str(long_dir), "short": str(short_dir), "articles_compared": compared,
              "unmatched_short_articles": max(0, len(short) - compared),
              "identical": identical, "token_match_rate": matched_tokens / total_tokens if total_tokens else None,
              "by_source": {k: dict(v) for k, v in by_source.items()}, "first_mismatches": first_mismatch,
              "all_identical": bool(compared and identical == compared)}
    if output:
        write_json(project_path(output), report)
    return report
