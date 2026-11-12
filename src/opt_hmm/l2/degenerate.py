"""Degenerate articles: one 1/200 pivot bin holds most of the positions that should carry the watermark signal.

Rule (as tmp/refilter.py): among the positions that carry a watermark signal (truth = 1; every position for an
unwatermarked article), the largest single bin of 200 holds more than `max_share` of them, and there are at least
`min_positions` such positions. Looping text repeats one (context, token) pair and so one pivot value.
A prompt group is degenerate when any of its articles, at any scanned length, is.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..io import read_json, read_jsonl, write_json
from ..paths import project_path
from .data import iter_position_groups
from .emission import BINS

MAX_SHARE = .3
MIN_POSITIONS = 50


def concentration(y, bins=BINS):
    """(largest-bin share, positions) of the pivots `y`."""
    y = np.asarray(y, float)
    if not len(y):
        return 0., 0
    counts = np.bincount(np.minimum((y * bins).astype(int), bins - 1), minlength=bins)
    return float(counts.max() / len(y)), int(len(y))


def scan(directory, max_share=MAX_SHARE, min_positions=MIN_POSITIONS):
    """Degenerate articles of one *-pos directory."""
    directory = Path(project_path(directory))
    metas = {a["article_id"]: a for a in read_jsonl(directory / "articles.jsonl")}
    found, scanned = [], 0
    for article_id, rows in iter_position_groups(directory / "positions.jsonl"):
        meta = metas[article_id]
        null = meta["recipe"]["kind"] == "null"
        scanned += 1
        share, count = concentration([r["Y"] for r in rows if null or r["truth"]])
        if count >= min_positions and share > max_share:
            found.append({"article_id": article_id, "group": meta["group"], "role": meta.get("prompt_role"),
                          "split": meta["split"], "scenario": meta["scenario"], "positions": count,
                          "max_share": share})
    return {"articles": scanned, "degenerate": found}


def run_degenerate(directories, output, max_share=MAX_SHARE, min_positions=MIN_POSITIONS):
    """Scan every directory and write the union of degenerate prompt groups."""
    inputs, flagged, roles = {}, defaultdict(list), {}
    for directory in directories:
        manifest = read_json(Path(project_path(directory)) / "manifest.json")
        result = scan(directory, max_share, min_positions)
        inputs[str(directory)] = {"tokens": manifest["config"]["tokens"], "side": manifest["side"], **result}
        for item in result["degenerate"]:
            flagged[item["group"]].append(str(directory))
            roles[item["group"]] = item["role"]
    groups = sorted(flagged)
    by_role = defaultdict(int)
    for g in groups:
        by_role[roles[g]] += 1
    report = {"rule": {"max_share": max_share, "min_positions": min_positions, "bins": BINS},
              "inputs": inputs, "groups": groups, "flagged_in": {g: flagged[g] for g in groups},
              "groups_by_role": dict(by_role)}
    write_json(project_path(output), report)
    return {"groups": len(groups), "by_role": dict(by_role),
            "articles_flagged": {d: len(r["degenerate"]) for d, r in inputs.items()}}
