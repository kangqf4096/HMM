"""Load one side (train or test) of an L2 build into per-split stacked arrays."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..io import read_json, read_jsonl
from ..l0.schema import validate_article
from ..paths import project_path
from .emission import bin_counts, bin_index
from .plan import recipe_class


class Stack:
    """Equal-length articles of one split as arrays. `docs` keeps the per-article records."""

    def __init__(self, docs):
        lengths = {len(d["y"]) for d in docs}
        if len(lengths) != 1:
            raise ValueError("Articles of one split must share one length.")
        self.docs = docs
        self.y = np.stack([d["y"] for d in docs])
        self.truth = np.stack([d["truth"] for d in docs])
        self.index = bin_index(self.y)
        self.counts = bin_counts(self.index)
        self.n_eff = self.y.shape[1]

    def __len__(self):
        return len(self.docs)

    def truth_counts(self):
        """(articles, bins) counts of the truth = 1 pivots."""
        return np.stack([np.bincount(i[t], minlength=self.counts.shape[1]) for i, t in zip(self.index, self.truth)])

    def select(self, keep):
        return Stack([d for d, k in zip(self.docs, keep) if k])


def iter_position_groups(path):
    """Consecutive rows of one article (build writes each article contiguously)."""
    current, rows = None, []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["article_id"] != current and rows:
                yield current, rows
                rows = []
            current = row["article_id"]
            rows.append(row)
    if rows:
        yield current, rows


def load_side(directory, validate=True):
    directory = Path(project_path(directory))
    manifest = read_json(directory / "manifest.json")
    metas = {a["article_id"]: (order, a) for order, a in enumerate(read_jsonl(directory / "articles.jsonl"))}
    m = manifest["config"]["context_width"]
    by_split, seen = defaultdict(list), set()
    for article_id, article_rows in iter_position_groups(directory / "positions.jsonl"):
        if article_id not in metas or article_id in seen:
            raise ValueError("positions.jsonl and articles.jsonl cover different articles.")
        seen.add(article_id)
        order, meta = metas[article_id]
        if validate:
            validate_article(article_rows, meta, manifest["key"], m)
        doc = {"id": meta["article_id"], "order": order, "split": meta["split"], "scenario": meta["scenario"],
               "recipe": meta["recipe"], "cls": recipe_class(meta["recipe"]), "group": meta["group"],
               "n_eff": meta["n_eff"], "eps_true": meta["eps_true"],
               "paper_cps": meta["paper_changepoints"],
               "y": np.array([r["Y"] for r in article_rows]),
               "truth": np.array([r["truth"] for r in article_rows], bool),
               "edited": np.array([r["edited"] for r in article_rows], bool),
               "src_has_wm": np.array([r["src_has_wm"] for r in article_rows], bool),
               "entropy": None if "entropy" not in meta else np.array(meta["entropy"], float),
               "edits": meta["edits"], "edit_stats": meta.get("edit_stats")}
        by_split[meta["split"]].append(doc)
    if seen != set(metas):
        raise ValueError("positions.jsonl and articles.jsonl cover different articles.")
    for docs in by_split.values():
        docs.sort(key=lambda d: d["order"])
    return {"manifest": manifest, "splits": {s: Stack(d) for s, d in by_split.items()}, "directory": str(directory)}
