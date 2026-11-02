"""L0 data contract. truth = S minus B, with B the seed-breaking window of every edit."""
import numpy as np

from ..protocol import extract_pivots

SPLITS = ("calibration", "dev", "test", "null_cal", "null_test", "pool")
TEST_SCENARIOS = ("full_wm", "switch", "resample_0.1", "resample_0.3")
SCENARIOS = TEST_SCENARIOS + ("null",)
FIELDS = ("article_id", "split", "scenario", "t", "token_id", "token", "Y",
          "truth", "edited", "dist_to_edit", "src_has_wm")


def scenario_for(split, index):
    """Deterministic scenario of the index-th article inside a role."""
    if split == "calibration":
        return "full_wm"
    if split in ("null_cal", "null_test"):
        return "null"
    if split in ("dev", "test"):
        return TEST_SCENARIOS[index % len(TEST_SCENARIOS)]
    raise ValueError(f"No scenario for split {split!r}.")


def resample_rate(scenario):
    return float(scenario.split("_", 1)[1]) if scenario.startswith("resample_") else None


def enabled_mask(n, ranges):
    """S: positions generated with the watermark rule on."""
    mask = np.zeros(n, bool)
    for start, stop in ranges:
        mask[start:stop] = True
    return mask


def breakage_mask(n, edited, context_width):
    """B = E union [t+1, t+m] for t in E; the seed of t+k depends on token t for k<=m."""
    mask = np.zeros(n, bool)
    for t in edited:
        mask[t:min(n, t + context_width + 1)] = True
    return mask


def truth_mask(n, ranges, edited, context_width):
    return enabled_mask(n, ranges) & ~breakage_mask(n, edited, context_width)


def distance_to_edit(n, edited):
    """Token distance to the nearest edit; None when the article has no edit."""
    if not len(edited):
        return [None] * n
    e = np.asarray(sorted(edited))
    return np.abs(np.arange(n)[:, None] - e[None, :]).min(axis=1).astype(int).tolist()


def build_article(source, final_tokens, edits, key, context_width, decode=None):
    """Long-format rows (t >= context_width only) and the article-level record."""
    n = len(final_tokens)
    if n <= context_width:
        raise ValueError("Article is shorter than the watermark context.")
    ranges = [list(r) for r in source["ranges"]]
    edited = sorted(int(e["t"]) for e in edits)
    enabled = enabled_mask(n, ranges)
    truth = truth_mask(n, ranges, edited, context_width)
    dist = distance_to_edit(n, edited)
    y, mask, _ = extract_pivots(final_tokens, context_width, key)
    edited_set = set(edited)
    rows = []
    for t in range(context_width, n):
        token_id = int(final_tokens[t])
        rows.append({"article_id": source["article_id"], "split": source["split"],
                     "scenario": source["scenario"], "t": t, "token_id": token_id,
                     "token": decode(token_id) if decode else str(token_id),
                     "Y": float(y[t]), "truth": bool(truth[t]), "edited": t in edited_set,
                     "dist_to_edit": dist[t], "src_has_wm": bool(enabled[t])})
    n_eff = n - context_width
    article = {"article_id": source["article_id"], "split": source["split"],
               "scenario": source["scenario"], "n_tokens": n, "n_eff": n_eff,
               "ranges": ranges, "edits": edits, "eps_true": float(truth[context_width:].sum() / n_eff),
               "token_ids": [int(t) for t in final_tokens]}
    return rows, article


def validate_article(rows, article, key, context_width):
    """Recompute Y and truth from the article record and compare with the rows."""
    n, m = article["n_tokens"], context_width
    if [r["t"] for r in rows] != list(range(m, n)) or article["n_eff"] != n - m:
        raise ValueError(f"{article['article_id']}: rows must cover t = {m}..{n - 1} exactly.")
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError(f"{article['article_id']}: row fields differ from the contract.")
        if row["split"] != article["split"] or row["scenario"] != article["scenario"]:
            raise ValueError(f"{article['article_id']}: split/scenario mismatch.")
        if not 0 < row["Y"] < 1:
            raise ValueError(f"{article['article_id']}: Y outside (0,1).")
        if row["truth"] and not row["src_has_wm"]:
            raise ValueError(f"{article['article_id']}: truth without a watermark source.")
        if row["truth"] and row["edited"]:
            raise ValueError(f"{article['article_id']}: an edited position cannot carry truth.")
    y, _, _ = extract_pivots(article["token_ids"], m, key)
    if not np.allclose([r["Y"] for r in rows], y[m:], rtol=0, atol=1e-15):
        raise ValueError(f"{article['article_id']}: Y does not match the stored tokens.")
    edited = sorted(e["t"] for e in article["edits"])
    expected = truth_mask(n, article["ranges"], edited, m)[m:]
    if not np.array_equal([r["truth"] for r in rows], expected):
        raise ValueError(f"{article['article_id']}: truth differs from S minus B.")
    if abs(article["eps_true"] - expected.mean()) > 1e-12:
        raise ValueError(f"{article['article_id']}: eps_true differs from |truth| / N_eff.")
