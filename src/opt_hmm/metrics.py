"""Token, segment, probability, and grouped paired evaluation."""
import numpy as np

from .detector import intervals


def segment_metrics(truth, predicted, min_iou=.5):
    true_segments, pred_segments = intervals(truth), intervals(predicted)
    k, j = len(true_segments), len(pred_segments)
    # Disjoint intervals in each sequence imply that positive-overlap matches
    # cannot cross. Ordered DP exactly maximizes (match count, total IoU).
    dp = [[(0, 0.) for _ in range(j + 1)] for _ in range(k + 1)]
    action = np.zeros((k + 1, j + 1), int)
    for a in range(1, k + 1):
        for b in range(1, j + 1):
            options = [(dp[a - 1][b], 1), (dp[a][b - 1], 2)]
            left, right = true_segments[a - 1], pred_segments[b - 1]
            intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
            union = left[1] - left[0] + right[1] - right[0] - intersection
            iou = intersection / union
            if iou >= min_iou:
                old = dp[a - 1][b - 1]
                options.append(((old[0] + 1, old[1] + iou), 3))
            dp[a][b], action[a, b] = max(options, key=lambda item: item[0])
    pairs = []
    a, b = k, j
    while a and b:
        choice = action[a, b]
        if choice == 3:
            pairs.append((a - 1, b - 1))
            a, b = a - 1, b - 1
        elif choice == 1:
            a -= 1
        else:
            b -= 1
    errors = [(abs(true_segments[a][0] - pred_segments[b][0]) +
               abs(true_segments[a][1] - pred_segments[b][1])) / 2 for a, b in pairs]
    c = len(pairs)
    return {"segment_tp": c, "true_segments": k, "predicted_segments": j,
            "segment_precision": c / j if j else 0.,
            "segment_recall": c / k if k else None,
            "segment_f1": 2 * c / (k + j) if k else None,
            "boundary_mae": float(np.mean(errors)) if errors else None}


def evaluate(truth, predicted, scores, epsilon, mask=None):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    scores = np.asarray(scores, float)
    mask = np.ones(len(truth), bool) if mask is None else np.asarray(mask, bool)
    if truth.shape != predicted.shape or truth.shape != mask.shape or truth.shape != scores.shape:
        raise ValueError("Evaluation arrays must have equal shapes.")
    if not mask.any():
        raise ValueError("Cannot evaluate a document with no available positions.")
    t, p, s = truth[mask], predicted[mask], scores[mask]
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    positives, negatives = int(t.sum()), int((~t).sum())
    actual = float(t.mean())
    result = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "available": int(mask.sum()), "signal_count": positives,
              "precision": tp / (tp + fp) if tp + fp else 0.,
              "recall": tp / positives if positives else None,
              "f1": 2 * tp / (2 * tp + fp + fn) if positives else None,
              "iou": tp / (tp + fp + fn) if positives else None,
              "zero_fpr": fp / negatives if negatives else None,
              "fdp": fp / max(tp + fp, 1), "any_fp": bool(fp),
              "actual_proportion": actual,
              "proportion_error": abs(epsilon - actual),
              "posterior_proportion_error": abs(float(s.mean()) - actual),
              "hard_proportion_error": abs(float(p.mean()) - actual),
              "brier": float(np.mean((s - t) ** 2))}
    result.update(segment_metrics(truth & mask, predicted & mask))
    return result


def reliability(truths, scores, bins=10):
    truths, scores = np.concatenate(truths), np.concatenate(scores)
    assignment = np.minimum((scores * bins).astype(int), bins - 1)
    return [{"lower": b / bins, "upper": (b + 1) / bins,
             "count": int(np.sum(assignment == b)),
             "mean_probability": float(scores[assignment == b].mean()) if np.any(assignment == b) else None,
             "signal_fraction": float(truths[assignment == b].mean()) if np.any(assignment == b) else None}
            for b in range(bins)]


def paired_bootstrap(rows, candidate, reference="independent", draws=1000, seed=42):
    # Average variants inside a prompt/article before resampling independent groups.
    by_group = {}
    for row in rows:
        if row["metrics"]["iou"] is None or row["method"] not in (candidate, reference):
            continue
        by_group.setdefault(row["group"], {}).setdefault(row["method"], []).append(row["metrics"]["iou"])
    differences = [np.mean(v[candidate]) - np.mean(v[reference])
                   for v in by_group.values() if candidate in v and reference in v]
    if not differences:
        return {"groups": 0, "mean": None, "ci95": [None, None]}
    d = np.asarray(differences)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(d, size=(draws, len(d)), replace=True), axis=1)
    return {"groups": len(d), "mean": float(d.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist()}
