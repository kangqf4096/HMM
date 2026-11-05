"""Position, segment and document metrics; per-article arrays are bootstrapped by article."""
import numpy as np

from ..detector import intervals


def position_metrics(truth, labels):
    t, p = np.asarray(truth, bool), np.asarray(labels, bool)
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    positive = int(t.sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "signal_count": positive,
            # Undefined recall/F1/IoU (no surviving signal) stay None; precision with no flag is 0 by convention.
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / positive if positive else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if positive else None,
            "iou": tp / (tp + fp + fn) if positive else None}


def segment_report(truth, labels, min_cover=.5):
    """A true segment is recalled when >= min_cover of it is flagged; boundary error vs its best-overlap flagged segment."""
    t, p = np.asarray(truth, bool), np.asarray(labels, bool)
    true_segments, pred_segments = intervals(t), intervals(p)
    hits, errors = 0, []
    for a, b in true_segments:
        if p[a:b].mean() < min_cover:
            continue
        hits += 1
        overlap = [max(0, min(b, d) - max(a, c)) for c, d in pred_segments]
        c, d = pred_segments[int(np.argmax(overlap))]
        errors.append((abs(a - c), abs(b - d)))
    return {"true_segments": len(true_segments), "predicted_segments": len(pred_segments),
            "recalled_segments": hits,
            "segment_recall": hits / len(true_segments) if true_segments else None,
            "boundary_errors": errors}


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def boundary_summary(error_lists):
    """Pooled start/end offsets of all matched segments."""
    errors = [e for errs in error_lists for e in errs]
    if not errors:
        return {"matched": 0, "mean": None, "median": None}
    flat = np.array(errors, float).ravel()
    return {"matched": len(errors), "mean": float(flat.mean()), "median": float(np.median(flat))}


def bootstrap_indices(n, draws, seed):
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def bootstrap_ci(values, indices):
    """95% percentile CI of the mean of `values`, resampled by article with shared `indices`."""
    values = np.asarray(values, float)
    if not len(values):
        return {"mean": None, "ci95": [None, None]}
    means = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}


def paired_difference(candidate, reference, indices):
    """Paired bootstrap of mean(candidate - reference) over the same articles."""
    return bootstrap_ci(np.asarray(candidate, float) - np.asarray(reference, float), indices)


def auc(positive, negative):
    """P(positive score > negative score) with ties counted half (document-level ROC area)."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if not len(positive) or not len(negative):
        return None
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + .5 * ties) / (len(positive) * len(negative)))


def changepoints_from_labels(labels):
    """Boundaries between runs of equal labels, as 'segment starts' (0 < c < n)."""
    labels = np.asarray(labels, bool)
    return [int(i) for i in np.flatnonzero(labels[1:] != labels[:-1]) + 1]


def rand_index(true_points, predicted_points, n):
    """Rand index of the segmentations of n positions cut at the given boundaries (each segment is a cluster).

    Same definition as the segmentation paper (1.pdf): the share of position pairs on which the two
    clusterings agree about 'same cluster / different clusters'.
    """
    ids = [np.searchsorted(np.sort(np.asarray(points, int)), np.arange(n), side="right")
           for points in (true_points, predicted_points)]
    same = [i[:, None] == i[None, :] for i in ids]
    upper = np.triu(np.ones((n, n), bool), 1)
    return float(((same[0] == same[1]) & upper).sum() / upper.sum())


def changepoint_recovery(true_points, predicted_points):
    """Counts and the mean distance from each true boundary to the nearest predicted one (None if undefined)."""
    distance = None
    if len(true_points) and len(predicted_points):
        distance = float(np.mean([np.min(np.abs(np.asarray(predicted_points) - c)) for c in true_points]))
    return {"true_changepoints": len(true_points), "predicted_changepoints": len(predicted_points),
            "nearest_distance": distance}
