"""Synonym-style edit: redraw a token from the original model distribution, without replacement."""
import numpy as np


def entropy_nats(p):
    p = np.asarray(p, float)
    positive = p[p > 0]
    return float(-np.sum(positive * np.log(positive)))


def resample(tokens, probs, rate, min_entropy, rng):
    """Edit int(rate * n) positions among those with entropy >= min_entropy.

    probs[t] is the model distribution at t given the ORIGINAL prefix, so every
    edit is independent of the other edits in the same article. The replacement
    is drawn from probs[t] with the original token removed and renormalized.
    """
    tokens = [int(t) for t in tokens]
    n = len(tokens)
    if probs.shape[0] != n or not 0 <= rate < 1 or min_entropy < 0:
        raise ValueError("Need one distribution per token, rate in [0,1) and min_entropy >= 0.")
    entropy = np.array([entropy_nats(p) for p in probs])
    eligible = np.flatnonzero(entropy >= min_entropy)
    requested = int(rate * n)
    chosen = np.sort(rng.choice(eligible, min(requested, len(eligible)), replace=False)) if requested else []
    final, edits = list(tokens), []
    for t in chosen:
        p = np.array(probs[t], float)
        old = tokens[t]
        p[old] = 0.
        if p.sum() <= 0:
            raise ValueError("No alternative token has positive probability.")
        new = int(rng.choice(len(p), p=p / p.sum()))
        final[t] = new
        edits.append({"t": int(t), "old": old, "new": new, "entropy_nats": float(entropy[t]),
                      "old_rank": int(1 + np.sum(probs[t] > probs[t][old])),
                      "new_rank": int(1 + np.sum(probs[t] > probs[t][new]))})
    return final, edits, {"requested": requested, "eligible": int(len(eligible)),
                          "edited": len(edits), "shortfall": requested - len(edits)}
