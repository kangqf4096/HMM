"""Token-level edits, independent of key/score, preserving an origin map."""
import numpy as np


def edit(token_ids, kind, rate, pool, rng):
    ids = list(map(int, token_ids))
    if kind not in ("replace", "insert", "delete") or not 0 <= rate < 1:
        raise ValueError("Unknown edit or invalid rate.")
    count = int(rate * len(ids))
    pool = np.asarray(pool, dtype=int)
    if kind != "delete" and (not len(pool) or np.any(pool < 0)):
        raise ValueError("Need a nonempty token pool.")
    if kind == "insert":
        targets = set(rng.choice(len(ids) + 1, count, replace=False).tolist())
        result, origin, actions = [], [], []
        for gap in range(len(ids) + 1):
            if gap in targets:
                token = int(rng.choice(pool))
                actions.append({"kind": kind, "gap": gap, "final": len(result), "new": token})
                result.append(token)
                origin.append(None)
            if gap < len(ids):
                result.append(ids[gap])
                origin.append(gap)
        return result, origin, actions
    targets = set(rng.choice(len(ids), count, replace=False).tolist())
    result, origin, actions = [], [], []
    for i, token in enumerate(ids):
        if i not in targets:
            result.append(token)
            origin.append(i)
        elif kind == "delete":
            actions.append({"kind": kind, "original": i, "old": token})
        else:
            candidates = pool[pool != token]
            if not len(candidates):
                raise ValueError("Replacement pool has no token different from the original.")
            replacement = int(rng.choice(candidates))
            actions.append({"kind": kind, "original": i, "final": len(result),
                            "old": token, "new": replacement})
            result.append(replacement)
            origin.append(None)
    return result, origin, actions
