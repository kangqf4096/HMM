"""Context-keyed Gumbel-max protocol shared by generation and verification.

BLAKE2 is used by the WATERMARK to derive a random seed, not for file checksums.
PCG64 is the explicitly versioned research PRNG, not a cryptographic claim.
"""
import hashlib
import struct

import numpy as np

from .emissions import check_pivots

PROTOCOL = "gumbel-context-blake2b-pcg64-v1"
PUBLIC_KEY = "opt-hmm-public-research-key-v1"


def context_seed(context, key=PUBLIC_KEY):
    key_bytes = key.encode("utf-8")
    if not key_bytes or len(key_bytes) > 64:
        raise ValueError("Watermark key must have 1..64 UTF-8 bytes.")
    ids = [int(t) for t in context]
    if any(t < 0 or t >= 2**32 for t in ids):
        raise ValueError("Token IDs must fit unsigned 32 bits.")
    payload = struct.pack("<I", len(ids)) + struct.pack("<" + "I" * len(ids), *ids)
    return int.from_bytes(hashlib.blake2b(payload, key=key_bytes, digest_size=16,
                                       person=b"opt-hmm-v1").digest(), "little")


def uniforms(seed, vocab_size):
    # One raw 64-bit PRNG word per uniform. Midpoints avoid 0 and 1 and are
    # exactly representable in float64; scalar verification can jump directly.
    raw = np.random.PCG64(seed).random_raw(vocab_size)
    return ((raw >> np.uint64(12)).astype(np.float64) + .5) * (2. ** -52)


def pivot(seed, token_id):
    generator = np.random.PCG64(seed)
    generator.advance(int(token_id))
    raw = int(generator.random_raw())
    return ((raw >> 12) + .5) * (2. ** -52)


def extract_pivots(token_ids, context_width, key=PUBLIC_KEY):
    if context_width < 1:
        raise ValueError("Context width must be positive.")
    n = len(token_ids)
    y, mask, seeds = np.full(n, np.nan), np.zeros(n, bool), [None] * n
    # Detector sees the continuation only, not the original prompt.
    for t in range(context_width, n):
        seed = context_seed(token_ids[t - context_width:t], key)
        y[t], mask[t], seeds[t] = pivot(seed, token_ids[t]), True, str(seed)
    return y, mask, seeds


def generation_contexts(prompt_ids, token_ids, context_width):
    if context_width < 1 or len(prompt_ids) < context_width:
        raise ValueError("Prompt is shorter than watermark context.")
    full = list(prompt_ids) + list(token_ids)
    offset = len(prompt_ids)
    return [full[offset + t - context_width:offset + t] for t in range(len(token_ids))]


def operational_labels(final_ids, origin, original_ids, contexts, enabled, context_width):
    if len(final_ids) != len(origin):
        raise ValueError("Origin mapping length mismatch.")
    source, signal = [], []
    for t, old in enumerate(origin):
        retained = (old is not None and 0 <= old < len(original_ids)
                    and final_ids[t] == original_ids[old] and bool(enabled[old]))
        source.append(bool(retained))
        signal.append(bool(retained and t >= context_width
                           and final_ids[t - context_width:t] == contexts[old]))
    return np.asarray(source, bool), np.asarray(signal, bool)


def gumbel_winner(logits, u, temperature=1.):
    logits, u = np.asarray(logits, float), check_pivots(u)
    if temperature <= 0 or logits.shape != u.shape or not np.any(np.isfinite(logits)):
        raise ValueError("Invalid Gumbel-max inputs.")
    return int(np.argmax(logits / temperature - np.log(-np.log(u))))
