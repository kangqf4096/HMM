"""Scaled two-state HMM. State 1 means surviving watermark coupling."""
from dataclasses import dataclass

import numpy as np


def transition(epsilon, rho):
    if not 0 <= epsilon <= 1 or not 0 <= rho < 1:
        raise ValueError("Need epsilon in [0,1] and rho in [0,1).")
    pi = np.array([1 - epsilon, epsilon], dtype=float)
    return pi, rho * np.eye(2) + (1 - rho) * np.tile(pi, (2, 1))


def _emissions(g, mask):
    g = np.asarray(g, dtype=float)
    if g.ndim != 1 or not len(g) or np.any(~np.isfinite(g)) or np.any(g <= 0):
        raise ValueError("Need a positive finite nonempty density-ratio array.")
    mask = np.ones(len(g), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != g.shape:
        raise ValueError("Mask must match sequence length.")
    return np.column_stack((np.ones(len(g)), np.where(mask, g, 1.))), mask


def _forward(b, pi, a):
    alpha = np.empty_like(b)
    scale = np.empty(len(b))
    alpha[0] = pi * b[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]
    for t in range(1, len(b)):
        alpha[t] = (alpha[t - 1] @ a) * b[t]
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]
    return alpha, scale


@dataclass
class Posterior:
    gamma: np.ndarray
    log_likelihood: float


def forward_log_likelihood(g, epsilon, rho, mask=None):
    b, _ = _emissions(g, mask)
    pi, a = transition(epsilon, rho)
    _, scale = _forward(b, pi, a)
    return float(np.log(scale).sum())


def forward_backward(g, epsilon, rho, mask=None):
    b, _ = _emissions(g, mask)
    pi, a = transition(epsilon, rho)
    if epsilon == 0:
        return Posterior(np.zeros(len(b)), 0.)
    if epsilon == 1:
        return Posterior(np.ones(len(b)), float(np.log(b[:, 1]).sum()))
    alpha, scale = _forward(b, pi, a)
    beta = np.ones_like(b)
    for t in range(len(b) - 2, -1, -1):
        beta[t] = a @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
    joint = alpha * beta
    gamma = joint[:, 1] / joint.sum(axis=1)
    return Posterior(np.clip(gamma, 0, 1), float(np.log(scale).sum()))


def viterbi(g, epsilon, rho, mask=None):
    b, _ = _emissions(g, mask)
    pi, a = transition(epsilon, rho)
    with np.errstate(divide="ignore"):
        loga, logpi, logb = np.log(a), np.log(pi), np.log(b)
    score = logpi + logb[0]
    parent = np.zeros((len(b), 2), dtype=int)
    for t in range(1, len(b)):
        candidates = score[:, None] + loga
        parent[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + logb[t]
    path = np.empty(len(b), dtype=int)
    path[-1] = score.argmax()
    for t in range(len(b) - 1, 0, -1):
        path[t - 1] = parent[t, path[t]]
    return path


def transition_general(a00, a11):
    if not (0 < a00 < 1 and 0 < a11 < 1):
        raise ValueError("Need a00 and a11 strictly inside (0,1).")
    return np.array([[a00, 1 - a00], [1 - a11, a11]], dtype=float)


def stationary(a00, a11):
    """P(z=1) under the stationary law of the 2-state chain with stay probabilities a00, a11."""
    return (1 - a00) / (2 - a00 - a11)


@dataclass
class TwoStateResult:
    """Leading dimensions broadcast over (sequence, parameter set); T is the last axis of `g`."""
    log_likelihood: np.ndarray   # relative to the all-null likelihood, summed over the T positions
    xi_sum: np.ndarray           # (..., 2, 2) pair posteriors summed over t (pairs with both positions in `mask`)
    gamma_first: np.ndarray      # P(z_1 = 1 | Y)
    gamma: np.ndarray | None     # (..., T) P(z_t = 1 | Y), only when requested


def forward_backward_two_state(g, pi1, a00, a11, mask=None, want_gamma=False):
    """Vectorised scaled forward-backward for f0 = 1, f1 / f0 = g, general transition (a00, a11).

    `pi1`, `a00`, `a11` broadcast against g.shape[:-1]; positions outside `mask` carry g = 1.
    """
    g = np.asarray(g, float)
    if g.ndim < 1 or not g.shape[-1] or np.any(~np.isfinite(g)) or np.any(g <= 0):
        raise ValueError("Need a positive finite density-ratio array with a nonempty last axis.")
    n = g.shape[-1]
    if mask is not None:
        mask = np.asarray(mask, bool)
        g = np.where(mask, g, 1.)
        pair = (mask[..., :-1] & mask[..., 1:]).astype(float)
    else:
        pair = None
    pi1, a00, a11 = (np.asarray(v, float) for v in (pi1, a00, a11))
    lead = np.broadcast_shapes(g.shape[:-1], pi1.shape, a00.shape, a11.shape)
    p = np.empty(lead + (n,))          # P(z_t = 1 | Y_1..t)
    c = np.empty(lead + (n,))          # scale: f(Y_t | past) / f0(Y_t)
    q = np.broadcast_to(pi1, lead).astype(float)
    for t in range(n):
        gt = g[..., t]
        c[..., t] = (1 - q) + q * gt
        p[..., t] = q * gt / c[..., t]
        q = (1 - p[..., t]) * (1 - a00) + p[..., t] * a11
    xi = np.zeros(lead + (2, 2))
    beta0 = np.ones(lead)
    beta1 = np.ones(lead)
    gamma = np.empty(lead + (n,)) if want_gamma else None
    if want_gamma:
        gamma[..., n - 1] = p[..., n - 1]
    for t in range(n - 2, -1, -1):
        # weights of the next step: b_{t+1}(j) beta_{t+1}(j) / c_{t+1}
        w0 = beta0 / c[..., t + 1]
        w1 = g[..., t + 1] * beta1 / c[..., t + 1]
        a0, a1 = 1 - p[..., t], p[..., t]
        pw = 1. if pair is None else pair[..., t]
        xi[..., 0, 0] += pw * a0 * a00 * w0
        xi[..., 0, 1] += pw * a0 * (1 - a00) * w1
        xi[..., 1, 0] += pw * a1 * (1 - a11) * w0
        xi[..., 1, 1] += pw * a1 * a11 * w1
        beta0, beta1 = a00 * w0 + (1 - a00) * w1, (1 - a11) * w0 + a11 * w1
        if want_gamma:
            gamma[..., t] = p[..., t] * beta1
    gamma_first = p[..., 0] * beta1
    return TwoStateResult(np.log(c).sum(axis=-1), xi, gamma_first, np.clip(gamma, 0, 1) if want_gamma else None)


def two_state_log_likelihood(g, pi1, a00, a11, mask=None):
    """Forward pass only (no backward), same broadcasting as forward_backward_two_state."""
    g = np.asarray(g, float)
    if mask is not None:
        g = np.where(np.asarray(mask, bool), g, 1.)
    pi1, a00, a11 = (np.asarray(v, float) for v in (pi1, a00, a11))
    lead = np.broadcast_shapes(g.shape[:-1], pi1.shape, a00.shape, a11.shape)
    q = np.broadcast_to(pi1, lead).astype(float)
    total = np.zeros(lead)
    for t in range(g.shape[-1]):
        gt = g[..., t]
        c = (1 - q) + q * gt
        total += np.log(c)
        p = q * gt / c
        q = (1 - p) * (1 - a00) + p * a11
    return total
