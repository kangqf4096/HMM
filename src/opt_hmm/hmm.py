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
