#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sampling utilities for ECM dataset construction.
"""

import numpy as np


def get_stratified_indices(n_total, n_pick, seed=42):
    """Pick n_pick indices from [0, n_total) with random-but-spread coverage."""
    n_total = int(n_total)
    n_pick = int(n_pick)
    if n_pick <= 0:
        return np.array([], dtype=int)
    if n_pick >= n_total:
        return np.arange(n_total, dtype=int)

    rng = np.random.default_rng(int(seed))
    picks = np.zeros(n_pick, dtype=int)
    edges = np.linspace(0, n_total, n_pick + 1, dtype=int)
    for i in range(n_pick):
        i0 = int(edges[i])
        i1 = int(edges[i + 1])
        if i1 <= i0:
            picks[i] = i0
        else:
            picks[i] = int(rng.integers(i0, i1))
    return np.sort(np.unique(picks))


def get_param_aware_indices(applied_strain, n_pick, seed=42, time_weight=0.20):
    """
    Parameter-aware subset selection from snapshot strain states.

    Uses farthest-point sampling (FPS) on normalized feature space:
      features = [eps_xx, eps_yy, gamma_xy, time_weight * normalized_step]
    """
    E = np.asarray(applied_strain, dtype=float)
    n_pick = int(n_pick)

    if E.ndim != 2 or E.shape[0] == 0 or E.shape[1] < 3:
        n_total = int(E.shape[0]) if E.ndim >= 1 else 0
        return get_stratified_indices(n_total, n_pick, seed=seed)

    n_total = int(E.shape[0])
    if n_pick <= 0:
        return np.array([], dtype=int)
    if n_pick >= n_total:
        return np.arange(n_total, dtype=int)

    X = np.asarray(E[:, :3], dtype=float)
    x_min = np.min(X, axis=0)
    x_span = np.ptp(X, axis=0)
    x_span[x_span < 1.0e-14] = 1.0
    Xn = (X - x_min) / x_span

    t = np.linspace(0.0, 1.0, n_total, dtype=float).reshape(-1, 1)
    F = np.concatenate([Xn, float(time_weight) * t], axis=1)

    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, n_total))

    selected = np.empty(n_pick, dtype=np.int64)
    selected[0] = first

    diff = F - F[first]
    min_d2 = np.einsum("ij,ij->i", diff, diff)
    min_d2[first] = -1.0
    n_sel = 1

    while n_sel < n_pick:
        idx = int(np.argmax(min_d2))
        if min_d2[idx] <= 0.0:
            # Degenerate feature cloud: fill remaining by stratified-in-time picks.
            remaining = np.setdiff1d(np.arange(n_total, dtype=np.int64), selected[:n_sel], assume_unique=False)
            need = n_pick - n_sel
            if remaining.size <= need:
                selected[n_sel:n_sel + remaining.size] = remaining
                n_sel += int(remaining.size)
                break
            extra_local = get_stratified_indices(remaining.size, need, seed=int(seed) + 911)
            selected[n_sel:n_sel + need] = remaining[extra_local[:need]]
            n_sel += need
            break

        selected[n_sel] = idx
        d = F - F[idx]
        d2 = np.einsum("ij,ij->i", d, d)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected[: n_sel + 1]] = -1.0
        n_sel += 1

    out = np.unique(selected[:n_sel]).astype(np.int64)
    if out.size < n_pick:
        remaining = np.setdiff1d(np.arange(n_total, dtype=np.int64), out, assume_unique=False)
        need = n_pick - out.size
        if remaining.size > 0:
            extra_local = get_stratified_indices(remaining.size, min(need, remaining.size), seed=int(seed) + 1777)
            out = np.concatenate([out, remaining[extra_local]])
            out = np.unique(out)

    return np.sort(out.astype(np.int64))

