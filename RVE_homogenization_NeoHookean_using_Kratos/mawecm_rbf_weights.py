#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RBF regression for MAW-ECM adaptive weight fields.
"""

from __future__ import annotations

import numpy as np


def _as_2d(arr, name):
    x = np.asarray(arr, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {x.shape}.")
    return x


def _length_scales(q_train, factor=1.0):
    q = _as_2d(q_train, "q_train")
    span = np.max(q, axis=0) - np.min(q, axis=0)
    std = np.std(q, axis=0)
    l = np.maximum(np.maximum(0.25 * span, std), 1.0e-10)
    return np.asarray(float(factor) * l, dtype=float)


def _gaussian_kernel(query, centers, length_scales):
    q = _as_2d(query, "query")
    c = _as_2d(centers, "centers")
    l = np.asarray(length_scales, dtype=float).reshape(1, 1, -1)
    diff = q[:, None, :] - c[None, :, :]
    r2 = np.sum((diff / l) ** 2, axis=2)
    return np.exp(-0.5 * r2)


def _poly_block(q, poly_mode):
    q = _as_2d(q, "q")
    mode = int(poly_mode)
    if mode == 0:
        return np.zeros((q.shape[0], 0), dtype=float)
    if mode == 1:
        return np.ones((q.shape[0], 1), dtype=float)
    if mode == 2:
        return np.hstack([np.ones((q.shape[0], 1), dtype=float), q])
    raise ValueError("poly_mode must be 0 (none), 1 (const), or 2 (affine).")


def _choose_center_ids(n_samples, n_centers):
    n = int(n_samples)
    k = int(n_centers)
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    # Deterministic stratified pick.
    pos = np.linspace(0, n - 1, num=k)
    ids = np.unique(np.round(pos).astype(np.int64))
    if ids.size < k:
        extra = np.setdiff1d(np.arange(n, dtype=np.int64), ids, assume_unique=False)
        ids = np.concatenate([ids, extra[: (k - ids.size)]])
    return np.sort(ids[:k])


def fit_mawecm_rbf(
    q_train,
    W_train,
    n_centers=0,
    poly_mode=1,
    lambda_reg=1.0e-10,
    length_scale_factor=1.0,
):
    """
    Fit multi-output Gaussian RBF model.

    Parameters
    ----------
    q_train : (N, q_dim)
    W_train : (n_weights, N)
    """
    q = _as_2d(q_train, "q_train")
    W = _as_2d(W_train, "W_train")

    n_samples = int(q.shape[0])
    n_weights = int(W.shape[0])
    if W.shape[1] != n_samples:
        raise ValueError(
            f"W_train second dimension ({W.shape[1]}) must match q_train samples ({n_samples})."
        )

    center_ids = _choose_center_ids(n_samples=n_samples, n_centers=int(n_centers))
    centers = q[center_ids, :]
    l = _length_scales(q, factor=length_scale_factor)

    Phi = _gaussian_kernel(q, centers, l)  # N x Nc
    lam = float(max(lambda_reg, 0.0))

    s = np.sqrt(np.mean(W * W, axis=1))
    s = np.maximum(s, 1.0e-14)
    Y = (W / s[:, None]).T  # N x n_weights

    P = _poly_block(q, poly_mode)
    npoly = int(P.shape[1])
    Nc = int(Phi.shape[1])
    PtP = P.T @ P if npoly > 0 else None
    PhiTPhi = Phi.T @ Phi + lam * np.eye(Nc, dtype=float)
    PhiTY = Phi.T @ Y

    if npoly == 0:
        Alpha = np.linalg.lstsq(PhiTPhi, PhiTY, rcond=None)[0]
        Beta = np.zeros((0, n_weights), dtype=float)
    else:
        A_sys = np.block(
            [
                [PhiTPhi, Phi.T @ P],
                [P.T @ Phi, PtP],
            ]
        )
        B_sys = np.vstack([PhiTY, P.T @ Y])
        X = np.linalg.lstsq(A_sys, B_sys, rcond=None)[0]
        Alpha = X[:Nc, :]
        Beta = X[Nc:, :]

    model = {
        "center_ids": center_ids,
        "centers": centers,
        "length_scales": l,
        "Alpha": Alpha,
        "Beta": Beta,
        "scale": s,
        "poly_mode": int(poly_mode),
        "lambda_reg": float(lam),
        "n_centers": int(centers.shape[0]),
    }

    W_rec = eval_mawecm_rbf(q_query=q, model=model, clip_nonnegative=False, renorm_target=None)
    rec_rel = float(np.linalg.norm(W_rec - W) / max(np.linalg.norm(W), 1.0e-30))
    model["train_rel_error"] = rec_rel
    return model


def eval_mawecm_rbf(q_query, model, clip_nonnegative=True, renorm_target=None):
    q = _as_2d(q_query, "q_query")

    centers = _as_2d(model["centers"], "model.centers")
    l = np.asarray(model["length_scales"], dtype=float).reshape(-1)
    Alpha = np.asarray(model["Alpha"], dtype=float)
    Beta = np.asarray(model["Beta"], dtype=float)
    s = np.asarray(model["scale"], dtype=float).reshape(-1)
    poly_mode = int(model["poly_mode"])

    Phi_q = _gaussian_kernel(q, centers, l)  # M x Nc
    Yh = Phi_q @ Alpha

    Pq = _poly_block(q, poly_mode)
    if Pq.shape[1] > 0:
        Yh = Yh + Pq @ Beta

    W = (Yh.T * s[:, None])  # n_weights x M

    if bool(clip_nonnegative):
        W = np.maximum(W, 0.0)

    if renorm_target is not None:
        target = float(renorm_target)
        sw = np.sum(W, axis=0)
        good = sw > 1.0e-30
        if np.any(good):
            W[:, good] *= (target / sw[good])[None, :]

    return W


def eval_mawecm_rbf_with_jacobian(q_query, model, clip_nonnegative=True, renorm_target=None):
    """
    Evaluate MAW-RBF weights and analytic Jacobian dW/dq.

    Returns
    -------
    W : ndarray, shape (n_weights, n_query)
        Weight values.
    dW_dq : ndarray, shape (n_weights, n_query, q_dim)
        Analytic derivatives of each weight with respect to each query coordinate.
    """
    q = _as_2d(q_query, "q_query")

    centers = _as_2d(model["centers"], "model.centers")
    l = np.asarray(model["length_scales"], dtype=float).reshape(-1)
    Alpha = np.asarray(model["Alpha"], dtype=float)
    Beta = np.asarray(model["Beta"], dtype=float)
    s = np.asarray(model["scale"], dtype=float).reshape(-1)
    poly_mode = int(model["poly_mode"])

    m_query = int(q.shape[0])
    q_dim = int(q.shape[1])
    n_weights = int(s.size)

    Phi_q = _gaussian_kernel(q, centers, l)  # (M, Nc)
    Yh = Phi_q @ Alpha  # (M, n_weights)

    # dPhi/dq_a = Phi * (-(q_a-c_a)/l_a^2)
    diff = q[:, None, :] - centers[None, :, :]  # (M, Nc, q_dim)
    inv_l2 = 1.0 / np.maximum(l * l, 1.0e-30)  # (q_dim,)
    dPhi_dq = -Phi_q[:, :, None] * (diff * inv_l2[None, None, :])  # (M, Nc, q_dim)
    dYh_dq = np.einsum("mca,cn->mna", dPhi_dq, Alpha)  # (M, n_weights, q_dim)

    Pq = _poly_block(q, poly_mode)
    if Pq.shape[1] > 0:
        Yh = Yh + Pq @ Beta
        if poly_mode == 2:
            dP_dq = np.zeros((m_query, Pq.shape[1], q_dim), dtype=float)
            for a in range(q_dim):
                if 1 + a < Pq.shape[1]:
                    dP_dq[:, 1 + a, a] = 1.0
            dYh_dq += np.einsum("mpa,pn->mna", dP_dq, Beta)

    W_tilde = Yh.T * s[:, None]  # (n_weights, M)
    dW_tilde = np.transpose(dYh_dq, (1, 0, 2)) * s[:, None, None]

    W = W_tilde.copy()
    dW = dW_tilde.copy()

    if bool(clip_nonnegative):
        mask_pos = W > 0.0
        W = np.maximum(W, 0.0)
        dW = dW * mask_pos[:, :, None]

    if renorm_target is not None:
        target = float(renorm_target)
        for m in range(m_query):
            wm = W[:, m]
            sw = float(np.sum(wm))
            if sw <= 1.0e-30:
                W[:, m] = 0.0
                dW[:, m, :] = 0.0
                continue
            dsw = np.sum(dW[:, m, :], axis=0)
            fac = target / sw
            dW[:, m, :] = fac * dW[:, m, :] - (target / (sw * sw)) * wm[:, None] * dsw[None, :]
            W[:, m] = fac * wm

    return W, dW
