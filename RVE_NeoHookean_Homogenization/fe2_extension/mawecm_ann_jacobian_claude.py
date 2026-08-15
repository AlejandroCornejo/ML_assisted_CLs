#!/usr/bin/env python3
"""Analytic Jacobian of mawecm/mawecm_ann_weights.py's eval_mawecm_ann,
the one genuinely-missing link in a consistent (non-finite-difference)
tangent for D-HPROM-ANN/HPROM-ANN: the ANN-based homogenization-weight
regressor w(mu) = target_sum * softmax(MLP(mu)) has no Jacobian anywhere
in the existing codebase (unlike its RBF sibling, which has one).

Mirrors eval_mawecm_ann's exact forward pass (same normalization, same
per-layer Linear+activation, same final softmax+scale), computing the
Jacobian dw/dq alongside it via straightforward backprop through each
step. Read-only with respect to mawecm/mawecm_ann_weights.py -- this is
a new, standalone function, not a modification of it.

Single-query only (q_query one row), matching how eval_mawecm_ann is
actually called in this project's coord_label="mu" MAW homogenization
path (hprom_ann_solver_rve.py's _evaluate_maw_hom_weights_current).
"""
from __future__ import annotations

import numpy as np


def _activation_and_deriv(z, name):
    key = str(name).strip().lower()
    if key == "silu":
        zc = np.clip(z, -60.0, 60.0)
        sig = 1.0 / (1.0 + np.exp(-zc))
        y = z * sig
        dy = sig + z * sig * (1.0 - sig)
        return y, dy
    if key == "tanh":
        y = np.tanh(z)
        dy = 1.0 - y * y
        return y, dy
    if key == "relu":
        y = np.maximum(z, 0.0)
        dy = (z > 0.0).astype(float)
        return y, dy
    if key == "gelu":
        c = np.sqrt(2.0 / np.pi)
        u = c * (z + 0.044715 * z**3)
        t = np.tanh(u)
        y = 0.5 * z * (1.0 + t)
        du_dz = c * (1.0 + 3.0 * 0.044715 * z**2)
        dy = 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * du_dz
        return y, dy
    raise ValueError(f"Unsupported ANN activation '{name}'.")


def eval_mawecm_ann_with_jacobian(q_query, model):
    """Returns (w, dw_dq): w has shape (n_out,), dw_dq has shape (n_out, n_in).
    q_query must be a single row, shape (1, n_in) or (n_in,)."""
    q = np.asarray(q_query, dtype=float).reshape(1, -1)
    n_in = q.shape[1]
    x_mean = np.asarray(model["x_mean"], dtype=float).reshape(1, -1)
    x_std = np.asarray(model["x_std"], dtype=float).reshape(1, -1)
    x_std_safe = np.maximum(x_std, 1.0e-12)
    activation = str(model.get("activation", "silu"))
    n_layers = int(model["n_layers"])
    target_sum = float(model["target_sum"])

    y = (q - x_mean) / x_std_safe  # (1, n_in)
    # Jacobian of y w.r.t. q: diag(1/x_std_safe), shape (n_in, n_in)
    J = np.diag((1.0 / x_std_safe).reshape(-1))  # (n_in, n_in), dy/dq

    for i in range(n_layers):
        W = np.asarray(model[f"W_{i}"], dtype=float)  # (n_out_i, n_in_i)
        b = np.asarray(model[f"b_{i}"], dtype=float).reshape(1, -1)
        z = y @ W.T + b  # (1, n_out_i)
        J = W @ J  # dz/dq = W @ (dy_prev/dq), shape (n_out_i, n_in)
        if i != n_layers - 1:
            y, dphi = _activation_and_deriv(z, activation)  # both (1, n_out_i)
            J = dphi.reshape(-1, 1) * J  # d(phi(z))/dq, shape (n_out_i, n_in)
        else:
            y = z  # logits, no activation on the last layer

    logits = y.reshape(-1)  # (n_out,)
    J_logits = J  # (n_out, n_in) -- d(logits)/dq

    zc = logits - np.max(logits)
    ez = np.exp(np.clip(zc, -700.0, 700.0))
    prob = ez / max(np.sum(ez), 1.0e-300)  # (n_out,)

    # Softmax Jacobian: d(prob_k)/d(logit_j) = prob_k*(delta_kj - prob_j)
    J_softmax = np.diag(prob) - np.outer(prob, prob)  # (n_out, n_out)
    J_prob = J_softmax @ J_logits  # (n_out, n_in) -- d(prob)/dq

    w = target_sum * prob
    dw_dq = target_sum * J_prob
    return w, dw_dq


if __name__ == "__main__":
    # Minimal self-test against finite differences on a random small MLP,
    # BEFORE wiring this into anything -- isolate correctness of this one
    # piece first.
    rng = np.random.default_rng(0)
    n_in, n_out = 3, 5
    hidden = (7, 6)
    dims = [n_in] + list(hidden) + [n_out]
    model = {
        "x_mean": rng.standard_normal(n_in) * 0.1,
        "x_std": np.abs(rng.standard_normal(n_in)) + 0.5,
        "activation": "gelu",
        "target_sum": 12.3,
        "n_layers": len(dims) - 1,
    }
    for i in range(len(dims) - 1):
        model[f"W_{i}"] = rng.standard_normal((dims[i + 1], dims[i])) * 0.5
        model[f"b_{i}"] = rng.standard_normal(dims[i + 1]) * 0.1

    import sys
    sys.path.insert(0, "/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/mawecm")
    from mawecm_ann_weights import eval_mawecm_ann

    q0 = rng.standard_normal((1, n_in))
    w0, dw_dq = eval_mawecm_ann_with_jacobian(q0, model)
    w0_check = eval_mawecm_ann(q0, model).reshape(-1)
    print("value match (should be ~0):", np.linalg.norm(w0 - w0_check))

    h = 1.0e-6
    dw_dq_fd = np.zeros((n_out, n_in))
    for k in range(n_in):
        qp = q0.copy(); qp[0, k] += h
        qm = q0.copy(); qm[0, k] -= h
        wp = eval_mawecm_ann(qp, model).reshape(-1)
        wm = eval_mawecm_ann(qm, model).reshape(-1)
        dw_dq_fd[:, k] = (wp - wm) / (2 * h)

    rel_err = np.linalg.norm(dw_dq - dw_dq_fd) / max(np.linalg.norm(dw_dq_fd), 1e-30)
    print(f"analytic vs FD Jacobian relative error: {rel_err:.3e}")
    print("PASS" if rel_err < 1e-6 else "FAIL")
