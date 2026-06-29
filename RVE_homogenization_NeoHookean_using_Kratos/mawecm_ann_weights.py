#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANN regression for MAW-ECM adaptive weight fields.

The output layer is interpreted as logits and mapped through a softmax:

    w(mu) = sum_target * softmax(logits(mu))

This enforces nonnegative weights and the requested sum of weights by
construction. The trained model is stored as plain NumPy arrays so online
solvers can evaluate it without loading a torch module.
"""

from __future__ import annotations

import copy
import time
import numpy as np


def _as_2d(arr, name):
    x = np.asarray(arr, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {x.shape}.")
    return x


def _parse_hidden_dims(hidden_dims):
    if isinstance(hidden_dims, (tuple, list)):
        vals = [int(v) for v in hidden_dims]
    else:
        vals = [int(v.strip()) for v in str(hidden_dims).split(",") if v.strip()]
    if not vals:
        raise ValueError("hidden_dims must contain at least one layer width.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"hidden_dims entries must be positive. Got {vals}.")
    return tuple(vals)


def _activation_numpy(x, name):
    key = str(name).strip().lower()
    if key == "silu":
        z = np.clip(x, -60.0, 60.0)
        return x / (1.0 + np.exp(-z))
    if key == "tanh":
        return np.tanh(x)
    if key == "relu":
        return np.maximum(x, 0.0)
    if key == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    raise ValueError(f"Unsupported ANN activation '{name}'.")


def _softmax_numpy(logits):
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(np.clip(z, -700.0, 700.0))
    return ez / np.maximum(np.sum(ez, axis=1, keepdims=True), 1.0e-300)


def eval_mawecm_ann(q_query, model):
    q = _as_2d(q_query, "q_query")
    x_mean = np.asarray(model["x_mean"], dtype=float).reshape(1, -1)
    x_std = np.asarray(model["x_std"], dtype=float).reshape(1, -1)
    activation = str(model.get("activation", "silu"))
    n_layers = int(model["n_layers"])
    target_sum = float(model["target_sum"])

    y = (q - x_mean) / np.maximum(x_std, 1.0e-12)
    for i in range(n_layers):
        W = np.asarray(model[f"W_{i}"], dtype=float)
        b = np.asarray(model[f"b_{i}"], dtype=float).reshape(1, -1)
        y = y @ W.T + b
        if i != n_layers - 1:
            y = _activation_numpy(y, activation)
    prob = _softmax_numpy(y)
    return (target_sum * prob).T


def fit_mawecm_ann(
    q_train,
    W_train,
    target_sum,
    hidden_dims=(128, 128, 128),
    activation="silu",
    epochs=2000,
    batch_size=2048,
    lr=1.0e-3,
    weight_decay=1.0e-6,
    val_fraction=0.1,
    patience=200,
    seed=11,
    mse_weight=10.0,
    verbose=False,
    label="MAW-ANN",
):
    q = _as_2d(q_train, "q_train")
    W = _as_2d(W_train, "W_train")
    if W.shape[1] != q.shape[0]:
        raise ValueError(
            f"W_train second dimension ({W.shape[1]}) must match q_train samples ({q.shape[0]})."
        )
    if np.min(W) < -1.0e-12:
        raise ValueError(
            f"{label}: ANN softmax weights require nonnegative W_train. min={np.min(W):.3e}."
        )
    target = float(target_sum)
    if target <= 0.0:
        target = float(np.mean(np.sum(W, axis=0)))
    if target <= 0.0:
        raise ValueError(f"{label}: target_sum must be positive.")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError(
            "Torch is required to train MAW-ANN weight regressors. "
            "Use --maw-hom-weight-regressor rbf or install torch."
        ) from exc

    tag = str(label)

    def _log(msg):
        if bool(verbose):
            print(f"  [{tag}] {msg}", flush=True)

    t0 = time.perf_counter()
    hidden = _parse_hidden_dims(hidden_dims)
    act_key = str(activation).strip().lower()
    if act_key not in ("silu", "tanh", "relu", "gelu"):
        raise ValueError("activation must be one of: silu, tanh, relu, gelu.")

    x_mean = np.mean(q, axis=0)
    x_std = np.maximum(np.std(q, axis=0), 1.0e-12)
    X = ((q - x_mean[None, :]) / x_std[None, :]).astype(np.float32)

    P = np.maximum(W.T / target, 1.0e-14)
    P = P / np.maximum(np.sum(P, axis=1, keepdims=True), 1.0e-30)
    P = P.astype(np.float32)

    n_samples = int(X.shape[0])
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n_samples)
    n_val = int(round(float(val_fraction) * n_samples))
    n_val = min(max(n_val, 1), max(n_samples - 1, 1))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if train_idx.size < 1:
        train_idx = idx
        val_idx = idx[: min(n_samples, 1)]

    class _WeightNet(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            layers = []
            last = int(in_dim)
            for h in hidden:
                layers.append(nn.Linear(last, int(h)))
                if act_key == "silu":
                    layers.append(nn.SiLU())
                elif act_key == "tanh":
                    layers.append(nn.Tanh())
                elif act_key == "relu":
                    layers.append(nn.ReLU())
                elif act_key == "gelu":
                    layers.append(nn.GELU())
                last = int(h)
            layers.append(nn.Linear(last, int(out_dim)))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _WeightNet(X.shape[1], P.shape[1]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    X_train = torch.from_numpy(X[train_idx])
    P_train = torch.from_numpy(P[train_idx])
    X_val = torch.from_numpy(X[val_idx]).to(device)
    P_val = torch.from_numpy(P[val_idx]).to(device)
    loader = DataLoader(
        TensorDataset(X_train, P_train),
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        drop_last=False,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    eps = 1.0e-12

    _log(
        "ANN fit start: "
        f"q_train={q.shape}, W_train={W.shape}, hidden={hidden}, "
        f"target_sum={target:.6e}, device={device}"
    )

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, pb in loader:
            xb = xb.to(device)
            pb = pb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            logp = F.log_softmax(logits, dim=1)
            pred = torch.exp(logp)
            loss_kl = torch.sum(pb * (torch.log(torch.clamp(pb, min=eps)) - logp), dim=1).mean()
            loss_mse = torch.mean((pred - pb) ** 2)
            loss = loss_kl + float(mse_weight) * loss_mse
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])

        model.eval()
        with torch.no_grad():
            logits_v = model(X_val)
            logp_v = F.log_softmax(logits_v, dim=1)
            pred_v = torch.exp(logp_v)
            loss_kl_v = torch.sum(
                P_val * (torch.log(torch.clamp(P_val, min=eps)) - logp_v),
                dim=1,
            ).mean()
            loss_mse_v = torch.mean((pred_v - P_val) ** 2)
            val_loss = float((loss_kl_v + float(mse_weight) * loss_mse_v).detach().cpu())

        if val_loss < best_val - 1.0e-10:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if bool(verbose) and (epoch == 1 or epoch % 100 == 0 or epoch == int(epochs)):
            train_loss = running / max(n_seen, 1)
            _log(
                f"epoch {epoch:05d}: train={train_loss:.3e}, "
                f"val={val_loss:.3e}, best={best_val:.3e}@{best_epoch}"
            )

        if int(patience) > 0 and no_improve >= int(patience):
            _log(f"early stopping at epoch {epoch}, best epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    ann = {
        "x_mean": x_mean.astype(float),
        "x_std": x_std.astype(float),
        "activation": act_key,
        "hidden_dims": np.asarray(hidden, dtype=np.int64),
        "target_sum": float(target),
        "n_layers": int(len(layers)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
    }
    for i, layer in enumerate(layers):
        ann[f"W_{i}"] = layer.weight.detach().cpu().numpy().astype(float)
        ann[f"b_{i}"] = layer.bias.detach().cpu().numpy().astype(float)

    W_rec = eval_mawecm_ann(q, ann)
    train_rel = float(np.linalg.norm(W_rec - W) / max(np.linalg.norm(W), 1.0e-30))
    W_val = W[:, val_idx]
    W_val_rec = eval_mawecm_ann(q[val_idx], ann)
    val_rel = float(np.linalg.norm(W_val_rec - W_val) / max(np.linalg.norm(W_val), 1.0e-30))
    ann["train_rel_error"] = train_rel
    ann["val_rel_error"] = val_rel
    ann["elapsed_sec"] = float(time.perf_counter() - t0)

    _log(
        f"ANN fit complete in {ann['elapsed_sec']:.2f}s; "
        f"train_rel={train_rel:.3e}, val_rel={val_rel:.3e}"
    )
    return ann
