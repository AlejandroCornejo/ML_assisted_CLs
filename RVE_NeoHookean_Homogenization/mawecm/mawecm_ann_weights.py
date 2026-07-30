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


def _pack_constraint_blocks(A_blocks, b_blocks, n_weights, n_samples, label):
    if A_blocks is None or b_blocks is None:
        return None, None, None
    if len(A_blocks) != int(n_samples) or len(b_blocks) != int(n_samples):
        raise ValueError(
            f"{label}: physical-loss blocks must match samples. "
            f"got len(A)={len(A_blocks)}, len(b)={len(b_blocks)}, samples={n_samples}."
        )

    m_max = max(int(np.asarray(A).shape[0]) for A in A_blocks) if A_blocks else 0
    if m_max <= 0:
        raise ValueError(f"{label}: physical-loss blocks have no rows.")
    A_pad = np.zeros((int(n_samples), int(m_max), int(n_weights)), dtype=np.float32)
    b_pad = np.zeros((int(n_samples), int(m_max)), dtype=np.float32)
    mask = np.zeros((int(n_samples), int(m_max)), dtype=np.float32)
    for i, (A, b) in enumerate(zip(A_blocks, b_blocks)):
        Aj = np.asarray(A, dtype=np.float32)
        bj = np.asarray(b, dtype=np.float32).reshape(-1)
        if Aj.ndim != 2:
            raise ValueError(f"{label}: A block {i} must be 2D, got {Aj.shape}.")
        if int(Aj.shape[1]) != int(n_weights):
            raise ValueError(
                f"{label}: A block {i} has {Aj.shape[1]} columns, expected {n_weights}."
            )
        if int(Aj.shape[0]) != int(bj.size):
            raise ValueError(
                f"{label}: A/b row mismatch in block {i}: {Aj.shape[0]} != {bj.size}."
            )
        m = int(bj.size)
        A_pad[i, :m, :] = Aj
        b_pad[i, :m] = bj
        mask[i, :m] = 1.0
    return A_pad, b_pad, mask


def _constraint_relative_error(W_eval, A_pad, b_pad, mask):
    if A_pad is None:
        return np.nan
    Wt = np.asarray(W_eval, dtype=float).T
    residual = np.einsum("smn,sn->sm", np.asarray(A_pad, dtype=float), Wt) - np.asarray(
        b_pad,
        dtype=float,
    )
    m = np.asarray(mask, dtype=float)
    num = float(np.sqrt(np.sum((residual * m) ** 2)))
    den = float(np.sqrt(np.sum((np.asarray(b_pad, dtype=float) * m) ** 2)))
    return num / max(den, 1.0e-30)


def fit_mawecm_ann(
    q_train,
    W_train,
    target_sum,
    constraint_A_blocks=None,
    constraint_b_blocks=None,
    physics_q_train=None,
    physics_constraint_A_blocks=None,
    physics_constraint_b_blocks=None,
    hidden_dims=(128, 128, 128),
    activation="silu",
    epochs=2000,
    batch_size=2048,
    lr=1.0e-3,
    weight_decay=1.0e-6,
    val_fraction=0.1,
    patience=200,
    lr_scheduler=True,
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=100,
    min_lr=1.0e-6,
    seed=11,
    mse_weight=10.0,
    physics_weight=0.0,
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

    q_phys = None
    if physics_q_train is not None:
        q_phys = _as_2d(physics_q_train, "physics_q_train")
        if q_phys.shape[1] != q.shape[1]:
            raise ValueError(
                f"{label}: physics_q_train dimension {q_phys.shape[1]} "
                f"does not match q_train dimension {q.shape[1]}."
            )
        norm_cloud = q_phys
    else:
        norm_cloud = q

    x_mean = np.mean(norm_cloud, axis=0)
    x_std = np.maximum(np.std(norm_cloud, axis=0), 1.0e-12)
    X = ((q - x_mean[None, :]) / x_std[None, :]).astype(np.float32)

    P = np.maximum(W.T / target, 1.0e-14)
    P = P / np.maximum(np.sum(P, axis=1, keepdims=True), 1.0e-30)
    P = P.astype(np.float32)

    phys_weight = float(physics_weight)
    use_physics = phys_weight > 0.0
    A_pad = b_pad = phys_mask = None
    X_phys = None
    if use_physics:
        phys_A_blocks = (
            physics_constraint_A_blocks
            if physics_constraint_A_blocks is not None
            else constraint_A_blocks
        )
        phys_b_blocks = (
            physics_constraint_b_blocks
            if physics_constraint_b_blocks is not None
            else constraint_b_blocks
        )
        q_for_physics = q_phys if q_phys is not None else q
        A_pad, b_pad, phys_mask = _pack_constraint_blocks(
            phys_A_blocks,
            phys_b_blocks,
            n_weights=W.shape[0],
            n_samples=q_for_physics.shape[0],
            label=label,
        )
        X_phys = ((q_for_physics - x_mean[None, :]) / x_std[None, :]).astype(np.float32)

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
    scheduler = None
    if bool(lr_scheduler):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(lr_scheduler_factor),
            patience=max(int(lr_scheduler_patience), 1),
            min_lr=float(min_lr),
        )

    X_train = torch.from_numpy(X[train_idx])
    P_train = torch.from_numpy(P[train_idx])
    X_val = torch.from_numpy(X[val_idx]).to(device)
    P_val = torch.from_numpy(P[val_idx]).to(device)
    if use_physics:
        if X_phys.shape[0] == X.shape[0] and q_phys is None:
            A_train = torch.from_numpy(A_pad[train_idx])
            b_train = torch.from_numpy(b_pad[train_idx])
            M_train = torch.from_numpy(phys_mask[train_idx])
            A_val = torch.from_numpy(A_pad[val_idx]).to(device)
            b_val = torch.from_numpy(b_pad[val_idx]).to(device)
            M_val = torch.from_numpy(phys_mask[val_idx]).to(device)
            dataset = TensorDataset(X_train, P_train, A_train, b_train, M_train)
            phys_loader = None
        else:
            n_phys = int(X_phys.shape[0])
            idx_phys = rng.permutation(n_phys)
            n_val_phys = int(round(float(val_fraction) * n_phys))
            n_val_phys = min(max(n_val_phys, 1), max(n_phys - 1, 1))
            val_phys_idx = idx_phys[:n_val_phys]
            train_phys_idx = idx_phys[n_val_phys:]
            X_phys_train = torch.from_numpy(X_phys[train_phys_idx])
            A_train = torch.from_numpy(A_pad[train_phys_idx])
            b_train = torch.from_numpy(b_pad[train_phys_idx])
            M_train = torch.from_numpy(phys_mask[train_phys_idx])
            X_phys_val = torch.from_numpy(X_phys[val_phys_idx]).to(device)
            A_val = torch.from_numpy(A_pad[val_phys_idx]).to(device)
            b_val = torch.from_numpy(b_pad[val_phys_idx]).to(device)
            M_val = torch.from_numpy(phys_mask[val_phys_idx]).to(device)
            dataset = TensorDataset(X_train, P_train)
            phys_dataset = TensorDataset(X_phys_train, A_train, b_train, M_train)
            phys_loader = DataLoader(
                phys_dataset,
                batch_size=max(int(batch_size), 1),
                shuffle=True,
                drop_last=False,
            )
    else:
        A_val = b_val = M_val = None
        X_phys_val = None
        phys_loader = None
        dataset = TensorDataset(X_train, P_train)
    loader = DataLoader(
        dataset,
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
        f"target_sum={target:.6e}, physics_weight={phys_weight:.3e}, device={device}"
    )

    def _loss_terms(logits, pb, Ab=None, bb=None, mb=None, x_phys_b=None):
        logp = F.log_softmax(logits, dim=1)
        pred = torch.exp(logp)
        if pb is None:
            loss_kl = logits.new_tensor(0.0)
            loss_mse = logits.new_tensor(0.0)
        else:
            loss_kl = torch.sum(
                pb * (torch.log(torch.clamp(pb, min=eps)) - logp),
                dim=1,
            ).mean()
            loss_mse = torch.mean((pred - pb) ** 2)
        loss_phys = logits.new_tensor(0.0)
        if use_physics and Ab is not None:
            if x_phys_b is not None:
                pred_phys = torch.softmax(model(x_phys_b), dim=1)
            else:
                pred_phys = pred
            w_pred = float(target) * pred_phys
            residual = torch.bmm(Ab, w_pred.unsqueeze(2)).squeeze(2) - bb
            residual = residual * mb
            rhs_norm2 = torch.clamp(torch.sum((bb * mb) ** 2, dim=1), min=1.0e-24)
            loss_phys = torch.mean(torch.sum(residual**2, dim=1) / rhs_norm2)
        loss = loss_kl + float(mse_weight) * loss_mse + phys_weight * loss_phys
        return loss, loss_phys

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running = 0.0
        running_phys = 0.0
        n_seen = 0
        phys_iter = iter(phys_loader) if phys_loader is not None else None
        for batch in loader:
            if use_physics:
                if phys_loader is None:
                    xb, pb, Ab, bb, mb = batch
                    Ab = Ab.to(device)
                    bb = bb.to(device)
                    mb = mb.to(device)
                    x_phys_b = None
                else:
                    xb, pb = batch
                    try:
                        x_phys_b, Ab, bb, mb = next(phys_iter)
                    except StopIteration:
                        phys_iter = iter(phys_loader)
                        x_phys_b, Ab, bb, mb = next(phys_iter)
                    x_phys_b = x_phys_b.to(device)
                    Ab = Ab.to(device)
                    bb = bb.to(device)
                    mb = mb.to(device)
            else:
                xb, pb = batch
                Ab = bb = mb = x_phys_b = None
            xb = xb.to(device)
            pb = pb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss, loss_phys = _loss_terms(logits, pb, Ab, bb, mb, x_phys_b)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu()) * int(xb.shape[0])
            running_phys += float(loss_phys.detach().cpu()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])

        model.eval()
        with torch.no_grad():
            logits_v = model(X_val)
            if use_physics and phys_loader is not None:
                val_loss_t, _ = _loss_terms(logits_v, P_val, None, None, None)
                logits_dummy = model(X_phys_val)
                _, val_phys_t = _loss_terms(logits_dummy, None, A_val, b_val, M_val)
                val_loss_t = val_loss_t + phys_weight * val_phys_t
            else:
                val_loss_t, val_phys_t = _loss_terms(logits_v, P_val, A_val, b_val, M_val)
            val_loss = float(val_loss_t.detach().cpu())
            val_phys = float(val_phys_t.detach().cpu())

        old_lr = float(opt.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(val_loss)
        new_lr = float(opt.param_groups[0]["lr"])
        if bool(verbose) and new_lr < old_lr:
            _log(f"lr reduced: {old_lr:.3e} -> {new_lr:.3e} at epoch {epoch}")

        if val_loss < best_val - 1.0e-10:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if bool(verbose) and (epoch == 1 or epoch % 100 == 0 or epoch == int(epochs)):
            train_loss = running / max(n_seen, 1)
            train_phys = running_phys / max(n_seen, 1)
            _log(
                f"epoch {epoch:05d}: train={train_loss:.3e}, "
                f"val={val_loss:.3e}, best={best_val:.3e}@{best_epoch}, "
                f"phys(train/val)={train_phys:.3e}/{val_phys:.3e}, "
                f"lr={float(opt.param_groups[0]['lr']):.3e}"
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
        "final_lr": float(opt.param_groups[0]["lr"]),
        "lr_scheduler": int(bool(lr_scheduler)),
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
    ann["physics_weight"] = float(phys_weight)
    if use_physics and (q_phys is not None or (X_phys is not None and X_phys.shape[0] != q.shape[0])):
        q_constraint = q_phys if q_phys is not None else q
        W_constraint_rec = eval_mawecm_ann(q_constraint, ann)
        ann["train_constraint_rel_error"] = _constraint_relative_error(
            W_constraint_rec,
            A_pad,
            b_pad,
            phys_mask,
        )
        ann["val_constraint_rel_error"] = _constraint_relative_error(
            W_constraint_rec[:, val_phys_idx],
            A_pad[val_phys_idx],
            b_pad[val_phys_idx],
            phys_mask[val_phys_idx],
        )
    else:
        ann["train_constraint_rel_error"] = _constraint_relative_error(W_rec, A_pad, b_pad, phys_mask)
        ann["val_constraint_rel_error"] = _constraint_relative_error(
            W_val_rec,
            A_pad[val_idx] if A_pad is not None else None,
            b_pad[val_idx] if b_pad is not None else None,
            phys_mask[val_idx] if phys_mask is not None else None,
        )
    ann["elapsed_sec"] = float(time.perf_counter() - t0)

    _log(
        f"ANN fit complete in {ann['elapsed_sec']:.2f}s; "
        f"train_rel={train_rel:.3e}, val_rel={val_rel:.3e}, "
        f"constraint_rel={ann['train_constraint_rel_error']:.3e}"
    )
    return ann
