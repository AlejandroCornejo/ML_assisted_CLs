#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 7b (POD-DL): train a POD-DL / POD-AE manifold model on fluctuation POD coordinates.

Workflow:
  1) Load q snapshots prepared in Stage 7a (or build on-the-fly if missing).
  2) Train autoencoder in POD space (q -> z -> q) with embedded scaling.
  3) Save model + metadata for online PROM solve.
"""

import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stage7a_prepare_pod_dl_dataset import collect_pod_dl_q_dataset
from pod_dl_manifold_model import PODDLAutoencoder


def _parse_hidden_dims(text):
    vals = []
    for p in str(text).split(","):
        s = p.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError("Empty hidden-dims list.")
    return tuple(vals)


def _parse_latent_sweep(text):
    vals = []
    for p in str(text).split(","):
        s = p.strip()
        if not s:
            continue
        vals.append(int(s))
    vals = sorted(set([v for v in vals if v > 0]))
    return vals


def _safe_rel_error_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.linalg.norm(y_true, ord="fro")
    if denom <= 0.0:
        return np.nan
    return float(100.0 * np.linalg.norm(y_true - y_pred, ord="fro") / denom)


def _collect_q_dataset(fom_dir, basis_dir, q_dim):
    return collect_pod_dl_q_dataset(fom_dir=fom_dir, basis_dir=basis_dir, q_dim=q_dim)


def _load_or_collect_q_dataset(data_dir, fom_dir, basis_dir, q_dim):
    q_file = os.path.join(data_dir, "q_dataset.npy")
    phi_file = os.path.join(data_dir, "phi_q.npy")
    meta_file = os.path.join(data_dir, "pod_dl_dataset_metadata.npz")

    if os.path.exists(q_file) and os.path.exists(phi_file):
        q_data = np.asarray(np.load(q_file), dtype=np.float64)
        phi_q = np.asarray(np.load(phi_file), dtype=np.float64)
        if q_data.ndim != 2 or phi_q.ndim != 2:
            raise RuntimeError(f"Invalid Stage 7a POD-DL dataset format in {data_dir}.")
        if int(q_data.shape[1]) != int(q_dim) or int(phi_q.shape[1]) != int(q_dim):
            raise RuntimeError(
                f"Stage 7a POD-DL q_dim mismatch: dataset has q_dim={q_data.shape[1]}, requested={int(q_dim)}."
            )
        used_traj = []
        steps_per_traj = []
        if os.path.exists(meta_file):
            meta = np.load(meta_file)
            if "used_trajectories" in meta:
                used_traj = [int(v) for v in np.asarray(meta["used_trajectories"]).ravel().tolist()]
            if "steps_per_trajectory" in meta:
                steps_per_traj = [int(v) for v in np.asarray(meta["steps_per_trajectory"]).ravel().tolist()]
        return q_data, phi_q, used_traj, steps_per_traj, "stage7a_cache"

    q_data, phi_q, used_traj, steps_per_traj = _collect_q_dataset(fom_dir=fom_dir, basis_dir=basis_dir, q_dim=q_dim)
    return q_data, phi_q, used_traj, steps_per_traj, "on_the_fly"


def _split_train_val(n_samples, validation_fraction, seed):
    if n_samples < 2:
        raise RuntimeError("Need at least 2 samples for train/val split.")
    n_val = int(np.floor(float(validation_fraction) * float(n_samples)))
    n_val = max(1, n_val)
    n_val = min(n_val, n_samples - 1)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_samples)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def _stats_from_train(x_train, scaling):
    scaling = str(scaling).strip().lower()
    if scaling == "zscore":
        q_mean = x_train.mean(axis=0)
        q_std = x_train.std(axis=0)
        q_std = np.where(q_std > 1e-10, q_std, 1.0)
        return {
            "q_mean": q_mean.astype(np.float32),
            "q_std": q_std.astype(np.float32),
            "q_min": None,
            "q_max": None,
        }
    if scaling in ("minmax", "minmax_-1_1"):
        q_min = x_train.min(axis=0)
        q_max = x_train.max(axis=0)
        return {
            "q_mean": None,
            "q_std": None,
            "q_min": q_min.astype(np.float32),
            "q_max": q_max.astype(np.float32),
        }
    raise ValueError("Unsupported scaling. Use zscore or minmax.")


def train_stage7b(
    fom_dir="stage_1_training_set_fom",
    basis_dir="stage_2_pod_rve",
    data_dir="stage_7_pod_dl_data",
    out_dir="stage_7_pod_dl_data",
    q_dim=9,
    latent_dim=4,
    latent_sweep=None,
    hidden_dims=(128, 64, 32),
    scaling="zscore",
    activation="elu",
    validation_fraction=0.10,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-5,
    epochs=1500,
    patience=100,
    min_improve=1e-12,
    clip_grad=1.0,
    use_scheduler=True,
    scheduler_factor=0.5,
    scheduler_patience=20,
    scheduler_threshold=1e-4,
    scheduler_min_lr=1e-6,
    scheduler_cooldown=5,
    seed=42,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_data, phi_q, used_traj, steps_per_traj, data_source = _load_or_collect_q_dataset(
        data_dir=data_dir,
        fom_dir=fom_dir,
        basis_dir=basis_dir,
        q_dim=q_dim,
    )
    n_samples = int(q_data.shape[0])
    q_dim = int(q_data.shape[1])

    if data_source != "stage7a_cache":
        np.save(os.path.join(data_dir, "q_dataset.npy"), q_data)
        np.save(os.path.join(data_dir, "phi_q.npy"), phi_q)

    train_idx, val_idx = _split_train_val(n_samples, validation_fraction, seed)
    x_train = q_data[train_idx, :].astype(np.float32)
    x_val = q_data[val_idx, :].astype(np.float32)
    x_val_t = torch.from_numpy(x_val).to(device)

    stats = _stats_from_train(x_train, scaling=scaling)

    if latent_sweep is None or len(latent_sweep) == 0:
        latent_candidates = [int(latent_dim)]
    else:
        latent_candidates = sorted(set([int(v) for v in latent_sweep if int(v) > 0] + [int(latent_dim)]))

    print("=" * 70)
    print("Stage 7b (POD-DL): POD-DL / POD-AE training")
    print("=" * 70)
    print(f"device: {device}")
    print(f"dataset source: {data_source}")
    print(f"data_dir: {data_dir}")
    print(f"q_data shape: {q_data.shape}")
    print(f"phi_q shape: {phi_q.shape}")
    print(f"train/val: {x_train.shape[0]}/{x_val.shape[0]}")
    print(f"latent candidates: {latent_candidates}")
    print(f"hidden_dims: {tuple(hidden_dims)}")
    print(f"scaling: {scaling} | activation: {activation}")

    best_val = float("inf")
    best_model_state = None
    best_candidate = None
    best_train_hist = []
    best_val_hist = []
    candidate_records = []

    for c_idx, z_dim in enumerate(latent_candidates):
        candidate_seed = int(seed) + int(c_idx)
        torch.manual_seed(candidate_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(candidate_seed)

        model = PODDLAutoencoder(
            q_dim=q_dim,
            latent_dim=int(z_dim),
            hidden_dims=hidden_dims,
            scaling=scaling,
            activation=activation,
            q_mean=stats["q_mean"],
            q_std=stats["q_std"],
            q_min=stats["q_min"],
            q_max=stats["q_max"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        scheduler = None
        if bool(use_scheduler):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(scheduler_factor),
                patience=int(scheduler_patience),
                threshold=float(scheduler_threshold),
                threshold_mode="rel",
                cooldown=int(scheduler_cooldown),
                min_lr=float(scheduler_min_lr),
            )

        loss_fn = nn.MSELoss()
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train)),
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=False,
        )

        local_best_val = float("inf")
        local_best_state = None
        bad_epochs = 0
        train_hist = []
        val_hist = []

        print("-" * 70)
        print(f"Training candidate latent_dim={z_dim}")
        for epoch in range(1, int(epochs) + 1):
            model.train()
            train_loss_acc = 0.0

            for (xb,) in train_loader:
                xb = xb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, xb)
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
                optimizer.step()
                train_loss_acc += float(loss.detach().cpu().item()) * xb.shape[0]

            train_mse = train_loss_acc / float(x_train.shape[0])
            train_hist.append(train_mse)

            model.eval()
            with torch.no_grad():
                val_mse = float(loss_fn(model(x_val_t), x_val_t).detach().cpu().item())
            val_hist.append(val_mse)
            if scheduler is not None:
                scheduler.step(val_mse)

            lr_now = float(optimizer.param_groups[0]["lr"])
            if epoch == 1 or epoch % 25 == 0:
                print(
                    f"[z={int(z_dim):3d} | epoch={epoch:4d}] "
                    f"train_mse={train_mse:.6e} | val_mse={val_mse:.6e} | lr={lr_now:.3e}"
                )

            if val_mse < (local_best_val - float(min_improve)):
                local_best_val = val_mse
                local_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(patience):
                    print(f"[EarlyStop z={int(z_dim)}] epoch={epoch}, best_val={local_best_val:.6e}")
                    break

        if local_best_state is None:
            local_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(local_best_state)
        model.eval()
        with torch.no_grad():
            train_pred = model(torch.from_numpy(x_train).to(device)).cpu().numpy()
            val_pred = model(torch.from_numpy(x_val).to(device)).cpu().numpy()

        train_rel = _safe_rel_error_percent(x_train, train_pred)
        val_rel = _safe_rel_error_percent(x_val, val_pred)
        final_lr = float(optimizer.param_groups[0]["lr"])

        candidate_records.append(
            {
                "latent_dim": int(z_dim),
                "best_val_mse": float(local_best_val),
                "train_rel_q_pct": float(train_rel),
                "val_rel_q_pct": float(val_rel),
                "epochs_ran": int(len(train_hist)),
                "final_lr": float(final_lr),
            }
        )

        print(
            f"[candidate z={int(z_dim)}] best_val_mse={local_best_val:.6e} | "
            f"train_rel_q={train_rel:.4f}% | val_rel_q={val_rel:.4f}%"
        )

        if local_best_val < best_val:
            best_val = float(local_best_val)
            best_model_state = local_best_state
            best_candidate = {
                "latent_dim": int(z_dim),
                "final_lr": float(final_lr),
                "train_rel_q_pct": float(train_rel),
                "val_rel_q_pct": float(val_rel),
            }
            best_train_hist = list(train_hist)
            best_val_hist = list(val_hist)

    if best_model_state is None or best_candidate is None:
        raise RuntimeError("No valid POD-DL model candidate was trained.")

    best_model = PODDLAutoencoder(
        q_dim=q_dim,
        latent_dim=int(best_candidate["latent_dim"]),
        hidden_dims=hidden_dims,
        scaling=scaling,
        activation=activation,
        q_mean=stats["q_mean"],
        q_std=stats["q_std"],
        q_min=stats["q_min"],
        q_max=stats["q_max"],
    ).to(device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    # Evaluate global reconstruction quality.
    with torch.no_grad():
        x_all = q_data.astype(np.float32)
        x_all_pred = best_model(torch.from_numpy(x_all).to(device)).cpu().numpy()
    rel_all_pct = _safe_rel_error_percent(q_data.astype(np.float32), x_all_pred)

    # Origin anchor diagnostic: decode(encode(0)).
    with torch.no_grad():
        q_zero = torch.zeros((1, q_dim), device=device)
        z_zero = best_model.encode(q_zero)
        q_ref = best_model.decode_from_latent(z_zero).cpu().numpy().reshape(-1)
    q_ref_norm = float(np.linalg.norm(q_ref))

    np.save(os.path.join(out_dir, "q_dataset.npy"), q_data)
    np.save(os.path.join(out_dir, "phi_q.npy"), phi_q)

    summary_txt = os.path.join(out_dir, "training_summary.txt")
    curve_png = os.path.join(out_dir, "training_history.png")
    model_file = os.path.join(out_dir, "pod_dl_autoencoder.pt")

    checkpoint = {
        "state_dict": best_model.state_dict(),
        "q_dim": int(q_dim),
        "latent_dim": int(best_candidate["latent_dim"]),
        "hidden_dims": tuple(int(v) for v in hidden_dims),
        "scaling": "minmax_-1_1" if str(scaling).lower().startswith("minmax") else "zscore",
        "activation": str(activation),
        "seed": int(seed),
        "basis_dir": str(basis_dir),
        "basis_file": os.path.join(basis_dir, "pod_basis_free.npy"),
        "fom_dir": str(fom_dir),
        "q_mean": stats["q_mean"],
        "q_std": stats["q_std"],
        "q_min": stats["q_min"],
        "q_max": stats["q_max"],
        "q_ref": q_ref.astype(np.float32),
        "train_indices": train_idx.astype(np.int64),
        "validation_indices": val_idx.astype(np.int64),
        "candidate_records": candidate_records,
    }
    torch.save(checkpoint, model_file)

    plt.figure(figsize=(8.0, 5.0))
    plt.plot(best_train_hist, label="train MSE", linewidth=1.5)
    plt.plot(best_val_hist, label="val MSE", linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_png, dpi=180)
    plt.close()

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Stage 7b POD-DL training summary\n")
        f.write(f"fom_dir={fom_dir}\n")
        f.write(f"basis_dir={basis_dir}\n")
        f.write(f"data_dir={data_dir}\n")
        f.write(f"out_dir={out_dir}\n")
        f.write(f"dataset_source={data_source}\n")
        f.write(f"q_dim={q_dim}\n")
        f.write(f"latent_dim={int(best_candidate['latent_dim'])}\n")
        f.write(f"hidden_dims={tuple(int(v) for v in hidden_dims)}\n")
        f.write(f"scaling={checkpoint['scaling']}\n")
        f.write(f"activation={activation}\n")
        f.write(f"n_samples={n_samples}\n")
        f.write(f"n_train={x_train.shape[0]}\n")
        f.write(f"n_val={x_val.shape[0]}\n")
        f.write(f"best_val_mse={best_val:.16e}\n")
        f.write(f"train_rel_q_pct={best_candidate['train_rel_q_pct']:.16e}\n")
        f.write(f"val_rel_q_pct={best_candidate['val_rel_q_pct']:.16e}\n")
        f.write(f"all_rel_q_pct={rel_all_pct:.16e}\n")
        f.write(f"q_ref_norm={q_ref_norm:.16e}\n")
        f.write(f"used_trajectories={used_traj}\n")
        f.write(f"steps_per_trajectory={steps_per_traj}\n")
        f.write(f"seed={int(seed)}\n")
        f.write(f"candidate_records={candidate_records}\n")

    print("=" * 70)
    print("Stage 7b (POD-DL) complete")
    print("=" * 70)
    print(f"Saved model: {model_file}")
    print(f"Saved basis slice: {os.path.join(out_dir, 'phi_q.npy')}")
    print(f"Saved dataset: {os.path.join(out_dir, 'q_dataset.npy')}")
    print(f"Saved curve: {curve_png}")
    print(f"Saved summary: {summary_txt}")
    print(f"Selected latent_dim: {int(best_candidate['latent_dim'])}")
    print(f"Best val MSE: {best_val:.6e}")
    print(f"Train rel q error [%]: {best_candidate['train_rel_q_pct']:.4f}")
    print(f"Val rel q error [%]:   {best_candidate['val_rel_q_pct']:.4f}")
    print(f"All rel q error [%]:   {rel_all_pct:.4f}")
    print(f"Origin anchor ||decode(encode(0))||: {q_ref_norm:.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 7b (POD-DL): train POD-DL / POD-AE manifold model.")
    p.add_argument("--fom-dir", type=str, default="stage_1_training_set_fom", help="Stage-1 FOM data directory.")
    p.add_argument("--basis-dir", type=str, default="stage_2_pod_rve", help="Stage-2 POD data directory.")
    p.add_argument(
        "--data-dir",
        type=str,
        default="stage_7_pod_dl_data",
        help="Stage 7a POD-DL dataset directory (q_dataset.npy, phi_q.npy).",
    )
    p.add_argument("--out-dir", type=str, default="stage_7_pod_dl_data", help="Output directory for POD-DL model.")
    p.add_argument("--q-dim", type=int, default=9, help="Number of POD coordinates used for POD-DL training.")
    p.add_argument("--latent-dim", type=int, default=4, help="Latent dimension for the selected model.")
    p.add_argument(
        "--latent-sweep",
        type=str,
        default="",
        help="Optional CSV sweep, e.g. '3,4,5,6'. Best candidate by validation MSE is selected.",
    )
    p.add_argument("--hidden-dims", type=str, default="128,64,32", help="CSV hidden widths for encoder MLP.")
    p.add_argument("--scaling", type=str, default="zscore", choices=["zscore", "minmax"], help="Embedded scaling.")
    p.add_argument("--activation", type=str, default="elu", choices=["elu", "tanh", "silu"], help="MLP activation.")
    p.add_argument("--validation-fraction", type=float, default=0.10, help="Validation split fraction.")
    p.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    p.add_argument("--epochs", type=int, default=1500, help="Maximum training epochs.")
    p.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    p.add_argument("--min-improve", type=float, default=1e-12, help="Minimum validation improvement.")
    p.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping norm.")
    p.add_argument("--no-scheduler", action="store_true", help="Disable ReduceLROnPlateau scheduler.")
    p.add_argument("--scheduler-factor", type=float, default=0.5, help="Scheduler LR decay factor.")
    p.add_argument("--scheduler-patience", type=int, default=20, help="Scheduler patience.")
    p.add_argument("--scheduler-threshold", type=float, default=1e-4, help="Scheduler relative threshold.")
    p.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="Scheduler min LR.")
    p.add_argument("--scheduler-cooldown", type=int, default=5, help="Scheduler cooldown.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = p.parse_args()

    latent_sweep = _parse_latent_sweep(args.latent_sweep) if str(args.latent_sweep).strip() else []

    train_stage7b(
        fom_dir=args.fom_dir,
        basis_dir=args.basis_dir,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        q_dim=args.q_dim,
        latent_dim=args.latent_dim,
        latent_sweep=latent_sweep,
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        scaling=args.scaling,
        activation=args.activation,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        min_improve=args.min_improve,
        clip_grad=args.clip_grad,
        use_scheduler=not args.no_scheduler,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_threshold=args.scheduler_threshold,
        scheduler_min_lr=args.scheduler_min_lr,
        scheduler_cooldown=args.scheduler_cooldown,
        seed=args.seed,
    )
