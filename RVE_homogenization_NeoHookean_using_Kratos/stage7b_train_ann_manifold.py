import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()
import argparse

# ============================================================
# ARCHITECTURE (MLP)
# ============================================================
class Scaler(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-10)

class Unscaler(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
    def forward(self, x):
        return x * self.std + self.mean


def _parse_hidden_layers(text):
    if isinstance(text, (list, tuple)):
        vals = [int(v) for v in text]
    else:
        t = str(text).strip()
        if not t:
            return (128, 128, 128, 128)
        vals = [int(v.strip()) for v in t.split(",") if v.strip()]
    if len(vals) < 1:
        raise ValueError("hidden_layers must contain at least one positive integer.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"All hidden layer widths must be > 0. Got: {vals}")
    return tuple(vals)


def _make_activation(name):
    key = str(name).strip().lower()
    if key == "elu":
        return nn.ELU()
    if key == "silu":
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation='{name}'. Use one of: elu, silu, relu, gelu, tanh.")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_layers=(128, 128, 128, 128),
        activation="silu",
        dropout=0.0,
        use_batchnorm=False,
    ):
        super().__init__()
        layers = []
        last_dim = in_dim
        act_name = str(activation).strip().lower()
        drop = float(dropout)
        bn = bool(use_batchnorm)
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_make_activation(act_name))
            if drop > 0.0:
                layers.append(nn.Dropout(p=drop))
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Handle single-sample inference cases common in ROM solvers
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0)
        return self.net(x)

class ManifoldANN(nn.Module):
    def __init__(
        self,
        x_mean,
        x_std,
        y_mean,
        y_std,
        in_dim=3,
        out_dim=6,
        hidden_layers=(128, 128, 128, 128),
        activation="silu",
        dropout=0.0,
        use_batchnorm=False,
        origin_anchored=False,
    ):
        super().__init__()
        if bool(origin_anchored) and bool(use_batchnorm):
            raise ValueError(
                "origin_anchored ANN is incompatible with BatchNorm. "
                "Use the default smooth network without BatchNorm."
            )
        self.origin_anchored = bool(origin_anchored)
        self.input_scaler = Scaler(x_mean, x_std)
        self.output_unscaler = Unscaler(y_mean, y_std)
        self.mlp = MLP(
            in_dim,
            out_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        
    def forward(self, x):
        x_scaled = self.input_scaler(x)
        y_scaled = self.mlp(x_scaled)
        if self.origin_anchored:
            zero = torch.zeros(
                (1, x_scaled.shape[-1]),
                dtype=x.dtype,
                device=x.device,
            )
            y0_scaled = self.mlp(self.input_scaler(zero))
            return (y_scaled - y0_scaled) * self.output_unscaler.std
        return self.output_unscaler(y_scaled)

# ============================================================
# TRAINING
# ============================================================
def _load_dataset_config(data_dir):
    meta_path = os.path.join(data_dir, "ann_dataset_metadata.npz")
    cfg = {
        "n_primary": None,
        "n_secondary": None,
        "input_dim": None,
        "include_macro_strain_input": False,
    }
    if not os.path.exists(meta_path):
        return cfg

    meta = np.load(meta_path)
    if "n_primary" in meta:
        cfg["n_primary"] = int(np.ravel(meta["n_primary"])[0])
    if "n_secondary" in meta:
        cfg["n_secondary"] = int(np.ravel(meta["n_secondary"])[0])
    if "input_dim" in meta:
        cfg["input_dim"] = int(np.ravel(meta["input_dim"])[0])
    if "include_macro_strain_input" in meta:
        cfg["include_macro_strain_input"] = bool(int(np.ravel(meta["include_macro_strain_input"])[0]))
    return cfg


def _select_input_file(data_dir):
    for name in ("q_m_train.npy", "q_p_train.npy", "ann_input_train.npy"):
        if os.path.exists(os.path.join(data_dir, name)):
            return name
    raise FileNotFoundError(
        f"No ANN input file found in {data_dir}. Expected q_m_train.npy, "
        "q_p_train.npy, or ann_input_train.npy."
    )


def _make_split_indices(X, val_fraction, seed, split_mode, spatial_bins):
    n_samples = int(X.shape[0])
    rng = np.random.default_rng(int(seed))
    mode = str(split_mode).strip().lower()

    if mode == "random":
        idx = rng.permutation(n_samples)
        n_train = min(max(int((1.0 - val_fraction) * n_samples), 2), n_samples - 1)
        return idx[:n_train], idx[n_train:]

    if mode != "spatial_blocks":
        raise ValueError(
            f"Unsupported split_mode='{split_mode}'. Use random or spatial_blocks."
        )

    bins = max(int(spatial_bins), 2)
    x_min = np.min(X, axis=0)
    x_span = np.maximum(np.max(X, axis=0) - x_min, 1e-14)
    x_unit = np.clip((X - x_min) / x_span, 0.0, 1.0 - np.finfo(float).eps)
    block = np.floor(x_unit * bins).astype(np.int64)

    # Stable integer hash: complete q_m blocks are assigned to validation,
    # avoiding the optimistic leakage produced by a pointwise random split.
    hashed = np.full(n_samples, int(seed) + 146959810, dtype=np.int64)
    for j in range(block.shape[1]):
        hashed = (hashed * 109951 + block[:, j] * (j + 17)) & 0x7FFFFFFF
    threshold = max(1, int(round(1000.0 * float(val_fraction))))
    val_mask = (hashed % 1000) < threshold

    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if train_idx.size < 2 or val_idx.size < 1:
        raise RuntimeError(
            "Spatial-block split produced an empty partition. "
            "Change --spatial-bins or --val-fraction."
        )
    return rng.permutation(train_idx), rng.permutation(val_idx)


def _copy_ls_decoder_files(data_dir, out_dir):
    if os.path.abspath(data_dir) == os.path.abspath(out_dir):
        return
    required = (
        "phi_m.npy",
        "phi_p.npy",
        "phi_s.npy",
        "A_m.npy",
        "ann_dataset_metadata.npz",
        "qm_init_mu_affine.npz",
    )
    optional = (
        "C_m.npy",
        "C_s.npy",
        "T_m.npy",
        "ls_coefficients_qfull_to_master.npy",
        "mu_to_qm_affine_diagnostics.npz",
        "mu_to_qm_affine_diagnostics.txt",
    )
    for name in required:
        src = os.path.join(data_dir, name)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"{src} is required for the LS-consistent ANN online decoder."
            )
        shutil.copy2(src, os.path.join(out_dir, name))
    for name in optional:
        src = os.path.join(data_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))


def _predict_numpy(model, X, device, batch_size):
    chunks = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), int(batch_size)):
            xb = torch.from_numpy(X[i:i + int(batch_size)]).to(device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_ann(
    data_dir="stage_7_ann_data",
    out_dir=None,
    seed=42,
    hidden_layers=(128, 128, 128, 128),
    activation="silu",
    dropout=0.0,
    use_batchnorm=False,
    val_fraction=0.1,
    split_mode="spatial_blocks",
    spatial_bins=12,
    max_samples=0,
    epochs=2000,
    patience=80,
    batch_size=2048,
    lr=1e-3,
    weight_decay=1e-6,
    lr_patience=25,
    lr_factor=0.5,
    loss="mse",
    smoothl1_beta=1.0,
    grad_clip_norm=0.0,
    origin_anchored=True,
    relative_loss_weight=5.0e-2,
    relative_loss_floor_fraction=1.0e-3,
):
    if out_dir is None or str(out_dir).strip() == "":
        out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    x_file = _select_input_file(data_dir)
    X = np.load(os.path.join(data_dir, x_file)).astype(np.float32)
    Y = np.load(os.path.join(data_dir, "q_s_train.npy")).astype(np.float32)
    cfg = _load_dataset_config(data_dir)
    if bool(cfg["include_macro_strain_input"]):
        raise RuntimeError(
            "Dataset was built with macro-strain manifold inputs (N(q,mu)), "
            "which is no longer supported. Rebuild Stage 7a without macro inputs."
        )
    in_dim = int(X.shape[1])
    out_dim = int(Y.shape[1])
    if cfg["input_dim"] is not None and cfg["input_dim"] != in_dim:
        raise ValueError(f"Dataset metadata/input mismatch: input_dim={cfg['input_dim']} vs X.shape[1]={in_dim}.")
    if cfg["n_secondary"] is not None and cfg["n_secondary"] != out_dim:
        raise ValueError(f"Dataset metadata/output mismatch: n_secondary={cfg['n_secondary']} vs Y.shape[1]={out_dim}.")

    max_samples = int(max_samples)
    if max_samples > 0 and max_samples < X.shape[0]:
        rng = np.random.default_rng(int(seed))
        keep = np.sort(rng.choice(X.shape[0], size=max_samples, replace=False))
        X = X[keep]
        Y = Y[keep]
    
    hidden_layers = _parse_hidden_layers(hidden_layers)
    activation = str(activation).strip().lower()
    dropout = float(dropout)
    use_batchnorm = bool(use_batchnorm)
    origin_anchored = bool(origin_anchored)
    if origin_anchored and use_batchnorm:
        raise ValueError(
            "Origin anchoring requires BatchNorm to be disabled."
        )
    val_fraction = float(val_fraction)
    if not (0.0 < val_fraction < 0.5):
        raise ValueError(f"val_fraction must be in (0, 0.5). Got {val_fraction}.")

    n_samples = X.shape[0]
    train_idx, val_idx = _make_split_indices(
        X,
        val_fraction=val_fraction,
        seed=seed,
        split_mode=split_mode,
        spatial_bins=spatial_bins,
    )
    n_train = int(train_idx.size)
    
    x_train, y_train = X[train_idx], Y[train_idx]
    x_val, y_val = X[val_idx], Y[val_idx]
    
    # Stats
    x_mean, x_std = x_train.mean(0), x_train.std(0)
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    y_std = np.where(y_std < 1e-10, 1.0, y_std)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ManifoldANN(
        x_mean,
        x_std,
        y_mean,
        y_std,
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        origin_anchored=origin_anchored,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_name = str(loss).strip().lower()
    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name in ("smoothl1", "huber"):
        criterion = nn.SmoothL1Loss(beta=float(smoothl1_beta))
    else:
        raise ValueError(f"Unsupported loss='{loss}'. Use one of: mse, smoothl1.")
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), 
                              batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), 
                            batch_size=int(batch_size))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=float(lr_factor),
        patience=int(lr_patience),
    )
    
    best_loss = float('inf')
    best_epoch = -1
    counter = 0
    
    history = {"train": [], "val": []}
    
    print("=" * 60)
    print("Stage 7b-ANN: Manifold MLP training (Normalized Loss)")
    print(f"  activation: {activation.upper()}")
    print(f"  hidden_layers: {hidden_layers}")
    print(f"  dropout: {dropout:.3f} | batchnorm: {int(use_batchnorm)}")
    print(
        f"  origin anchor: {int(origin_anchored)} | "
        f"relative loss weight: {float(relative_loss_weight):.3e}"
    )
    print(
        f"  loss: {loss_name}"
        + (f" (beta={float(smoothl1_beta):.3g})" if loss_name in ("smoothl1", "huber") else "")
    )
    print(f"  input_dim:  {in_dim} (n_primary={cfg['n_primary'] if cfg['n_primary'] is not None else in_dim})")
    print(f"  output_dim: {out_dim}")
    print("=" * 60)
    print(f"Training on device: {device}")
    print(
        f"Input file: {x_file} | samples: {n_samples} | "
        f"train/val: {n_train}/{len(val_idx)} | split={split_mode}"
    )
    print(
        f"  optimizer: AdamW(lr={float(lr):.2e}, weight_decay={float(weight_decay):.2e}), "
        f"scheduler: ReduceLROnPlateau(factor={float(lr_factor):.2f}, patience={int(lr_patience)})"
    )
    # Internal training helpers for normalized space
    y_mean_t = torch.tensor(y_mean, device=device)
    y_std_t = torch.tensor(y_std, device=device)
    y_norm_train = np.linalg.norm(y_train.astype(np.float64), axis=1)
    relative_floor = max(
        float(relative_loss_floor_fraction)
        * float(np.median(y_norm_train[y_norm_train > 0.0])),
        1.0e-12,
    )
    relative_floor_sq = relative_floor * relative_floor

    def _scaled_prediction_and_target(xb, yb):
        x_scaled = model.input_scaler(xb)
        y_scaled_raw = model.mlp(x_scaled)
        if origin_anchored:
            zero = torch.zeros(
                (1, xb.shape[1]),
                dtype=xb.dtype,
                device=xb.device,
            )
            y0_scaled = model.mlp(model.input_scaler(zero))
            return y_scaled_raw - y0_scaled, yb / y_std_t
        return y_scaled_raw, (yb - y_mean_t) / y_std_t

    def _training_loss(y_scaled_pred, y_scaled_target, y_target):
        base_loss = criterion(y_scaled_pred, y_scaled_target)
        if float(relative_loss_weight) <= 0.0:
            return base_loss
        y_pred = y_scaled_pred * y_std_t
        if not origin_anchored:
            y_pred = y_pred + y_mean_t
        sample_error_sq = torch.sum((y_pred - y_target) ** 2, dim=1)
        sample_target_sq = torch.sum(y_target ** 2, dim=1)
        relative_loss = torch.mean(
            sample_error_sq / (sample_target_sq + relative_floor_sq)
        )
        return base_loss + float(relative_loss_weight) * relative_loss

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # Predict
            y_scaled_pred, y_scaled_target = _scaled_prediction_and_target(xb, yb)
            loss = _training_loss(y_scaled_pred, y_scaled_target, yb)
            loss.backward()
            if float(grad_clip_norm) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
        
        train_loss = train_loss_sum / n_train
        
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_scaled_pred, y_scaled_target = _scaled_prediction_and_target(xb, yb)
                val_loss_sum += (
                    _training_loss(y_scaled_pred, y_scaled_target, yb).item()
                    * xb.size(0)
                )
        val_loss = val_loss_sum / len(val_idx)
        
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        
        # Step the scheduler
        scheduler.step(val_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:4d}: Scaled MSE Loss={val_loss:.4e} | LR={optimizer.param_groups[0]['lr']:.2e}")
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = int(epoch + 1)
            torch.save(model.state_dict(), os.path.join(out_dir, "manifold_ann.pt"))
            # Save metadata for inference
            np.savez(os.path.join(out_dir, "manifold_ann_metadata.npz"),
                     x_mean=x_mean, x_std=x_std,
                     y_mean=y_mean, y_std=y_std,
                     input_dim=np.array([in_dim], dtype=np.int64),
                     output_dim=np.array([out_dim], dtype=np.int64),
                     n_primary=np.array([cfg["n_primary"] if cfg["n_primary"] is not None else in_dim], dtype=np.int64),
                     n_secondary=np.array([cfg["n_secondary"] if cfg["n_secondary"] is not None else out_dim], dtype=np.int64),
                     include_macro_strain_input=np.array([0], dtype=np.int64),
                     hidden_layers=np.array(hidden_layers, dtype=np.int64),
                     activation=np.array([activation], dtype=np.str_),
                     dropout=np.array([float(dropout)], dtype=np.float64),
                     use_batchnorm=np.array([1 if use_batchnorm else 0], dtype=np.int64),
                     origin_anchored=np.array([1 if origin_anchored else 0], dtype=np.int64),
                     loss=np.array([loss_name], dtype=np.str_))
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_epoch < 0:
        raise RuntimeError("ANN training did not produce a finite validation checkpoint.")
    model.load_state_dict(
        torch.load(
            os.path.join(out_dir, "manifold_ann.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    y_val_pred = _predict_numpy(model, x_val, device=device, batch_size=batch_size)
    val_global_rel = float(
        np.linalg.norm(y_val_pred - y_val)
        / max(np.linalg.norm(y_val), 1e-30)
    )
    val_component_rel = np.linalg.norm(y_val_pred - y_val, axis=0) / np.maximum(
        np.linalg.norm(y_val, axis=0),
        1e-30,
    )
    val_target_norm = np.linalg.norm(y_val.astype(np.float64), axis=1)
    val_error_norm = np.linalg.norm(
        y_val_pred.astype(np.float64) - y_val.astype(np.float64),
        axis=1,
    )
    val_sample_rel = val_error_norm / np.maximum(val_target_norm, relative_floor)
    low_amplitude_mask = val_target_norm <= (10.0 * relative_floor)
    if np.any(low_amplitude_mask):
        val_low_amplitude_rel = float(
            np.linalg.norm(
                y_val_pred[low_amplitude_mask] - y_val[low_amplitude_mask]
            )
            / max(np.linalg.norm(y_val[low_amplitude_mask]), 1.0e-30)
        )
    else:
        val_low_amplitude_rel = float("nan")
    with torch.no_grad():
        origin_prediction = model(
            torch.zeros((1, in_dim), dtype=torch.float32, device=device)
        ).detach().cpu().numpy().reshape(-1)
    origin_prediction_norm = float(np.linalg.norm(origin_prediction))
    np.savez(
        os.path.join(out_dir, "ann_validation_predictions.npz"),
        indices=val_idx,
        q_m=x_val,
        q_s_true=y_val,
        q_s_pred=y_val_pred,
        component_rel_error=val_component_rel,
        global_rel_error=np.array([val_global_rel], dtype=float),
        sample_relative_error=val_sample_rel,
        low_amplitude_mask=low_amplitude_mask,
        low_amplitude_global_rel_error=np.array([val_low_amplitude_rel], dtype=float),
        origin_prediction=origin_prediction,
        origin_prediction_norm=np.array([origin_prediction_norm], dtype=float),
    )
    _copy_ls_decoder_files(data_dir, out_dir)

    # Plot history
    plt.figure(figsize=(8,5))
    plt.plot(history["train"], label="Train")
    plt.plot(history["val"], label="Val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(out_dir, "training_history.png"), dpi=150)
    plt.close()
    summary_path = os.path.join(out_dir, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 7b ANN training summary\n")
        f.write(f"data_dir={data_dir}\n")
        f.write(f"out_dir={out_dir}\n")
        f.write(f"seed={int(seed)}\n")
        f.write(f"samples_total={int(n_samples)}\n")
        f.write(f"samples_train={int(n_train)}\n")
        f.write(f"samples_val={int(len(val_idx))}\n")
        f.write(f"split_mode={split_mode}\n")
        f.write(f"spatial_bins={int(spatial_bins)}\n")
        f.write(f"input_dim={int(in_dim)}\n")
        f.write(f"output_dim={int(out_dim)}\n")
        f.write(f"hidden_layers={hidden_layers}\n")
        f.write(f"activation={activation}\n")
        f.write(f"dropout={float(dropout):.16e}\n")
        f.write(f"use_batchnorm={int(use_batchnorm)}\n")
        f.write(f"origin_anchored={int(origin_anchored)}\n")
        f.write(f"relative_loss_weight={float(relative_loss_weight):.16e}\n")
        f.write(f"relative_loss_floor_fraction={float(relative_loss_floor_fraction):.16e}\n")
        f.write(f"relative_loss_floor={float(relative_floor):.16e}\n")
        f.write(f"loss={loss_name}\n")
        f.write(f"smoothl1_beta={float(smoothl1_beta):.16e}\n")
        f.write(f"lr={float(lr):.16e}\n")
        f.write(f"weight_decay={float(weight_decay):.16e}\n")
        f.write(f"batch_size={int(batch_size)}\n")
        f.write(f"epochs_max={int(epochs)}\n")
        f.write(f"epochs_ran={int(len(history['val']))}\n")
        f.write(f"best_epoch={int(best_epoch)}\n")
        f.write(f"best_val_objective={float(best_loss):.16e}\n")
        f.write(f"last_val_objective={float(history['val'][-1]):.16e}\n")
        f.write(f"last_train_objective={float(history['train'][-1]):.16e}\n")
        # Compatibility aliases used by the existing Stage 7b sweep.
        f.write(f"best_val_scaled_mse={float(best_loss):.16e}\n")
        f.write(f"last_val_scaled_mse={float(history['val'][-1]):.16e}\n")
        f.write(f"last_train_scaled_mse={float(history['train'][-1]):.16e}\n")
        f.write(f"validation_qs_global_rel_error={val_global_rel:.16e}\n")
        f.write(
            "validation_qs_component_rel_error="
            + ",".join(f"{v:.16e}" for v in val_component_rel)
            + "\n"
        )
        f.write(f"validation_sample_rel_median={float(np.median(val_sample_rel)):.16e}\n")
        f.write(f"validation_sample_rel_p95={float(np.quantile(val_sample_rel, 0.95)):.16e}\n")
        f.write(f"validation_low_amplitude_count={int(np.sum(low_amplitude_mask))}\n")
        f.write(f"validation_low_amplitude_global_rel_error={val_low_amplitude_rel:.16e}\n")
        f.write(f"origin_prediction_norm={origin_prediction_norm:.16e}\n")
    print(f"Summary saved to: {summary_path}")
    print(f"Training complete. Best validation objective: {best_loss:.4e}")
    print(f"Validation q_s global relative error: {val_global_rel:.4e}")
    print(
        "Validation sample-relative error: "
        f"median={np.median(val_sample_rel):.4e}, "
        f"p95={np.quantile(val_sample_rel, 0.95):.4e}"
    )
    print(
        "Validation low-amplitude q_s relative error: "
        f"{val_low_amplitude_rel:.4e} ({int(np.sum(low_amplitude_mask))} samples)"
    )
    print(f"Origin constraint ||q_s(0)||: {origin_prediction_norm:.4e}")
    return {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "samples_total": int(n_samples),
        "samples_train": int(n_train),
        "samples_val": int(len(val_idx)),
        "input_dim": int(in_dim),
        "output_dim": int(out_dim),
        "hidden_layers": tuple(int(v) for v in hidden_layers),
        "activation": str(activation),
        "dropout": float(dropout),
        "use_batchnorm": int(use_batchnorm),
        "origin_anchored": int(origin_anchored),
        "relative_loss_weight": float(relative_loss_weight),
        "relative_loss_floor_fraction": float(relative_loss_floor_fraction),
        "loss": str(loss_name),
        "smoothl1_beta": float(smoothl1_beta),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "epochs_max": int(epochs),
        "epochs_ran": int(len(history["val"])),
        "best_epoch": int(best_epoch),
        "best_val_objective": float(best_loss),
        "last_val_objective": float(history["val"][-1]),
        "last_train_objective": float(history["train"][-1]),
        "best_val_scaled_mse": float(best_loss),
        "last_val_scaled_mse": float(history["val"][-1]),
        "last_train_scaled_mse": float(history["train"][-1]),
        "validation_qs_global_rel_error": float(val_global_rel),
        "validation_sample_rel_median": float(np.median(val_sample_rel)),
        "validation_sample_rel_p95": float(np.quantile(val_sample_rel, 0.95)),
        "validation_low_amplitude_global_rel_error": float(val_low_amplitude_rel),
        "origin_prediction_norm": float(origin_prediction_norm),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7b: train ANN manifold map")
    parser.add_argument("--data-dir", type=str, default="stage_7_ann_data", help="Dataset directory.")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory for model/metadata/plots. Defaults to data-dir.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and training init.")
    parser.add_argument("--hidden-layers", type=str, default="128,128,128,128", help="CSV hidden layer widths.")
    parser.add_argument("--activation", type=str, default="silu", choices=["elu", "silu", "relu", "gelu", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability in hidden layers.")
    parser.add_argument("--batchnorm", action="store_true", help="Enable BatchNorm1d (disabled by default for smooth online Jacobians).")
    parser.add_argument("--no-batchnorm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction in (0,0.5).")
    parser.add_argument("--split-mode", type=str, default="spatial_blocks", choices=["random", "spatial_blocks"])
    parser.add_argument("--spatial-bins", type=int, default=12, help="Bins per q_m coordinate for spatial-block validation.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional deterministic training subset (0 uses all samples).")
    parser.add_argument("--epochs", type=int, default=2000, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience (epochs).")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="AdamW weight decay.")
    parser.add_argument("--lr-patience", type=int, default=25, help="LR scheduler patience.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR scheduler factor.")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1"], help="Training loss.")
    parser.add_argument("--smoothl1-beta", type=float, default=1.0, help="Beta for SmoothL1 loss.")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Gradient clipping norm (0 disables).")
    parser.add_argument(
        "--no-origin-anchor",
        action="store_true",
        help="Disable the exact q_m=0 -> q_s=0 architecture constraint.",
    )
    parser.add_argument(
        "--relative-loss-weight",
        type=float,
        default=5.0e-2,
        help="Weight of the sample-relative q_s error term.",
    )
    parser.add_argument(
        "--relative-loss-floor-fraction",
        type=float,
        default=1.0e-3,
        help="Relative-loss denominator floor as a fraction of median nonzero ||q_s||.",
    )
    args = parser.parse_args()
    train_ann(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        dropout=args.dropout,
        use_batchnorm=bool(args.batchnorm and not args.no_batchnorm),
        val_fraction=args.val_fraction,
        split_mode=args.split_mode,
        spatial_bins=args.spatial_bins,
        max_samples=args.max_samples,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        loss=args.loss,
        smoothl1_beta=args.smoothl1_beta,
        grad_clip_norm=args.grad_clip_norm,
        origin_anchored=not args.no_origin_anchor,
        relative_loss_weight=args.relative_loss_weight,
        relative_loss_floor_fraction=args.relative_loss_floor_fraction,
    )
