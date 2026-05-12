import os
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
            return (128, 256, 256, 256, 128)
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
        hidden_layers=(128, 256, 256, 256, 128),
        activation="elu",
        dropout=0.0,
        use_batchnorm=True,
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
        hidden_layers=(128, 256, 256, 256, 128),
        activation="elu",
        dropout=0.0,
        use_batchnorm=True,
    ):
        super().__init__()
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


def train_ann(
    data_dir="stage_7_ann_data",
    out_dir=None,
    seed=42,
    hidden_layers=(128, 256, 256, 256, 128),
    activation="elu",
    dropout=0.0,
    use_batchnorm=True,
    val_fraction=0.1,
    epochs=2000,
    patience=80,
    batch_size=1024,
    lr=1e-3,
    weight_decay=1e-5,
    lr_patience=25,
    lr_factor=0.5,
    loss="mse",
    smoothl1_beta=1.0,
    grad_clip_norm=0.0,
):
    if out_dir is None or str(out_dir).strip() == "":
        out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    x_file = "ann_input_train.npy" if os.path.exists(os.path.join(data_dir, "ann_input_train.npy")) else "q_p_train.npy"
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
    
    hidden_layers = _parse_hidden_layers(hidden_layers)
    activation = str(activation).strip().lower()
    dropout = float(dropout)
    use_batchnorm = bool(use_batchnorm)
    val_fraction = float(val_fraction)
    if not (0.0 < val_fraction < 0.5):
        raise ValueError(f"val_fraction must be in (0, 0.5). Got {val_fraction}.")

    # Split
    n_samples = X.shape[0]
    n_train = int((1.0 - val_fraction) * n_samples)
    n_train = min(max(n_train, 2), n_samples - 1)
    idx = np.random.permutation(n_samples)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    
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
        verbose=True,
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
        f"  loss: {loss_name}"
        + (f" (beta={float(smoothl1_beta):.3g})" if loss_name in ("smoothl1", "huber") else "")
    )
    print(f"  input_dim:  {in_dim} (n_primary={cfg['n_primary'] if cfg['n_primary'] is not None else in_dim})")
    print(f"  output_dim: {out_dim}")
    print("=" * 60)
    print(f"Training on device: {device}")
    print(f"Input file: {x_file} | samples: {n_samples} | train/val: {n_train}/{n_samples - n_train}")
    print(
        f"  optimizer: AdamW(lr={float(lr):.2e}, weight_decay={float(weight_decay):.2e}), "
        f"scheduler: ReduceLROnPlateau(factor={float(lr_factor):.2f}, patience={int(lr_patience)})"
    )
    # Internal training helpers for normalized space
    y_mean_t = torch.tensor(y_mean, device=device)
    y_std_t = torch.tensor(y_std, device=device)

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # Predict
            x_scaled = model.input_scaler(xb)
            y_scaled_pred = model.mlp(x_scaled)
            
            # Target in normalized space
            y_scaled_target = (yb - y_mean_t) / y_std_t
            
            loss = criterion(y_scaled_pred, y_scaled_target)
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
                x_scaled = model.input_scaler(xb)
                y_scaled_pred = model.mlp(x_scaled)
                y_scaled_target = (yb - y_mean_t) / y_std_t
                val_loss_sum += criterion(y_scaled_pred, y_scaled_target).item() * xb.size(0)
        val_loss = val_loss_sum / (n_samples - n_train)
        
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
                     loss=np.array([loss_name], dtype=np.str_))
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
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
        f.write(f"samples_val={int(n_samples - n_train)}\n")
        f.write(f"input_dim={int(in_dim)}\n")
        f.write(f"output_dim={int(out_dim)}\n")
        f.write(f"hidden_layers={hidden_layers}\n")
        f.write(f"activation={activation}\n")
        f.write(f"dropout={float(dropout):.16e}\n")
        f.write(f"use_batchnorm={int(use_batchnorm)}\n")
        f.write(f"loss={loss_name}\n")
        f.write(f"smoothl1_beta={float(smoothl1_beta):.16e}\n")
        f.write(f"lr={float(lr):.16e}\n")
        f.write(f"weight_decay={float(weight_decay):.16e}\n")
        f.write(f"batch_size={int(batch_size)}\n")
        f.write(f"epochs_max={int(epochs)}\n")
        f.write(f"epochs_ran={int(len(history['val']))}\n")
        f.write(f"best_epoch={int(best_epoch)}\n")
        f.write(f"best_val_scaled_mse={float(best_loss):.16e}\n")
        f.write(f"last_val_scaled_mse={float(history['val'][-1]):.16e}\n")
        f.write(f"last_train_scaled_mse={float(history['train'][-1]):.16e}\n")
    print(f"Summary saved to: {summary_path}")
    print(f"Training complete. Best loss: {best_loss:.4e}")
    return {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "samples_total": int(n_samples),
        "samples_train": int(n_train),
        "samples_val": int(n_samples - n_train),
        "input_dim": int(in_dim),
        "output_dim": int(out_dim),
        "hidden_layers": tuple(int(v) for v in hidden_layers),
        "activation": str(activation),
        "dropout": float(dropout),
        "use_batchnorm": int(use_batchnorm),
        "loss": str(loss_name),
        "smoothl1_beta": float(smoothl1_beta),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "epochs_max": int(epochs),
        "epochs_ran": int(len(history["val"])),
        "best_epoch": int(best_epoch),
        "best_val_scaled_mse": float(best_loss),
        "last_val_scaled_mse": float(history["val"][-1]),
        "last_train_scaled_mse": float(history["train"][-1]),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7b: train ANN manifold map")
    parser.add_argument("--data-dir", type=str, default="stage_7_ann_data", help="Dataset directory.")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory for model/metadata/plots. Defaults to data-dir.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and training init.")
    parser.add_argument("--hidden-layers", type=str, default="128,256,256,256,128", help="CSV hidden layer widths.")
    parser.add_argument("--activation", type=str, default="elu", choices=["elu", "silu", "relu", "gelu", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability in hidden layers.")
    parser.add_argument("--no-batchnorm", action="store_true", help="Disable BatchNorm1d in hidden layers.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction in (0,0.5).")
    parser.add_argument("--epochs", type=int, default=2000, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience (epochs).")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--lr-patience", type=int, default=25, help="LR scheduler patience.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR scheduler factor.")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1"], help="Training loss.")
    parser.add_argument("--smoothl1-beta", type=float, default=1.0, help="Beta for SmoothL1 loss.")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Gradient clipping norm (0 disables).")
    args = parser.parse_args()
    train_ann(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        dropout=args.dropout,
        use_batchnorm=not args.no_batchnorm,
        val_fraction=args.val_fraction,
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
    )
