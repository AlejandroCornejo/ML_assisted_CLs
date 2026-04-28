import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
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

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=[128, 256, 256, 256, 128]):
        super().__init__()
        layers = []
        last_dim = in_dim
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.SiLU()) 
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
    def __init__(self, x_mean, x_std, y_mean, y_std, in_dim=3, out_dim=6):
        super().__init__()
        self.input_scaler = Scaler(x_mean, x_std)
        self.output_unscaler = Unscaler(y_mean, y_std)
        self.mlp = MLP(in_dim, out_dim)
        
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


def train_ann(data_dir="stage_7_ann_data", seed=42):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    x_file = "ann_input_train.npy" if os.path.exists(os.path.join(data_dir, "ann_input_train.npy")) else "q_p_train.npy"
    X = np.load(os.path.join(data_dir, x_file)).astype(np.float32)
    Y = np.load(os.path.join(data_dir, "q_s_train.npy")).astype(np.float32)
    cfg = _load_dataset_config(data_dir)
    in_dim = int(X.shape[1])
    out_dim = int(Y.shape[1])
    if cfg["input_dim"] is not None and cfg["input_dim"] != in_dim:
        raise ValueError(f"Dataset metadata/input mismatch: input_dim={cfg['input_dim']} vs X.shape[1]={in_dim}.")
    if cfg["n_secondary"] is not None and cfg["n_secondary"] != out_dim:
        raise ValueError(f"Dataset metadata/output mismatch: n_secondary={cfg['n_secondary']} vs Y.shape[1]={out_dim}.")
    
    # Split
    n_samples = X.shape[0]
    n_train = int(0.9 * n_samples)
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
    model = ManifoldANN(x_mean, x_std, y_mean, y_std, in_dim=in_dim, out_dim=out_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), 
                              batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), 
                            batch_size=1024)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=True)
    
    epochs = 2000
    patience = 80
    best_loss = float('inf')
    counter = 0
    
    history = {"train": [], "val": []}
    
    print("=" * 60)
    print("Stage 7b-ANN: Manifold MLP training (Normalized Loss)")
    print(f"  input_dim:  {in_dim} (n_primary={cfg['n_primary'] if cfg['n_primary'] is not None else in_dim})")
    print(f"  output_dim: {out_dim}")
    print("=" * 60)
    print(f"Training on device: {device}")
    print(f"Input file: {x_file} | samples: {n_samples}")
    if cfg["include_macro_strain_input"]:
        print("  Note: Input includes macro strain [Exx,Eyy,Gxy].")

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
            torch.save(model.state_dict(), os.path.join(data_dir, "manifold_ann.pt"))
            # Save metadata for inference
            np.savez(os.path.join(data_dir, "manifold_ann_metadata.npz"),
                     x_mean=x_mean, x_std=x_std,
                     y_mean=y_mean, y_std=y_std,
                     input_dim=np.array([in_dim], dtype=np.int64),
                     output_dim=np.array([out_dim], dtype=np.int64),
                     n_primary=np.array([cfg["n_primary"] if cfg["n_primary"] is not None else in_dim], dtype=np.int64),
                     n_secondary=np.array([cfg["n_secondary"] if cfg["n_secondary"] is not None else out_dim], dtype=np.int64),
                     include_macro_strain_input=np.array([1 if cfg["include_macro_strain_input"] else 0], dtype=np.int64))
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
    plt.savefig(os.path.join(data_dir, "training_history.png"), dpi=150)
    print(f"Training complete. Best loss: {best_loss:.4e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7b: train ANN manifold map")
    parser.add_argument("--data-dir", type=str, default="stage_7_ann_data", help="Dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and training init.")
    args = parser.parse_args()
    train_ann(data_dir=args.data_dir, seed=args.seed)
