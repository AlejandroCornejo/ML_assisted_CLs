#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn


class ZScoreScaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std = np.asarray(std, dtype=np.float32).reshape(-1)
        std = np.where(std > float(eps), std, 1.0).astype(np.float32)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class ZScoreUnscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std = np.asarray(std, dtype=np.float32).reshape(-1)
        std = np.where(std > float(eps), std, 1.0).astype(np.float32)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


class MinMaxScaler(nn.Module):
    def __init__(self, x_min, x_max, eps=1e-12):
        super().__init__()
        x_min = np.asarray(x_min, dtype=np.float32).reshape(-1)
        x_max = np.asarray(x_max, dtype=np.float32).reshape(-1)
        center = 0.5 * (x_max + x_min)
        half_range = 0.5 * (x_max - x_min)
        half_range = np.where(half_range > float(eps), half_range, 1.0).astype(np.float32)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("half_range", torch.tensor(half_range, dtype=torch.float32))

    def forward(self, x):
        return (x - self.center) / self.half_range


class MinMaxUnscaler(nn.Module):
    def __init__(self, x_min, x_max, eps=1e-12):
        super().__init__()
        x_min = np.asarray(x_min, dtype=np.float32).reshape(-1)
        x_max = np.asarray(x_max, dtype=np.float32).reshape(-1)
        center = 0.5 * (x_max + x_min)
        half_range = 0.5 * (x_max - x_min)
        half_range = np.where(half_range > float(eps), half_range, 1.0).astype(np.float32)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("half_range", torch.tensor(half_range, dtype=torch.float32))

    def forward(self, y):
        return y * self.half_range + self.center


def _activation_module(name):
    key = str(name).strip().lower()
    if key == "elu":
        return nn.ELU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'. Use one of: elu, tanh, silu.")


def build_mlp(in_dim, hidden_dims, out_dim, activation="elu"):
    dims = [int(in_dim)] + [int(v) for v in hidden_dims] + [int(out_dim)]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(_activation_module(activation))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PODDLAutoencoder(nn.Module):
    """
    POD-level autoencoder:
        q_raw -> q_norm -> z -> q_norm_hat -> q_raw_hat

    Scaling is embedded in the model for robust online/offline consistency.
    """

    def __init__(
        self,
        q_dim,
        latent_dim,
        hidden_dims=(128, 64, 32),
        scaling="zscore",
        activation="elu",
        q_mean=None,
        q_std=None,
        q_min=None,
        q_max=None,
    ):
        super().__init__()
        q_dim = int(q_dim)
        latent_dim = int(latent_dim)
        hidden_dims = tuple(int(v) for v in hidden_dims)
        if q_dim < 1 or latent_dim < 1:
            raise ValueError(f"Invalid dimensions: q_dim={q_dim}, latent_dim={latent_dim}.")

        self.q_dim = int(q_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = hidden_dims
        self.scaling = str(scaling).strip().lower()
        self.activation = str(activation).strip().lower()

        if self.scaling == "zscore":
            if q_mean is None:
                q_mean = np.zeros(q_dim, dtype=np.float32)
            if q_std is None:
                q_std = np.ones(q_dim, dtype=np.float32)
            self.scaler = ZScoreScaler(q_mean, q_std)
            self.unscaler = ZScoreUnscaler(q_mean, q_std)
        elif self.scaling in ("minmax", "minmax_-1_1"):
            self.scaling = "minmax_-1_1"
            if q_min is None:
                q_min = np.zeros(q_dim, dtype=np.float32)
            if q_max is None:
                q_max = np.ones(q_dim, dtype=np.float32)
            self.scaler = MinMaxScaler(q_min, q_max)
            self.unscaler = MinMaxUnscaler(q_min, q_max)
        else:
            raise ValueError("Unsupported scaling. Use 'zscore' or 'minmax'.")

        self.encoder = build_mlp(q_dim, hidden_dims, latent_dim, activation=self.activation)
        self.decoder = build_mlp(latent_dim, tuple(reversed(hidden_dims)), q_dim, activation=self.activation)

    def forward(self, q_raw):
        q_norm = self.scaler(q_raw)
        z = self.encoder(q_norm)
        q_norm_hat = self.decoder(z)
        q_raw_hat = self.unscaler(q_norm_hat)
        return q_raw_hat

    def encode(self, q_raw):
        q_norm = self.scaler(q_raw)
        return self.encoder(q_norm)

    def decode_from_latent(self, z):
        q_norm_hat = self.decoder(z)
        return self.unscaler(q_norm_hat)


def _infer_scaling_from_state_dict(state_dict, fallback="zscore"):
    keys = set(state_dict.keys())
    if any(k.startswith("scaler.mean") for k in keys):
        return "zscore"
    if any(k.startswith("scaler.center") for k in keys):
        return "minmax_-1_1"
    return str(fallback)


def _default_activation_for_scaling(scaling):
    if str(scaling).strip().lower() == "zscore":
        return "elu"
    return "tanh"


def build_model_from_checkpoint(checkpoint):
    state_dict = checkpoint["state_dict"]

    q_dim = int(checkpoint["q_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    hidden_dims = tuple(int(v) for v in checkpoint.get("hidden_dims", (128, 64, 32)))
    scaling_ckpt = checkpoint.get("scaling", "zscore")
    scaling = _infer_scaling_from_state_dict(state_dict, fallback=scaling_ckpt)
    activation = str(checkpoint.get("activation", _default_activation_for_scaling(scaling)))

    q_mean = checkpoint.get("q_mean", None)
    q_std = checkpoint.get("q_std", None)
    q_min = checkpoint.get("q_min", None)
    q_max = checkpoint.get("q_max", None)

    model = PODDLAutoencoder(
        q_dim=q_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        scaling=scaling,
        activation=activation,
        q_mean=q_mean,
        q_std=q_std,
        q_min=q_min,
        q_max=q_max,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_pod_dl_checkpoint(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint '{model_path}' does not contain key 'state_dict'.")

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    return model, checkpoint, device


def load_pod_dl_model(model_dir="stage_7_pod_dl_data", model_filename="pod_dl_autoencoder.pt", device=None):
    model_path = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"POD-DL checkpoint not found: {model_path}")
    return load_pod_dl_checkpoint(model_path, device=device)
