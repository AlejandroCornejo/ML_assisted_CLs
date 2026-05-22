#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 4 (2D-MAWECM): train sparse anisotropic GP decoder q_m -> q_s.

This stage is independent from MAW-ECM weights:
- MAW-ECM weights remain RBF-based in Stage8b.
- Here we only replace the PROM decoder map with sparse GP (ARD RBF kernel).
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

from sparse_gp_manifold_model import save_sparse_gp_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage4: train sparse PROM-GPR from Stage3 dataset")
    p.add_argument(
        "--dataset-file",
        type=str,
        default="stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    )
    p.add_argument("--out-dir", type=str, default="stage_4_prom_gpr_sparse")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--gpr-input-space",
        type=str,
        default="q_m",
        choices=["q_m", "mu"],
        help="Decoder input: q_m (recommended) or mu.",
    )
    p.add_argument(
        "--gpr-target-space",
        type=str,
        default="q_s",
        choices=["q_s"],
        help="Decoder target. For master/slave PROM this must be q_s.",
    )

    p.add_argument("--num-inducing", type=int, default=451)
    p.add_argument(
        "--inducing-selection",
        type=str,
        default="kmeans",
        choices=["random", "kmeans"],
    )
    p.add_argument("--kmeans-max-iters", type=int, default=40)
    p.add_argument("--kmeans-batch-size", type=int, default=4096)
    p.add_argument("--kmeans-fit-samples", type=int, default=40000)

    p.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="If >0, random subset size used from Stage3 train split (0 uses full train split).",
    )
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--min-noise", type=float, default=1e-6)
    p.add_argument("--fixed-inducing", action="store_true", help="Disable inducing-point updates.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(y_true), 1e-30))
    rmse = float(np.sqrt(np.mean(err * err)))
    max_abs = float(np.max(np.abs(err)))
    rel_l2_per_comp = np.linalg.norm(err, axis=0) / np.maximum(np.linalg.norm(y_true, axis=0), 1e-30)
    rmse_per_comp = np.sqrt(np.mean(err * err, axis=0))
    return {
        "rel_l2": rel_l2,
        "rmse": rmse,
        "max_abs": max_abs,
        "rel_l2_per_comp": [float(x) for x in rel_l2_per_comp],
        "rmse_per_comp": [float(x) for x in rmse_per_comp],
    }


def _standardize_from_train(
    x_all: np.ndarray,
    y_all: np.ndarray,
    idx_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray(x_all[idx_train], dtype=float)
    y_train = np.asarray(y_all[idx_train], dtype=float)
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std = np.where(x_std < 1e-14, 1.0, x_std)
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)
    y_std = np.where(y_std < 1e-14, 1.0, y_std)
    x_z = (np.asarray(x_all, dtype=float) - x_mean[None, :]) / x_std[None, :]
    y_z = (np.asarray(y_all, dtype=float) - y_mean[None, :]) / y_std[None, :]
    return x_z, y_z, x_mean, x_std, y_mean, y_std


def _choose_inducing_points(
    x_train: np.ndarray,
    num_inducing: int,
    method: str,
    seed: int,
    kmeans_max_iters: int,
    kmeans_batch_size: int,
    kmeans_fit_samples: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = int(x_train.shape[0])
    m = int(num_inducing)
    if m <= 0:
        raise ValueError("num_inducing must be > 0.")
    if m > n:
        raise ValueError(f"num_inducing={m} exceeds available train samples={n}.")

    mode = str(method).strip().lower()
    if mode == "random":
        idx = rng.choice(n, size=m, replace=False)
        return np.asarray(x_train[idx, :], dtype=np.float64)
    if mode != "kmeans":
        raise ValueError(f"Unsupported inducing-selection='{method}'.")

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as ex:
        raise RuntimeError(
            "kmeans inducing selection requires scikit-learn. "
            "Install scikit-learn or use --inducing-selection random."
        ) from ex

    n_fit = int(kmeans_fit_samples)
    if n_fit <= 0 or n_fit > n:
        n_fit = n
    if n_fit < m:
        n_fit = m

    if n_fit < n:
        idx_fit = rng.choice(n, size=n_fit, replace=False)
        x_fit = x_train[idx_fit, :]
    else:
        x_fit = x_train

    print(
        f"[Sparse-GP] Inducing selection: kmeans_minibatch | fit_samples={x_fit.shape[0]} | "
        f"centers={m} | batch_size={int(kmeans_batch_size)} | iters={int(kmeans_max_iters)}"
    )
    km = MiniBatchKMeans(
        n_clusters=m,
        random_state=int(seed),
        batch_size=int(max(32, kmeans_batch_size)),
        max_iter=int(max(1, kmeans_max_iters)),
        n_init="auto",
        verbose=0,
    )
    km.fit(x_fit)
    return np.asarray(km.cluster_centers_, dtype=np.float64)


def _fit_sparse_gp_output(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    inducing_init: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    min_noise: float,
    learn_inducing: bool,
    device: str,
    seed: int,
) -> Dict[str, object]:
    try:
        import torch
        import gpytorch
    except Exception as ex:
        raise RuntimeError(
            "True sparse-GP training requires torch + gpytorch."
        ) from ex

    torch.manual_seed(int(seed))
    dtype = torch.float64
    dev = torch.device(device)

    xtr = torch.as_tensor(x_train, dtype=dtype, device=dev)
    ytr = torch.as_tensor(y_train, dtype=dtype, device=dev)
    xva = torch.as_tensor(x_val, dtype=dtype, device=dev) if x_val is not None else None
    yva = torch.as_tensor(y_val, dtype=dtype, device=dev) if y_val is not None else None

    inducing = torch.as_tensor(inducing_init, dtype=dtype, device=dev).clone()

    class _SVGP(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            vd = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
            vs = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                vd,
                learn_inducing_locations=bool(learn_inducing),
            )
            super().__init__(vs)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )

    model = _SVGP(inducing).to(dev).double()
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(float(min_noise))
    ).to(dev).double()

    model.train()
    likelihood.train()
    params = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=xtr.size(0))

    n_train = int(xtr.size(0))
    bs = int(max(8, batch_size))
    n_batches = max(1, int(np.ceil(n_train / bs)))
    best_state = None
    best_score = np.inf
    train_hist = []
    val_hist = []

    for ep in range(1, int(max(1, epochs)) + 1):
        perm = torch.randperm(n_train, device=dev)
        ep_loss = 0.0

        for b in range(n_batches):
            i0 = b * bs
            i1 = min((b + 1) * bs, n_train)
            idx = perm[i0:i1]
            xb = xtr[idx]
            yb = ytr[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = -mll(out, yb)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item()) * float(i1 - i0)

        ep_loss /= float(n_train)
        train_hist.append(ep_loss)

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            if xva is not None and yva is not None:
                pred = model(xva).mean
                val_mse = float(torch.mean((pred - yva) ** 2).item())
            else:
                pred = model(xtr).mean
                val_mse = float(torch.mean((pred - ytr) ** 2).item())
        val_hist.append(val_mse)
        if val_mse < best_score:
            best_score = val_mse
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "likelihood": {k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()},
            }

        if ep == 1 or ep % 10 == 0 or ep == int(epochs):
            print(
                f"      epoch {ep:04d}/{int(epochs):04d} | "
                f"train_loss={ep_loss:.6e} | val_mse={val_mse:.6e}"
            )
        model.train()
        likelihood.train()

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        z_dev = model.variational_strategy.inducing_points.detach()
        m_white = model.variational_strategy.variational_distribution.mean.detach()
        ls = model.covar_module.base_kernel.lengthscale.detach().reshape(-1)
        oscale = float(model.covar_module.outputscale.detach().cpu().item())
        noise = float(likelihood.noise.detach().cpu().item())

        kzz = model.covar_module(z_dev).to_dense().detach()
        jitter = max(float(min_noise), 1.0e-12)
        kzz = kzz + jitter * torch.eye(kzz.size(0), dtype=kzz.dtype, device=kzz.device)
        L = torch.linalg.cholesky(kzz)
        alpha = torch.linalg.solve_triangular(L.T, m_white.unsqueeze(-1), upper=True).squeeze(-1)

    return {
        "inducing_points": np.asarray(z_dev.cpu().numpy(), dtype=np.float64),
        "alpha": np.asarray(alpha.cpu().numpy(), dtype=np.float64),
        "lengthscales": np.asarray(ls.cpu().numpy(), dtype=np.float64),
        "outputscale": float(oscale),
        "noise": float(noise),
        "train_history": np.asarray(train_hist, dtype=np.float64),
        "val_history": np.asarray(val_hist, dtype=np.float64),
        "best_val_mse": float(best_score),
    }


def _predict_analytic_batch(x_scaled: np.ndarray, model_payload: Dict[str, object]) -> np.ndarray:
    z = np.asarray(model_payload["inducing_points"], dtype=float)
    alpha = np.asarray(model_payload["alpha"], dtype=float)
    ls = np.asarray(model_payload["lengthscales"], dtype=float)
    oscale = float(model_payload["outputscale"])

    inv_l2 = 1.0 / (ls * ls)
    x2 = np.sum((x_scaled * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True)
    z2 = np.sum((z * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True).T
    cross = (x_scaled * inv_l2[None, :]) @ z.T
    sq = np.maximum(x2 + z2 - 2.0 * cross, 0.0)
    k = oscale * np.exp(-0.5 * sq)
    return k @ alpha


def _resolve_device(device_arg: str) -> str:
    d = str(device_arg).strip().lower()
    if d in ("cpu", "cuda"):
        return d
    if d != "auto":
        raise ValueError("--device must be one of: auto, cpu, cuda")
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.lr <= 0.0:
        raise ValueError("--lr must be > 0.")
    if args.min_noise <= 0.0:
        raise ValueError("--min-noise must be > 0.")

    data = np.load(args.dataset_file, allow_pickle=True)
    if "q_m_all" not in data or "q_s_all" not in data:
        raise RuntimeError("Stage3 dataset must contain q_m_all and q_s_all.")

    q_m_all = np.asarray(data["q_m_all"], dtype=float)
    q_s_all = np.asarray(data["q_s_all"], dtype=float)
    mu_all = np.asarray(data["mu_all"], dtype=float)
    idx_train = np.asarray(data["idx_train"], dtype=np.int64)
    idx_val = np.asarray(data["idx_val"], dtype=np.int64)
    idx_test = np.asarray(data["idx_test"], dtype=np.int64)
    mu_space = str(np.ravel(data["mu_space"])[0]) if "mu_space" in data else "gx_gxy"
    target_space = str(np.ravel(data["target_space"])[0]) if "target_space" in data else "q_s"
    mapping = str(np.ravel(data["mapping"])[0]) if "mapping" in data else "green_lagrange_upper"

    input_space = str(args.gpr_input_space).strip().lower()
    if input_space == "q_m":
        x_all = q_m_all
    elif input_space == "mu":
        x_all = mu_all
    else:
        raise RuntimeError(f"Unsupported gpr-input-space='{args.gpr_input_space}'.")
    y_all = q_s_all

    train_pool = idx_train.copy()
    n_pool = int(train_pool.size)
    n_use = int(args.train_samples)
    if n_use <= 0 or n_use > n_pool:
        n_use = n_pool
    rng = np.random.default_rng(int(args.seed))
    if n_use < n_pool:
        idx_train_eff = np.sort(rng.choice(train_pool, size=n_use, replace=False).astype(np.int64))
    else:
        idx_train_eff = np.sort(train_pool)

    x_z, y_z, x_mean, x_std, y_mean, y_std = _standardize_from_train(
        x_all=x_all,
        y_all=y_all,
        idx_train=idx_train_eff,
    )
    x_train = x_z[idx_train_eff]
    y_train = y_z[idx_train_eff]
    x_val = x_z[idx_val] if idx_val.size > 0 else None
    y_val = y_z[idx_val] if idx_val.size > 0 else None

    in_dim = int(x_train.shape[1])
    out_dim = int(y_train.shape[1])
    n_primary = int(q_m_all.shape[1])
    n_secondary = int(q_s_all.shape[1])
    if out_dim != n_secondary:
        raise RuntimeError(f"Output dim mismatch: out_dim={out_dim}, n_secondary={n_secondary}.")

    if int(args.num_inducing) > int(x_train.shape[0]):
        raise RuntimeError(
            f"num_inducing={int(args.num_inducing)} exceeds effective train samples={int(x_train.shape[0])}."
        )

    device = _resolve_device(args.device)
    print("=" * 72)
    print("Stage 4: PROM sparse-GPR training")
    print("=" * 72)
    print(f"dataset_file : {args.dataset_file}")
    print(f"out_dir      : {args.out_dir}")
    print(f"input_space  : {input_space}")
    print(f"target_space : {args.gpr_target_space}")
    print(f"train/val/test sizes: {idx_train_eff.size}/{idx_val.size}/{idx_test.size}")
    print(f"dim(input)={in_dim}, dim(target)={out_dim}, n_primary={n_primary}, n_secondary={n_secondary}")
    print(f"device       : {device}")
    print(f"num_inducing : {int(args.num_inducing)} ({args.inducing_selection})")

    inducing_init = _choose_inducing_points(
        x_train=x_train,
        num_inducing=int(args.num_inducing),
        method=args.inducing_selection,
        seed=int(args.seed),
        kmeans_max_iters=int(args.kmeans_max_iters),
        kmeans_batch_size=int(args.kmeans_batch_size),
        kmeans_fit_samples=int(args.kmeans_fit_samples),
    )

    all_inducing = np.zeros((out_dim, int(args.num_inducing), in_dim), dtype=np.float64)
    all_alpha = np.zeros((out_dim, int(args.num_inducing)), dtype=np.float64)
    all_ls = np.zeros((out_dim, in_dim), dtype=np.float64)
    all_os = np.zeros((out_dim,), dtype=np.float64)
    all_noise = np.zeros((out_dim,), dtype=np.float64)
    best_val_mse = np.zeros((out_dim,), dtype=np.float64)
    train_histories = []
    val_histories = []

    for j in range(out_dim):
        print(f"\n[Sparse-GP] Training output {j + 1}/{out_dim}")
        stats = _fit_sparse_gp_output(
            x_train=x_train,
            y_train=y_train[:, j],
            x_val=x_val,
            y_val=None if y_val is None else y_val[:, j],
            inducing_init=inducing_init,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            min_noise=float(args.min_noise),
            learn_inducing=(not bool(args.fixed_inducing)),
            device=device,
            seed=int(args.seed + 997 * (j + 1)),
        )
        all_inducing[j, :, :] = stats["inducing_points"]
        all_alpha[j, :] = stats["alpha"]
        all_ls[j, :] = stats["lengthscales"]
        all_os[j] = float(stats["outputscale"])
        all_noise[j] = float(stats["noise"])
        best_val_mse[j] = float(stats["best_val_mse"])
        train_histories.append(stats["train_history"])
        val_histories.append(stats["val_history"])

    # Evaluate exported analytic map
    y_pred_z = np.zeros_like(y_z)
    for j in range(out_dim):
        local_model = {
            "inducing_points": all_inducing[j, :, :],
            "alpha": all_alpha[j, :],
            "lengthscales": all_ls[j, :],
            "outputscale": all_os[j],
        }
        y_pred_z[:, j] = _predict_analytic_batch(x_z, local_model)
    y_pred = y_pred_z * y_std[None, :] + y_mean[None, :]

    m_train = _compute_metrics(y_all[idx_train_eff], y_pred[idx_train_eff])
    m_val = _compute_metrics(y_all[idx_val], y_pred[idx_val]) if idx_val.size > 0 else None
    m_test = _compute_metrics(y_all[idx_test], y_pred[idx_test]) if idx_test.size > 0 else None

    model_dict = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "inducing_points": all_inducing,
        "alpha": all_alpha,
        "lengthscales": all_ls,
        "outputscales": all_os,
        "noise": all_noise,
        "input_dim": int(in_dim),
        "output_dim": int(out_dim),
        "n_primary": int(n_primary),
        "n_secondary": int(n_secondary),
        "kernel_name": "rbf",
        "model_family": "sparse_gp",
        "num_inducing": int(args.num_inducing),
        "backend": "gpytorch",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "train_samples_used": int(idx_train_eff.size),
        "loss_history": np.asarray([h[-1] if len(h) else np.nan for h in val_histories], dtype=float),
        "include_macro_strain_input": False,
    }

    save_sparse_gp_model(os.path.join(args.out_dir, "sparse_gp_model.npz"), model_dict)

    np.savez(
        os.path.join(args.out_dir, "sparse_gp_meta.npz"),
        input_space=np.array([input_space], dtype=object),
        target_space=np.array([str(args.gpr_target_space)], dtype=object),
        x_mean=np.asarray(x_mean, dtype=float),
        x_std=np.asarray(x_std, dtype=float),
        y_mean=np.asarray(y_mean, dtype=float),
        y_std=np.asarray(y_std, dtype=float),
        q_m_dim=np.array([int(n_primary)], dtype=np.int64),
        q_s_dim=np.array([int(n_secondary)], dtype=np.int64),
        mu_space=np.array([mu_space], dtype=object),
        mapping=np.array([mapping], dtype=object),
        stage3_target_space=np.array([target_space], dtype=object),
    )

    np.savez(
        os.path.join(args.out_dir, "predictions_sparse_gpr.npz"),
        y_pred_all=np.asarray(y_pred, dtype=float),
        y_true_all=np.asarray(y_all, dtype=float),
        idx_train=np.asarray(idx_train_eff, dtype=np.int64),
        idx_val=np.asarray(idx_val, dtype=np.int64),
        idx_test=np.asarray(idx_test, dtype=np.int64),
    )

    summary = {
        "dataset_file": args.dataset_file,
        "out_dir": args.out_dir,
        "input_space": input_space,
        "target_space": str(args.gpr_target_space),
        "dim": {"input": int(in_dim), "target": int(out_dim)},
        "sizes": {
            "train_pool": int(train_pool.size),
            "train_used": int(idx_train_eff.size),
            "val": int(idx_val.size),
            "test": int(idx_test.size),
        },
        "sparse_gp": {
            "num_inducing": int(args.num_inducing),
            "inducing_selection": str(args.inducing_selection),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "min_noise": float(args.min_noise),
            "learn_inducing": int(not bool(args.fixed_inducing)),
            "device": device,
            "mean_best_val_mse_per_output": float(np.nanmean(best_val_mse)),
            "train": m_train,
            "val": m_val,
            "test": m_test,
        },
    }

    with open(os.path.join(args.out_dir, "stage4_sparse_gpr_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "stage4_sparse_gpr_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_rel = float(m_train["rel_l2"])
    val_rel = float(m_val["rel_l2"]) if m_val is not None else float("nan")
    test_rel = float(m_test["rel_l2"]) if m_test is not None else float("nan")
    print(f"[INFO] relL2 train/val/test = {train_rel:.3e} / {val_rel:.3e} / {test_rel:.3e}")
    print(f"[OK] Stage 4 sparse-GPR outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
