#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 7b (true sparse-GP): train sparse Gaussian-process manifold q_p -> q_s.

This script trains a pure sparse variational GP model (per output component), then
exports an analytic online model to `sparse_gp_model.npz` for PROM/HPROM usage.

Online evaluator uses exact closed-form expressions:
  q_s_j(x) = sum_m alpha_{j,m} * k_j(x, z_{j,m})
  d q_s_j / d x_i = sum_m alpha_{j,m} * d k_j / d x_i
with ARD RBF kernels.
"""

import argparse
import os
import numpy as np

from sparse_gp_manifold_model import save_sparse_gp_model


DEFAULT_DATA_DIR = "stage_7_ann_data"
DEFAULT_OUT_DIR = "stage_7_gpr_data"


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

    meta = np.load(meta_path, allow_pickle=True)
    if "n_primary" in meta:
        cfg["n_primary"] = int(np.ravel(meta["n_primary"])[0])
    if "n_secondary" in meta:
        cfg["n_secondary"] = int(np.ravel(meta["n_secondary"])[0])
    if "input_dim" in meta:
        cfg["input_dim"] = int(np.ravel(meta["input_dim"])[0])
    if "include_macro_strain_input" in meta:
        cfg["include_macro_strain_input"] = bool(int(np.ravel(meta["include_macro_strain_input"])[0]))
    return cfg


def _load_dataset(data_dir):
    x_file = "ann_input_train.npy" if os.path.exists(os.path.join(data_dir, "ann_input_train.npy")) else "q_p_train.npy"
    x = np.load(os.path.join(data_dir, x_file)).astype(np.float64)
    y = np.load(os.path.join(data_dir, "q_s_train.npy")).astype(np.float64)
    phi_p = np.load(os.path.join(data_dir, "phi_p.npy"))
    phi_s = np.load(os.path.join(data_dir, "phi_s.npy"))

    cfg = _load_dataset_config(data_dir)
    n_primary = int(cfg["n_primary"] if cfg["n_primary"] is not None else phi_p.shape[1])
    n_secondary = int(cfg["n_secondary"] if cfg["n_secondary"] is not None else phi_s.shape[1])
    include_macro = bool(cfg["include_macro_strain_input"]) if cfg["input_dim"] is not None else False

    if include_macro:
        raise RuntimeError(
            "Dataset was built with macro-strain manifold inputs (N(q,mu)); "
            "pure sparse-GP branch requires q_p-only inputs."
        )

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D arrays. Got x={x.shape}, y={y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: x has {x.shape[0]}, y has {y.shape[0]}.")

    if x.shape[1] != n_primary:
        raise ValueError(
            f"Input dim mismatch: x has {x.shape[1]}, metadata n_primary={n_primary}."
        )
    if y.shape[1] != n_secondary:
        raise ValueError(
            f"Output dim mismatch: y has {y.shape[1]}, metadata n_secondary={n_secondary}."
        )

    return x, y, phi_p, phi_s, x_file, n_primary, n_secondary


def _standardize(x, y):
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)

    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)

    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std
    return x_scaled, y_scaled, x_mean, x_std, y_mean, y_std


def _build_optional_mu_qp_affine(data_dir, n_primary):
    mu_path = os.path.join(data_dir, "ls_targets_train.npy")
    if not os.path.exists(mu_path):
        return None

    qp_path = os.path.join(data_dir, "q_p_train.npy")
    if os.path.exists(qp_path):
        qp = np.load(qp_path).astype(np.float64)
    else:
        x_file = "ann_input_train.npy" if os.path.exists(os.path.join(data_dir, "ann_input_train.npy")) else "q_p_train.npy"
        qp = np.load(os.path.join(data_dir, x_file)).astype(np.float64)

    mu = np.load(mu_path).astype(np.float64)
    if mu.ndim != 2 or qp.ndim != 2:
        return None
    if mu.shape[0] != qp.shape[0]:
        return None
    if qp.shape[1] < int(n_primary):
        return None

    mu_dim = int(min(3, mu.shape[1]))
    if mu_dim <= 0:
        return None
    x = np.asarray(mu[:, :mu_dim], dtype=float)
    y = np.asarray(qp[:, : int(n_primary)], dtype=float)
    x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=float)])
    b_aff, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    y_hat = x_aug @ b_aff
    rel = float(np.linalg.norm(y_hat - y) / max(np.linalg.norm(y), 1e-30))
    return {
        "b_aff": np.asarray(b_aff, dtype=float),
        "mu_dim": int(mu_dim),
        "qp_dim": int(n_primary),
        "rel_fit": rel,
        "n_samples": int(x.shape[0]),
    }


def _choose_inducing_points(
    x_train,
    num_inducing,
    method,
    seed,
    kmeans_max_iters,
    kmeans_batch_size,
    kmeans_fit_samples,
):
    rng = np.random.default_rng(int(seed))
    n = int(x_train.shape[0])
    m = int(num_inducing)
    if m <= 0:
        raise ValueError("num_inducing must be > 0.")
    if n < m:
        raise ValueError(f"Need at least num_inducing samples ({m}), got {n}.")

    mode = str(method).strip().lower()
    if mode in ("random", "rand"):
        idx = rng.choice(n, size=m, replace=False)
        return np.asarray(x_train[idx, :], dtype=np.float64)

    if mode not in ("kmeans", "k-means", "minibatch_kmeans", "mini_batch_kmeans"):
        raise ValueError(f"Unsupported inducing selection mode '{method}'.")

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as ex:
        raise RuntimeError(
            "kmeans inducing selection requested but scikit-learn is not available. "
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

    kmeans = MiniBatchKMeans(
        n_clusters=m,
        random_state=int(seed),
        batch_size=int(max(32, kmeans_batch_size)),
        max_iter=int(max(1, kmeans_max_iters)),
        n_init="auto",
        verbose=0,
    )
    kmeans.fit(x_fit)
    return np.asarray(kmeans.cluster_centers_, dtype=np.float64)


def _fit_sparse_gp_output(
    x_train,
    y_train,
    x_val,
    y_val,
    inducing_init,
    epochs,
    batch_size,
    lr,
    weight_decay,
    min_noise,
    learn_inducing,
    device,
    seed,
):
    try:
        import torch
        import gpytorch
    except Exception as ex:
        raise RuntimeError(
            "True sparse-GP training requires torch + gpytorch. "
            "Install them before running Stage 7b sparse-GP."
        ) from ex

    torch.manual_seed(int(seed))

    dtype = torch.float64
    dev = torch.device(device)

    xtr = torch.as_tensor(x_train, dtype=dtype, device=dev)
    ytr = torch.as_tensor(y_train, dtype=dtype, device=dev)
    xva = None
    yva = None
    if x_val is not None and y_val is not None and len(x_val) > 0:
        xva = torch.as_tensor(x_val, dtype=dtype, device=dev)
        yva = torch.as_tensor(y_val, dtype=dtype, device=dev)

    inducing = torch.as_tensor(inducing_init, dtype=dtype, device=dev).clone()

    class _SVGPRegressor(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=bool(learn_inducing),
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    model = _SVGPRegressor(inducing)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(float(min_noise))
    )

    model = model.to(dev).double()
    likelihood = likelihood.to(dev).double()

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
    train_history = []
    val_history = []

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
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item()) * float(i1 - i0)

        ep_loss /= float(n_train)
        train_history.append(ep_loss)

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            if xva is not None:
                pred = model(xva).mean
                val_mse = torch.mean((pred - yva) ** 2).item()
            else:
                pred = model(xtr).mean
                val_mse = torch.mean((pred - ytr) ** 2).item()
        val_history.append(float(val_mse))

        score = float(val_mse)
        if score < best_score:
            best_score = score
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "likelihood": {k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()},
            }

        if ep == 1 or ep % 10 == 0 or ep == int(epochs):
            print(
                f"      epoch {ep:04d}/{int(epochs):04d} | train_loss={ep_loss:.6e} | val_mse={val_mse:.6e}"
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
        m_white_dev = model.variational_strategy.variational_distribution.mean.detach()
        lengthscales_dev = model.covar_module.base_kernel.lengthscale.detach().reshape(-1)
        outputscale = float(model.covar_module.outputscale.detach().cpu().item())
        noise = float(likelihood.noise.detach().cpu().item())

        # Predictive mean in GPyTorch VariationalStrategy is:
        #   k(x,Z) Kzz^{-1/2} m_white
        # with Kzz = k(Z,Z). If Kzz = L L^T, then alpha = L^{-T} m_white.
        kzz_dev = model.covar_module(z_dev).to_dense().detach()
        jitter = max(float(min_noise), 1.0e-12)
        kzz_dev = kzz_dev + jitter * torch.eye(
            kzz_dev.size(0), dtype=kzz_dev.dtype, device=kzz_dev.device
        )
        L_dev = torch.linalg.cholesky(kzz_dev)
        alpha_dev = torch.linalg.solve_triangular(
            L_dev.T, m_white_dev.unsqueeze(-1), upper=True
        ).squeeze(-1)

        z = z_dev.cpu().numpy()
        alpha = alpha_dev.cpu().numpy()
        lengthscales = lengthscales_dev.cpu().numpy()

    return {
        "inducing_points": np.asarray(z, dtype=np.float64),
        "alpha": np.asarray(alpha, dtype=np.float64),
        "lengthscales": np.asarray(lengthscales, dtype=np.float64),
        "outputscale": float(outputscale),
        "noise": float(noise),
        "train_history": np.asarray(train_history, dtype=np.float64),
        "val_history": np.asarray(val_history, dtype=np.float64),
        "best_val_mse": float(best_score),
    }


def _predict_analytic_batch(x_scaled, model_payload):
    # x_scaled: (n, d)
    z = model_payload["inducing_points"]
    alpha = model_payload["alpha"]
    ls = model_payload["lengthscales"]
    oscale = float(model_payload["outputscale"])

    inv_l2 = 1.0 / (ls * ls)
    # Pairwise RBF
    # k(x,z)=os*exp(-0.5*sum((x-z)^2/l^2))
    x2 = np.sum((x_scaled * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True)
    z2 = np.sum((z * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True).T
    cross = (x_scaled * inv_l2[None, :]) @ z.T
    sq = np.maximum(x2 + z2 - 2.0 * cross, 0.0)
    k = oscale * np.exp(-0.5 * sq)
    return k @ alpha


def _resolve_device(device_arg):
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


def train_sparse_gp(
    data_dir=DEFAULT_DATA_DIR,
    out_dir=DEFAULT_OUT_DIR,
    num_inducing=800,
    inducing_selection="kmeans",
    kmeans_max_iters=40,
    kmeans_batch_size=4096,
    kmeans_fit_samples=40000,
    train_samples=0,
    val_fraction=0.1,
    epochs=120,
    batch_size=2048,
    lr=0.05,
    weight_decay=0.0,
    min_noise=1e-6,
    learn_inducing=True,
    device="auto",
    seed=42,
):
    os.makedirs(out_dir, exist_ok=True)

    x, y, phi_p, phi_s, x_file, n_primary, n_secondary = _load_dataset(data_dir)
    x_scaled, y_scaled, x_mean, x_std, y_mean, y_std = _standardize(x, y)

    rng = np.random.default_rng(int(seed))
    n_total = int(x_scaled.shape[0])

    n_use = int(train_samples)
    if n_use <= 0 or n_use > n_total:
        n_use = n_total

    if n_use < n_total:
        idx_use = rng.choice(n_total, size=n_use, replace=False)
        x_use = x_scaled[idx_use, :]
        y_use = y_scaled[idx_use, :]
    else:
        idx_use = np.arange(n_total, dtype=np.int64)
        x_use = x_scaled
        y_use = y_scaled

    n_val = int(np.floor(float(val_fraction) * float(n_use)))
    n_val = max(0, min(n_val, n_use - 1))

    perm = rng.permutation(n_use)
    if n_val > 0:
        idx_val = perm[:n_val]
        idx_train = perm[n_val:]
        x_val = x_use[idx_val, :]
        y_val = y_use[idx_val, :]
    else:
        idx_train = perm
        x_val = None
        y_val = None

    x_train = x_use[idx_train, :]
    y_train = y_use[idx_train, :]

    in_dim = int(x_train.shape[1])
    out_dim = int(y_train.shape[1])

    if int(num_inducing) > int(x_train.shape[0]):
        raise RuntimeError(
            f"num_inducing={int(num_inducing)} exceeds training samples={int(x_train.shape[0])}."
        )

    dev = _resolve_device(device)
    print("=" * 72)
    print("Stage 7b Sparse-GP manifold training (true sparse GP, analytic export)")
    print("=" * 72)
    print(f"data_dir: {data_dir}")
    print(f"out_dir:  {out_dir}")
    print(f"input file: {x_file}")
    print(f"samples: total={n_total}, used={n_use}, train={x_train.shape[0]}, val={0 if x_val is None else x_val.shape[0]}")
    print(f"dims: input={in_dim} (n_primary={n_primary}), output={out_dim} (n_secondary={n_secondary})")
    print(f"device: {dev}")
    print(f"inducing: m={int(num_inducing)}, selection={inducing_selection}")

    inducing_init = _choose_inducing_points(
        x_train=x_train,
        num_inducing=int(num_inducing),
        method=inducing_selection,
        seed=int(seed),
        kmeans_max_iters=int(kmeans_max_iters),
        kmeans_batch_size=int(kmeans_batch_size),
        kmeans_fit_samples=int(kmeans_fit_samples),
    )

    all_inducing = np.zeros((out_dim, int(num_inducing), in_dim), dtype=np.float64)
    all_alpha = np.zeros((out_dim, int(num_inducing)), dtype=np.float64)
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
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            min_noise=float(min_noise),
            learn_inducing=bool(learn_inducing),
            device=dev,
            seed=int(seed + 997 * (j + 1)),
        )
        all_inducing[j, :, :] = stats["inducing_points"]
        all_alpha[j, :] = stats["alpha"]
        all_ls[j, :] = stats["lengthscales"]
        all_os[j] = float(stats["outputscale"])
        all_noise[j] = float(stats["noise"])
        best_val_mse[j] = float(stats["best_val_mse"])
        train_histories.append(stats["train_history"])
        val_histories.append(stats["val_history"])

    # Compute scaled fit metrics with exported analytic map
    yhat_scaled = np.zeros_like(y_use)
    for j in range(out_dim):
        local_model = {
            "inducing_points": all_inducing[j, :, :],
            "alpha": all_alpha[j, :],
            "lengthscales": all_ls[j, :],
            "outputscale": all_os[j],
        }
        yhat_scaled[:, j] = _predict_analytic_batch(x_use, local_model)

    rel_l2_scaled = float(
        np.linalg.norm(yhat_scaled - y_use) / max(np.linalg.norm(y_use), 1e-30)
    )

    if x_val is not None and y_val is not None and x_val.shape[0] > 0:
        yhat_val = np.zeros_like(y_val)
        for j in range(out_dim):
            local_model = {
                "inducing_points": all_inducing[j, :, :],
                "alpha": all_alpha[j, :],
                "lengthscales": all_ls[j, :],
                "outputscale": all_os[j],
            }
            yhat_val[:, j] = _predict_analytic_batch(x_val, local_model)
        rel_l2_val_scaled = float(
            np.linalg.norm(yhat_val - y_val) / max(np.linalg.norm(y_val), 1e-30)
        )
    else:
        rel_l2_val_scaled = np.nan

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
        "num_inducing": int(num_inducing),
        "backend": "gpytorch",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "train_samples_used": int(n_use),
        "loss_history": np.asarray(
            [h[-1] if len(h) > 0 else np.nan for h in val_histories],
            dtype=float,
        ),
    }

    model_path = os.path.join(out_dir, "sparse_gp_model.npz")
    save_sparse_gp_model(model_path, model_dict)

    np.save(os.path.join(out_dir, "phi_p.npy"), phi_p)
    np.save(os.path.join(out_dir, "phi_s.npy"), phi_s)

    mu_qp_aff = _build_optional_mu_qp_affine(data_dir, n_primary=n_primary)
    if mu_qp_aff is not None:
        np.savez(
            os.path.join(out_dir, "qp_init_mu_affine.npz"),
            b_aff=np.asarray(mu_qp_aff["b_aff"], dtype=float),
            mu_dim=np.array([int(mu_qp_aff["mu_dim"])], dtype=np.int64),
            qp_dim=np.array([int(mu_qp_aff["qp_dim"])], dtype=np.int64),
            rel_fit=np.array([float(mu_qp_aff["rel_fit"])], dtype=float),
            n_samples=np.array([int(mu_qp_aff["n_samples"])], dtype=np.int64),
        )
        print(
            "[Sparse-GP] Saved q_p initializer map: "
            f"{os.path.join(out_dir, 'qp_init_mu_affine.npz')} "
            f"(mu_dim={mu_qp_aff['mu_dim']}, rel_fit={mu_qp_aff['rel_fit']:.3e})"
        )
    else:
        print("[Sparse-GP] No LS targets found; qp_init_mu_affine.npz was not generated.")

    summary_path = os.path.join(out_dir, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 7b Sparse-GP manifold summary\n")
        f.write("model_family=sparse_gp\n")
        f.write("kernel=rbf_ard\n")
        f.write(f"input_file={x_file}\n")
        f.write(f"n_samples_total={n_total}\n")
        f.write(f"n_samples_used={n_use}\n")
        f.write(f"n_train={x_train.shape[0]}\n")
        f.write(f"n_val={0 if x_val is None else x_val.shape[0]}\n")
        f.write(f"input_dim={in_dim}\n")
        f.write(f"output_dim={out_dim}\n")
        f.write(f"n_primary={n_primary}\n")
        f.write(f"n_secondary={n_secondary}\n")
        f.write("include_macro_strain_input=0\n")
        f.write(f"num_inducing={int(num_inducing)}\n")
        f.write(f"inducing_selection={inducing_selection}\n")
        f.write(f"epochs={int(epochs)}\n")
        f.write(f"batch_size={int(batch_size)}\n")
        f.write(f"lr={float(lr):.16e}\n")
        f.write(f"weight_decay={float(weight_decay):.16e}\n")
        f.write(f"min_noise={float(min_noise):.16e}\n")
        f.write(f"learn_inducing={int(bool(learn_inducing))}\n")
        f.write(f"device={dev}\n")
        f.write(f"scaled_train_rel_l2={rel_l2_scaled:.16e}\n")
        f.write(f"scaled_val_rel_l2={rel_l2_val_scaled:.16e}\n")
        f.write(
            "mean_best_val_mse_per_output="
            f"{float(np.nanmean(best_val_mse)):.16e}\n"
        )
        if mu_qp_aff is not None:
            f.write("qp_init_mu_affine_available=1\n")
            f.write(f"qp_init_mu_affine_mu_dim={int(mu_qp_aff['mu_dim'])}\n")
            f.write(f"qp_init_mu_affine_rel_fit={float(mu_qp_aff['rel_fit']):.16e}\n")
        else:
            f.write("qp_init_mu_affine_available=0\n")

    print("\n" + "=" * 72)
    print("Sparse-GP training complete")
    print("=" * 72)
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Scaled train rel L2: {rel_l2_scaled:.6e}")
    if np.isfinite(rel_l2_val_scaled):
        print(f"Scaled val   rel L2: {rel_l2_val_scaled:.6e}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Stage 7b true sparse-GP training with analytic export for PROM/HPROM."
    )
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--num-inducing", type=int, default=800)
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
        help="If >0, random subset size used for training (0 uses all).",
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--min-noise", type=float, default=1e-6)
    p.add_argument("--fixed-inducing", action="store_true", help="Disable inducing-point location updates.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()

    if args.val_fraction < 0.0 or args.val_fraction >= 1.0:
        raise ValueError("--val-fraction must satisfy 0 <= val_fraction < 1.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.lr <= 0.0:
        raise ValueError("--lr must be > 0.")
    if args.min_noise <= 0.0:
        raise ValueError("--min-noise must be > 0.")

    train_sparse_gp(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        num_inducing=args.num_inducing,
        inducing_selection=args.inducing_selection,
        kmeans_max_iters=args.kmeans_max_iters,
        kmeans_batch_size=args.kmeans_batch_size,
        kmeans_fit_samples=args.kmeans_fit_samples,
        train_samples=args.train_samples,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_noise=args.min_noise,
        learn_inducing=(not bool(args.fixed_inducing)),
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
