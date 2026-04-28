import os
import argparse
import numpy as np

from rbf_manifold_model import _kernel_from_dist2, save_rbf_model


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


def _pairwise_dist2(x, c):
    # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
    x2 = np.sum(x * x, axis=1, keepdims=True)
    c2 = np.sum(c * c, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (x @ c.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def _canonical_kernel_name(name):
    k = str(name).strip().lower()
    if k in ("inverse_multiquadric", "imq"):
        return "imq"
    if k in ("gaussian", "gauss"):
        return "gaussian"
    if k in ("multiquadric", "mq"):
        return "mq"
    raise ValueError(f"Unsupported kernel '{name}'.")


def _parse_csv_floats(text):
    if text is None:
        return []
    vals = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals


def _parse_csv_strings(text):
    if text is None:
        return []
    vals = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(s)
    return vals


def _build_kfold_splits(n_samples, n_folds, rng):
    n = int(n_samples)
    k = int(max(2, min(int(n_folds), n)))
    perm = rng.permutation(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: (n % k)] += 1

    splits = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + int(fold_size)
        val_idx = perm[start:stop]
        train_idx = np.concatenate([perm[:start], perm[stop:]])
        splits.append((train_idx, val_idx))
        start = stop
    return splits


def _solve_weights_from_phi(phi, y, ridge):
    m = int(phi.shape[1])
    a = phi.T @ phi
    if float(ridge) > 0.0:
        a += float(ridge) * np.eye(m, dtype=np.float64)
    b = phi.T @ y
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(a, b, rcond=None)
    return w


def _crossval_rel_l2(x, y, centers_scaled, kernel, epsilon, ridge, splits):
    fold_errs = []
    for train_idx, val_idx in splits:
        x_tr = x[train_idx, :]
        y_tr = y[train_idx, :]
        x_va = x[val_idx, :]
        y_va = y[val_idx, :]

        phi_tr = _kernel_from_dist2(_pairwise_dist2(x_tr, centers_scaled), kernel, epsilon)
        w = _solve_weights_from_phi(phi_tr, y_tr, ridge)

        phi_va = _kernel_from_dist2(_pairwise_dist2(x_va, centers_scaled), kernel, epsilon)
        y_hat = phi_va @ w
        denom = np.linalg.norm(y_va, ord="fro")
        if denom <= 1e-30:
            continue
        fold_errs.append(float(np.linalg.norm(y_hat - y_va, ord="fro") / denom))

    if not fold_errs:
        return np.inf
    return float(np.mean(fold_errs))


def _auto_epsilon(x_scaled, centers_scaled, rng):
    n = x_scaled.shape[0]
    m = centers_scaled.shape[0]
    ns = min(5000, n)
    xs = x_scaled[rng.choice(n, size=ns, replace=False)]
    d2 = _pairwise_dist2(xs, centers_scaled)

    # nearest center distance (exclude exact zero only if present)
    nn = np.sqrt(np.min(d2, axis=1))
    nn = nn[nn > 1e-14]
    if nn.size == 0:
        return 1.0
    med = float(np.median(nn))
    if med <= 1e-14 or not np.isfinite(med):
        return 1.0
    return 1.0 / med


def _fit_full_dataset_weights(x_scaled, y_scaled, centers_scaled, kernel, epsilon, ridge, block_size):
    m = int(centers_scaled.shape[0])
    out_dim = int(y_scaled.shape[1])
    a = float(ridge) * np.eye(m, dtype=np.float64)
    b = np.zeros((m, out_dim), dtype=np.float64)

    n_samples = int(x_scaled.shape[0])
    bs = int(max(1, block_size))
    for i0 in range(0, n_samples, bs):
        i1 = min(i0 + bs, n_samples)
        xb = x_scaled[i0:i1, :]
        yb = y_scaled[i0:i1, :]
        phi = _kernel_from_dist2(_pairwise_dist2(xb, centers_scaled), kernel, epsilon)
        a += phi.T @ phi
        b += phi.T @ yb

    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(a, b, rcond=None)
    return w


def _relative_l2_on_dataset(x_scaled, y_scaled, centers_scaled, weights, kernel, epsilon, block_size):
    err_num = 0.0
    y_norm2 = 0.0
    n_samples = int(x_scaled.shape[0])
    bs = int(max(1, block_size))
    for i0 in range(0, n_samples, bs):
        i1 = min(i0 + bs, n_samples)
        xb = x_scaled[i0:i1, :]
        yb = y_scaled[i0:i1, :]
        phi = _kernel_from_dist2(_pairwise_dist2(xb, centers_scaled), kernel, epsilon)
        yp = phi @ weights
        diff = yp - yb
        err_num += float(np.sum(diff * diff))
        y_norm2 += float(np.sum(yb * yb))
    return float(np.sqrt(err_num / max(y_norm2, 1e-30)))


def _run_grid_search(
    x_scaled,
    y_scaled,
    centers_scaled,
    base_kernel,
    base_epsilon,
    base_ridge,
    grid_kernels,
    grid_eps_values,
    grid_eps_factors,
    grid_ridges,
    grid_folds,
    grid_max_samples,
    grid_seed,
):
    rng = np.random.default_rng(int(grid_seed))

    kernels = _parse_csv_strings(grid_kernels) if grid_kernels else [str(base_kernel)]
    kernels = [_canonical_kernel_name(k) for k in kernels]
    kernels = list(dict.fromkeys(kernels))

    if grid_eps_values:
        eps_grid = _parse_csv_floats(grid_eps_values)
    else:
        factors = _parse_csv_floats(grid_eps_factors)
        if not factors:
            factors = [1.0]
        eps_grid = [float(base_epsilon) * float(f) for f in factors]
    eps_grid = sorted({float(v) for v in eps_grid if float(v) > 0.0})
    if not eps_grid:
        raise RuntimeError("Empty epsilon grid for RBF search.")

    if grid_ridges:
        ridge_grid = _parse_csv_floats(grid_ridges)
    else:
        ridge_grid = [float(base_ridge)]
    ridge_grid = sorted({float(v) for v in ridge_grid if float(v) >= 0.0})
    if not ridge_grid:
        raise RuntimeError("Empty ridge grid for RBF search.")

    n_samples = int(x_scaled.shape[0])
    n_search = int(min(max(10, int(grid_max_samples)), n_samples))
    idx = rng.choice(n_samples, size=n_search, replace=False)
    x_s = x_scaled[idx, :]
    y_s = y_scaled[idx, :]

    splits = _build_kfold_splits(n_search, grid_folds, rng)
    n_folds = len(splits)
    n_comb = len(kernels) * len(eps_grid) * len(ridge_grid)

    print("-" * 60)
    print("Stage 7b-RBF grid search (Burgers-style CV)")
    print(f"  search samples: {n_search}/{n_samples}")
    print(f"  folds: {n_folds}")
    print(f"  kernels: {kernels}")
    print(f"  epsilon grid: {eps_grid}")
    print(f"  ridge grid: {ridge_grid}")
    print(f"  input_dim: {x_scaled.shape[1]} (n_primary={centers_scaled.shape[1]})")
    print(f"  output_dim: {y_scaled.shape[1]}")
    print(f"  combinations: {n_comb}")
    print("-" * 60)

    results = []
    for kernel in kernels:
        for eps in eps_grid:
            for ridge in ridge_grid:
                cv_rel_l2 = _crossval_rel_l2(
                    x=x_s,
                    y=y_s,
                    centers_scaled=centers_scaled,
                    kernel=kernel,
                    epsilon=float(eps),
                    ridge=float(ridge),
                    splits=splits,
                )
                results.append(
                    {
                        "kernel": str(kernel),
                        "epsilon": float(eps),
                        "ridge": float(ridge),
                        "cv_rel_l2": float(cv_rel_l2),
                    }
                )
                if np.isfinite(cv_rel_l2):
                    print(
                        f"  [grid] kernel={kernel:8s} eps={eps:.4e} ridge={ridge:.3e} cv_rel_l2={cv_rel_l2:.4e}"
                    )
                else:
                    print(
                        f"  [grid] kernel={kernel:8s} eps={eps:.4e} ridge={ridge:.3e} cv_rel_l2=inf"
                    )

    finite = [r for r in results if np.isfinite(r["cv_rel_l2"])]
    if not finite:
        raise RuntimeError("All grid-search combinations failed.")
    finite.sort(key=lambda d: d["cv_rel_l2"])
    best = finite[0]

    print("-" * 60)
    print(
        "Best grid point: "
        f"kernel={best['kernel']}, epsilon={best['epsilon']:.6e}, "
        f"ridge={best['ridge']:.3e}, cv_rel_l2={best['cv_rel_l2']:.6e}"
    )
    print("-" * 60)

    return best, results, n_search, n_folds


def train_rbf(
    data_dir="stage_7_ann_data",
    out_dir="stage_7_rbf_data",
    max_centers=1200,
    kernel="inverse_multiquadric",
    epsilon=None,
    ridge=1e-8,
    block_size=2048,
    seed=42,
    grid_search=False,
    grid_kernels=None,
    grid_eps_values=None,
    grid_eps_factors="0.25,0.5,1.0,2.0,4.0",
    grid_ridges=None,
    grid_folds=3,
    grid_max_samples=12000,
    grid_seed=42,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    x_file = "ann_input_train.npy" if os.path.exists(os.path.join(data_dir, "ann_input_train.npy")) else "q_p_train.npy"
    x = np.load(os.path.join(data_dir, x_file)).astype(np.float64)
    y = np.load(os.path.join(data_dir, "q_s_train.npy")).astype(np.float64)

    cfg = _load_dataset_config(data_dir)
    n_samples, in_dim = x.shape
    out_dim = int(y.shape[1])

    n_primary = int(cfg["n_primary"] if cfg["n_primary"] is not None else in_dim)
    n_secondary = int(cfg["n_secondary"] if cfg["n_secondary"] is not None else out_dim)
    include_macro = bool(cfg["include_macro_strain_input"]) if cfg["input_dim"] is not None else (in_dim > n_primary)

    if out_dim != n_secondary:
        raise ValueError(f"Output dimension mismatch: y has {out_dim}, expected n_secondary={n_secondary}.")
    if in_dim != int(cfg["input_dim"] if cfg["input_dim"] is not None else in_dim):
        raise ValueError("Input dimension mismatch with dataset metadata.")

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)

    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std = np.where(y_std < 1e-10, 1.0, y_std)

    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    m = int(min(max_centers, n_samples))
    center_indices = rng.choice(n_samples, size=m, replace=False)
    centers_scaled = x_scaled[center_indices, :].copy()

    if epsilon is None or float(epsilon) <= 0.0:
        eps_base = _auto_epsilon(x_scaled, centers_scaled, rng)
    else:
        eps_base = float(epsilon)

    kernel_sel = _canonical_kernel_name(kernel)
    ridge_sel = float(ridge)
    eps_sel = float(eps_base)
    grid_results = []
    search_n_samples = 0
    search_n_folds = 0

    if bool(grid_search):
        best, grid_results, search_n_samples, search_n_folds = _run_grid_search(
            x_scaled=x_scaled,
            y_scaled=y_scaled,
            centers_scaled=centers_scaled,
            base_kernel=kernel_sel,
            base_epsilon=eps_base,
            base_ridge=ridge_sel,
            grid_kernels=grid_kernels,
            grid_eps_values=grid_eps_values,
            grid_eps_factors=grid_eps_factors,
            grid_ridges=grid_ridges,
            grid_folds=grid_folds,
            grid_max_samples=grid_max_samples,
            grid_seed=grid_seed,
        )
        kernel_sel = _canonical_kernel_name(best["kernel"])
        eps_sel = float(best["epsilon"])
        ridge_sel = float(best["ridge"])

    print("=" * 60)
    print("Stage 7b-RBF: Compact-center manifold training")
    print(f"  input_dim: {in_dim} (n_primary={n_primary})")
    print(f"  output_dim: {out_dim}")
    print("=" * 60)
    print(f"Input file: {x_file}")
    print(f"X shape: {x.shape} | Y shape: {y.shape}")
    print(f"n_primary={n_primary} | n_secondary={n_secondary} | include_macro_strain_input={include_macro}")
    print(
        f"Centers: {m} | Kernel: {kernel_sel} | epsilon={eps_sel:.6e} | ridge={ridge_sel:.3e}"
    )
    if bool(grid_search):
        print(
            f"Grid-search selected hyperparameters from {search_n_samples} samples and {search_n_folds} folds."
        )

    weights_scaled = _fit_full_dataset_weights(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        centers_scaled=centers_scaled,
        kernel=kernel_sel,
        epsilon=eps_sel,
        ridge=ridge_sel,
        block_size=block_size,
    )

    rel_l2 = _relative_l2_on_dataset(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        centers_scaled=centers_scaled,
        weights=weights_scaled,
        kernel=kernel_sel,
        epsilon=eps_sel,
        block_size=block_size,
    )
    print(f"Scaled training relative L2 error: {rel_l2:.6e}")

    model = {
        "centers_scaled": centers_scaled,
        "weights_scaled": weights_scaled,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "epsilon": eps_sel,
        "ridge": ridge_sel,
        "kernel_name": str(kernel_sel),
        "input_dim": int(in_dim),
        "output_dim": int(out_dim),
        "n_primary": int(n_primary),
        "n_secondary": int(n_secondary),
        "include_macro_strain_input": bool(include_macro),
        "center_indices": center_indices,
    }

    model_path = os.path.join(out_dir, "rbf_model.npz")
    save_rbf_model(model_path, model)
    np.save(os.path.join(out_dir, "phi_p.npy"), np.load(os.path.join(data_dir, "phi_p.npy")))
    np.save(os.path.join(out_dir, "phi_s.npy"), np.load(os.path.join(data_dir, "phi_s.npy")))

    summary_path = os.path.join(out_dir, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 7b-RBF compact-center summary\n")
        f.write(f"input_file={x_file}\n")
        f.write(f"n_samples={n_samples}\n")
        f.write(f"input_dim={in_dim}\n")
        f.write(f"output_dim={out_dim}\n")
        f.write(f"n_primary={n_primary}\n")
        f.write(f"n_secondary={n_secondary}\n")
        f.write(f"include_macro_strain_input={int(include_macro)}\n")
        f.write(f"n_centers={m}\n")
        f.write(f"kernel={kernel_sel}\n")
        f.write(f"epsilon={eps_sel:.16e}\n")
        f.write(f"ridge={ridge_sel:.16e}\n")
        f.write(f"base_epsilon={eps_base:.16e}\n")
        f.write(f"grid_search={int(bool(grid_search))}\n")
        if bool(grid_search):
            f.write(f"grid_search_samples={search_n_samples}\n")
            f.write(f"grid_search_folds={search_n_folds}\n")
            f.write(f"grid_candidates={len(grid_results)}\n")
        f.write(f"scaled_train_rel_l2={rel_l2:.16e}\n")

    if bool(grid_search) and grid_results:
        grid_file = os.path.join(out_dir, "grid_search_results.csv")
        with open(grid_file, "w", encoding="utf-8") as f:
            f.write("kernel,epsilon,ridge,cv_rel_l2\n")
            for r in grid_results:
                f.write(
                    f"{r['kernel']},{r['epsilon']:.16e},{r['ridge']:.16e},{r['cv_rel_l2']:.16e}\n"
                )
        print(f"Saved grid-search table to: {grid_file}")

    print(f"Saved model to: {model_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7b-RBF: train compact-center RBF manifold q_p -> q_s")
    parser.add_argument("--data-dir", type=str, default="stage_7_ann_data", help="Input dataset directory from Stage 7a.")
    parser.add_argument("--out-dir", type=str, default="stage_7_rbf_data", help="Output directory for trained RBF model.")
    parser.add_argument("--max-centers", type=int, default=1200, help="Number of compact RBF centers.")
    parser.add_argument("--kernel", type=str, default="inverse_multiquadric", choices=["inverse_multiquadric", "imq", "gaussian", "multiquadric", "mq", "gauss"], help="RBF kernel type.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Kernel epsilon; <=0 means auto.")
    parser.add_argument("--ridge", type=float, default=1e-8, help="Ridge regularization on normal equations.")
    parser.add_argument("--block-size", type=int, default=2048, help="Block size for streaming normal-equation assembly.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--grid-search", action="store_true", help="Run Burgers-style K-fold grid search for kernel/epsilon/ridge.")
    parser.add_argument("--grid-kernels", type=str, default="", help="Comma-separated kernel list (e.g. 'imq,gaussian,mq').")
    parser.add_argument("--grid-eps-values", type=str, default="", help="Comma-separated epsilon values (absolute).")
    parser.add_argument("--grid-eps-factors", type=str, default="0.25,0.5,1.0,2.0,4.0", help="Comma-separated factors applied to base epsilon when grid-eps-values is empty.")
    parser.add_argument("--grid-ridges", type=str, default="", help="Comma-separated ridge values.")
    parser.add_argument("--grid-folds", type=int, default=3, help="Number of K-fold splits for grid search.")
    parser.add_argument("--grid-max-samples", type=int, default=12000, help="Max samples used in grid search subset.")
    parser.add_argument("--grid-seed", type=int, default=42, help="Random seed for grid-search subset and folds.")
    args = parser.parse_args()

    train_rbf(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_centers=args.max_centers,
        kernel=args.kernel,
        epsilon=args.epsilon,
        ridge=args.ridge,
        block_size=args.block_size,
        seed=args.seed,
        grid_search=args.grid_search,
        grid_kernels=args.grid_kernels if args.grid_kernels else None,
        grid_eps_values=args.grid_eps_values if args.grid_eps_values else None,
        grid_eps_factors=args.grid_eps_factors,
        grid_ridges=args.grid_ridges if args.grid_ridges else None,
        grid_folds=args.grid_folds,
        grid_max_samples=args.grid_max_samples,
        grid_seed=args.grid_seed,
    )
