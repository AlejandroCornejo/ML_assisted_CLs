import os
import argparse
import numpy as np

from rbf_manifold_model import build_rbf_phi_matrix, save_rbf_model

DEFAULT_N_CENTERS = 4000


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


def _normalize_positive_scales(scales):
    s = np.asarray(scales, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("Empty anisotropic scales.")
    if np.any(s <= 0.0):
        raise ValueError("Anisotropic scales must be strictly positive.")
    log_gmean = float(np.mean(np.log(s)))
    return s / np.exp(log_gmean)


def _compute_linear_sensitivity_scales(x_scaled, y_scaled):
    # Fit y ≈ x A and use per-input column influence as directional sensitivity.
    a_ls, *_ = np.linalg.lstsq(x_scaled, y_scaled, rcond=None)
    sens = np.linalg.norm(a_ls, axis=1)
    sens = np.where(sens <= 1e-14, 1e-14, sens)
    return _normalize_positive_scales(sens)


def _build_base_anisotropic_scales(
    x_scaled,
    y_scaled,
    input_dim,
    rbf_metric="isotropic",
    anisotropic_scales=None,
    anisotropic_power=1.0,
    anisotropic_clip_min=0.25,
    anisotropic_clip_max=4.0,
):
    mode = str(rbf_metric).strip().lower()
    if mode not in ("isotropic", "anisotropic"):
        raise ValueError(f"Unsupported rbf metric '{rbf_metric}'.")
    if mode == "isotropic":
        return np.ones(int(input_dim), dtype=float), "isotropic"

    if anisotropic_scales is not None and str(anisotropic_scales).strip():
        vals = _parse_csv_floats(anisotropic_scales)
        if len(vals) != int(input_dim):
            raise ValueError(
                f"--anisotropic-scales must have {int(input_dim)} values, got {len(vals)}."
            )
        s = _normalize_positive_scales(vals)
        src = "manual"
    else:
        s = _compute_linear_sensitivity_scales(x_scaled, y_scaled)
        src = "linear_sensitivity"

    p = float(anisotropic_power)
    if not np.isfinite(p):
        p = 1.0
    s = np.power(s, p)
    s = _normalize_positive_scales(s)

    cmin = float(anisotropic_clip_min)
    cmax = float(anisotropic_clip_max)
    if cmin > 0.0 and cmax > cmin:
        s = np.clip(s, cmin, cmax)
        s = _normalize_positive_scales(s)

    return s, src


def _select_centers_random(x_scaled, n_centers, rng):
    n_samples = int(x_scaled.shape[0])
    idx = rng.choice(n_samples, size=int(n_centers), replace=False)
    centers = x_scaled[idx, :].copy()
    return centers, idx


def _select_centers_kmeans_minibatch(
    x_scaled,
    n_centers,
    rng,
    kmeans_max_iters=30,
    kmeans_batch_size=4096,
    kmeans_fit_samples=0,
):
    n_samples, in_dim = x_scaled.shape
    n_centers = int(n_centers)
    if n_centers <= 0:
        raise ValueError("n_centers must be > 0.")
    if n_samples < n_centers:
        raise ValueError(f"Need at least n_centers samples ({n_centers}), got {n_samples}.")

    n_fit = int(kmeans_fit_samples) if int(kmeans_fit_samples) > 0 else n_samples
    n_fit = min(n_fit, n_samples)
    if n_fit < n_centers:
        n_fit = n_centers

    if n_fit < n_samples:
        fit_idx = rng.choice(n_samples, size=n_fit, replace=False)
        x_fit = x_scaled[fit_idx, :]
    else:
        fit_idx = None
        x_fit = x_scaled

    init_idx_local = rng.choice(int(x_fit.shape[0]), size=n_centers, replace=False)
    centers = x_fit[init_idx_local, :].copy()
    counts = np.ones(n_centers, dtype=np.float64)
    batch_size = int(max(64, min(int(kmeans_batch_size), int(x_fit.shape[0]))))
    max_iters = int(max(1, kmeans_max_iters))

    print(
        f"[RBF] Center selection: kmeans_minibatch | fit_samples={x_fit.shape[0]} | "
        f"centers={n_centers} | batch_size={batch_size} | iters={max_iters}"
    )

    for it in range(max_iters):
        bidx = rng.choice(int(x_fit.shape[0]), size=batch_size, replace=False)
        xb = x_fit[bidx, :]
        d2 = _pairwise_dist2(xb, centers)
        assign = np.argmin(d2, axis=1)
        for j in np.unique(assign):
            pts = xb[assign == j, :]
            if pts.size == 0:
                continue
            n_old = counts[int(j)]
            n_new = float(pts.shape[0])
            centers[int(j), :] = (n_old * centers[int(j), :] + np.sum(pts, axis=0)) / (n_old + n_new)
            counts[int(j)] = n_old + n_new
        if (it + 1) % 10 == 0 or it == max_iters - 1:
            disp = np.mean(np.linalg.norm(centers, axis=1))
            print(f"  [kmeans] iter {it+1:03d}/{max_iters:03d} | mean(||c||)={disp:.3e}")

    # Final full assignment on fit set for a stable centroid update.
    d2_full = _pairwise_dist2(x_fit, centers)
    assign_full = np.argmin(d2_full, axis=1)
    for j in range(n_centers):
        pts = x_fit[assign_full == j, :]
        if pts.shape[0] > 0:
            centers[j, :] = np.mean(pts, axis=0)

    return centers, None


def _select_centers(
    x_scaled,
    n_centers,
    method,
    rng,
    kmeans_max_iters=30,
    kmeans_batch_size=4096,
    kmeans_fit_samples=0,
):
    mode = str(method).strip().lower()
    if mode in ("random", "rand"):
        return _select_centers_random(x_scaled, n_centers, rng)
    if mode in ("kmeans", "k-means", "mini_batch_kmeans", "minibatch_kmeans"):
        return _select_centers_kmeans_minibatch(
            x_scaled=x_scaled,
            n_centers=n_centers,
            rng=rng,
            kmeans_max_iters=kmeans_max_iters,
            kmeans_batch_size=kmeans_batch_size,
            kmeans_fit_samples=kmeans_fit_samples,
        )
    raise ValueError(f"Unsupported center selection method '{method}'.")


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


def _crossval_rel_l2(x, y, centers_scaled, kernel, epsilon_vec, ridge, splits):
    fold_errs = []
    for train_idx, val_idx in splits:
        x_tr = x[train_idx, :]
        y_tr = y[train_idx, :]
        x_va = x[val_idx, :]
        y_va = y[val_idx, :]

        phi_tr = build_rbf_phi_matrix(x_tr, centers_scaled, kernel, epsilon_vec)
        w = _solve_weights_from_phi(phi_tr, y_tr, ridge)

        phi_va = build_rbf_phi_matrix(x_va, centers_scaled, kernel, epsilon_vec)
        y_hat = phi_va @ w
        denom = np.linalg.norm(y_va, ord="fro")
        if denom <= 1e-30:
            continue
        fold_errs.append(float(np.linalg.norm(y_hat - y_va, ord="fro") / denom))

    if not fold_errs:
        return np.inf
    return float(np.mean(fold_errs))


def _fit_full_dataset_weights(x_scaled, y_scaled, centers_scaled, kernel, epsilon_vec, ridge, block_size):
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
        phi = build_rbf_phi_matrix(xb, centers_scaled, kernel, epsilon_vec)
        a += phi.T @ phi
        b += phi.T @ yb

    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(a, b, rcond=None)
    return w


def _relative_l2_on_dataset(x_scaled, y_scaled, centers_scaled, weights, kernel, epsilon_vec, block_size):
    err_num = 0.0
    y_norm2 = 0.0
    n_samples = int(x_scaled.shape[0])
    bs = int(max(1, block_size))
    for i0 in range(0, n_samples, bs):
        i1 = min(i0 + bs, n_samples)
        xb = x_scaled[i0:i1, :]
        yb = y_scaled[i0:i1, :]
        phi = build_rbf_phi_matrix(xb, centers_scaled, kernel, epsilon_vec)
        yp = phi @ weights
        diff = yp - yb
        err_num += float(np.sum(diff * diff))
        y_norm2 += float(np.sum(yb * yb))
    return float(np.sqrt(err_num / max(y_norm2, 1e-30)))


def _run_grid_search(
    x_scaled,
    y_scaled,
    centers_scaled,
    base_scales,
    base_kernel,
    base_ridge,
    grid_kernels,
    grid_eps_values,
    grid_ridges,
    grid_folds,
    grid_max_samples,
    grid_seed,
):
    rng = np.random.default_rng(int(grid_seed))

    kernels = _parse_csv_strings(grid_kernels) if grid_kernels else [str(base_kernel)]
    kernels = [_canonical_kernel_name(k) for k in kernels]
    kernels = list(dict.fromkeys(kernels))

    eps_grid = _parse_csv_floats(grid_eps_values)
    eps_grid = sorted({float(v) for v in eps_grid if float(v) > 0.0})
    if not eps_grid:
        raise RuntimeError(
            "Empty epsilon grid for RBF search. "
            "Provide explicit absolute values via --grid-eps-values."
        )

    if grid_ridges:
        ridge_grid = _parse_csv_floats(grid_ridges)
    else:
        ridge_grid = [float(base_ridge)]
    ridge_grid = sorted({float(v) for v in ridge_grid if float(v) >= 0.0})
    if not ridge_grid:
        raise RuntimeError("Empty ridge grid for RBF search.")

    n_samples = int(x_scaled.shape[0])
    # Policy: always use all available training data for RBF grid search.
    n_search = int(n_samples)
    idx = np.arange(n_samples, dtype=np.int64)
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
    print(f"  anisotropic scales: {np.asarray(base_scales, dtype=float)}")
    print(f"  ridge grid: {ridge_grid}")
    print(f"  input_dim: {x_scaled.shape[1]} (n_primary={centers_scaled.shape[1]})")
    print(f"  output_dim: {y_scaled.shape[1]}")
    print(f"  combinations: {n_comb}")
    print("-" * 60)

    results = []
    for kernel in kernels:
        for eps in eps_grid:
            for ridge in ridge_grid:
                eps_vec = float(eps) * np.asarray(base_scales, dtype=float)
                cv_rel_l2 = _crossval_rel_l2(
                    x=x_s,
                    y=y_s,
                    centers_scaled=centers_scaled,
                    kernel=kernel,
                    epsilon_vec=eps_vec,
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
    max_centers=DEFAULT_N_CENTERS,
    center_selection="random",
    sparse_prune_centers=0,
    rbf_metric="isotropic",
    anisotropic_scales=None,
    anisotropic_power=1.0,
    anisotropic_clip_min=0.25,
    anisotropic_clip_max=4.0,
    kmeans_max_iters=30,
    kmeans_batch_size=4096,
    kmeans_fit_samples=0,
    kernel="inverse_multiquadric",
    ridge=1e-8,
    block_size=2048,
    seed=42,
    grid_search=True,
    grid_kernels=None,
    grid_eps_values=None,
    grid_ridges=None,
    grid_folds=3,
    grid_max_samples=0,
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
    if include_macro:
        raise RuntimeError(
            "Dataset was built with macro-strain manifold inputs (N(q,mu)), "
            "which is no longer supported. Rebuild Stage 7a without macro inputs."
        )

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

    if not bool(grid_search):
        raise RuntimeError(
            "Stage 7b-RBF policy: grid search is mandatory. "
            "Training without grid search is not allowed."
        )

    max_centers = int(max_centers)
    if max_centers <= 0:
        raise RuntimeError("Stage 7b-RBF policy: max_centers must be a positive integer.")

    if int(n_samples) < max_centers:
        raise RuntimeError(
            f"Stage 7b-RBF policy requires at least {max_centers} samples "
            f"to place {max_centers} centers, but dataset has {n_samples}."
        )

    m = max_centers
    centers_scaled, center_indices = _select_centers(
        x_scaled=x_scaled,
        n_centers=m,
        method=center_selection,
        rng=rng,
        kmeans_max_iters=kmeans_max_iters,
        kmeans_batch_size=kmeans_batch_size,
        kmeans_fit_samples=kmeans_fit_samples,
    )

    kernel_sel = _canonical_kernel_name(kernel)
    ridge_sel = float(ridge)
    eps_sel = 0.0
    eps_vec_sel = np.ones(in_dim, dtype=float)
    grid_results = []
    search_n_samples = 0
    search_n_folds = 0
    metric_scales, metric_source = _build_base_anisotropic_scales(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        input_dim=in_dim,
        rbf_metric=rbf_metric,
        anisotropic_scales=anisotropic_scales,
        anisotropic_power=anisotropic_power,
        anisotropic_clip_min=anisotropic_clip_min,
        anisotropic_clip_max=anisotropic_clip_max,
    )

    best, grid_results, search_n_samples, search_n_folds = _run_grid_search(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        centers_scaled=centers_scaled,
        base_scales=metric_scales,
        base_kernel=kernel_sel,
        base_ridge=ridge_sel,
        grid_kernels=grid_kernels,
        grid_eps_values=grid_eps_values,
        grid_ridges=grid_ridges,
        grid_folds=grid_folds,
        grid_max_samples=grid_max_samples,
        grid_seed=grid_seed,
    )
    kernel_sel = _canonical_kernel_name(best["kernel"])
    eps_sel = float(best["epsilon"])
    eps_vec_sel = eps_sel * metric_scales
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
        f"Centers: {m} | selection={center_selection} | "
        f"Kernel: {kernel_sel} | epsilon={eps_sel:.6e} | ridge={ridge_sel:.3e}"
    )
    print(
        f"RBF metric: {rbf_metric} | anisotropic source: {metric_source} | "
        f"scales={metric_scales}"
    )
    print(
        f"Grid-search selected hyperparameters from {search_n_samples} samples and {search_n_folds} folds."
    )

    weights_scaled = _fit_full_dataset_weights(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        centers_scaled=centers_scaled,
        kernel=kernel_sel,
        epsilon_vec=eps_vec_sel,
        ridge=ridge_sel,
        block_size=block_size,
    )

    sparse_keep = int(sparse_prune_centers)
    sparse_keep = max(0, sparse_keep)
    if sparse_keep > 0 and sparse_keep < int(centers_scaled.shape[0]):
        print(
            f"[RBF] Sparse pruning enabled: keep {sparse_keep}/{centers_scaled.shape[0]} centers "
            "by ||weights||_2, then refit."
        )
        center_scores = np.linalg.norm(weights_scaled, axis=1)
        keep = np.argpartition(center_scores, -sparse_keep)[-sparse_keep:]
        keep = keep[np.argsort(center_scores[keep])[::-1]]

        centers_scaled = centers_scaled[keep, :]
        if center_indices is not None:
            center_indices = np.asarray(center_indices, dtype=np.int64)[keep]
        else:
            center_indices = None

        weights_scaled = _fit_full_dataset_weights(
            x_scaled=x_scaled,
            y_scaled=y_scaled,
            centers_scaled=centers_scaled,
            kernel=kernel_sel,
            epsilon_vec=eps_vec_sel,
            ridge=ridge_sel,
            block_size=block_size,
        )
        print(f"[RBF] Sparse pruning done. Active centers: {centers_scaled.shape[0]}")

    rel_l2 = _relative_l2_on_dataset(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        centers_scaled=centers_scaled,
        weights=weights_scaled,
        kernel=kernel_sel,
        epsilon_vec=eps_vec_sel,
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
        "epsilon_vector": eps_vec_sel,
        "ridge": ridge_sel,
        "kernel_name": str(kernel_sel),
        "input_dim": int(in_dim),
        "output_dim": int(out_dim),
        "n_primary": int(n_primary),
        "n_secondary": int(n_secondary),
        "include_macro_strain_input": False,
        "center_indices": center_indices,
        "center_selection": str(center_selection),
        "sparse_prune_centers": int(sparse_prune_centers),
        "rbf_metric": str(rbf_metric),
        "anisotropic_scales_base": metric_scales,
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
        f.write("include_macro_strain_input=0\n")
        f.write(f"n_centers={m}\n")
        f.write(f"n_centers_active={int(centers_scaled.shape[0])}\n")
        f.write(f"center_selection={center_selection}\n")
        f.write(f"sparse_prune_centers={int(sparse_prune_centers)}\n")
        f.write(f"rbf_metric={rbf_metric}\n")
        f.write(f"anisotropic_scale_source={metric_source}\n")
        f.write(
            "anisotropic_scales_base="
            + ",".join([f"{v:.16e}" for v in np.asarray(metric_scales, dtype=float)])
            + "\n"
        )
        f.write(f"kernel={kernel_sel}\n")
        f.write(f"epsilon={eps_sel:.16e}\n")
        f.write(
            "epsilon_vector="
            + ",".join([f"{v:.16e}" for v in np.asarray(eps_vec_sel, dtype=float)])
            + "\n"
        )
        f.write(f"ridge={ridge_sel:.16e}\n")
        f.write("grid_search=1\n")
        f.write(f"grid_search_samples={search_n_samples}\n")
        f.write(f"grid_search_folds={search_n_folds}\n")
        f.write(f"grid_candidates={len(grid_results)}\n")
        f.write(f"scaled_train_rel_l2={rel_l2:.16e}\n")

    if grid_results:
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
    parser.add_argument("--max-centers", type=int, default=DEFAULT_N_CENTERS, help=f"Number of RBF centers (default: {DEFAULT_N_CENTERS}).")
    parser.add_argument(
        "--center-selection",
        type=str,
        default="random",
        choices=["random", "kmeans"],
        help="Center selection strategy: random sample or mini-batch kmeans centroids.",
    )
    parser.add_argument(
        "--rbf-metric",
        type=str,
        default="isotropic",
        choices=["isotropic", "anisotropic"],
        help="Kernel metric type. anisotropic uses a per-dimension epsilon vector.",
    )
    parser.add_argument(
        "--anisotropic-scales",
        type=str,
        default="",
        help=(
            "Optional comma-separated positive per-dimension scales. "
            "If empty and --rbf-metric=anisotropic, scales are estimated from linear sensitivity."
        ),
    )
    parser.add_argument(
        "--anisotropic-power",
        type=float,
        default=1.0,
        help="Exponent applied to base anisotropic scales before normalization.",
    )
    parser.add_argument(
        "--anisotropic-clip-min",
        type=float,
        default=0.25,
        help="Minimum clipping bound for anisotropic scales (before renormalization).",
    )
    parser.add_argument(
        "--anisotropic-clip-max",
        type=float,
        default=4.0,
        help="Maximum clipping bound for anisotropic scales (before renormalization).",
    )
    parser.add_argument(
        "--sparse-prune-centers",
        type=int,
        default=0,
        help="If >0, keep this many centers by ||weights||_2 and refit (sparse RBF).",
    )
    parser.add_argument(
        "--kmeans-max-iters",
        type=int,
        default=30,
        help="Mini-batch kmeans iterations when --center-selection=kmeans.",
    )
    parser.add_argument(
        "--kmeans-batch-size",
        type=int,
        default=4096,
        help="Mini-batch size when --center-selection=kmeans.",
    )
    parser.add_argument(
        "--kmeans-fit-samples",
        type=int,
        default=0,
        help="Optional cap of samples used for kmeans fitting (0 uses all).",
    )
    parser.add_argument("--kernel", type=str, default="inverse_multiquadric", choices=["inverse_multiquadric", "imq", "gaussian", "multiquadric", "mq", "gauss"], help="RBF kernel type.")
    parser.add_argument("--ridge", type=float, default=1e-8, help="Ridge regularization on normal equations.")
    parser.add_argument("--block-size", type=int, default=2048, help="Block size for streaming normal-equation assembly.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--grid-search", action="store_true", help="Deprecated flag; grid search is always enabled.")
    parser.add_argument("--grid-kernels", type=str, default="", help="Comma-separated kernel list (e.g. 'imq,gaussian,mq').")
    parser.add_argument(
        "--grid-eps-values",
        type=str,
        default="0.01,0.05,0.1,0.25,0.5,1.0,2.0,5.0",
        help="Comma-separated epsilon values (absolute).",
    )
    parser.add_argument("--grid-ridges", type=str, default="", help="Comma-separated ridge values.")
    parser.add_argument("--grid-folds", type=int, default=3, help="Number of K-fold splits for grid search.")
    parser.add_argument("--grid-max-samples", type=int, default=0, help="Deprecated in policy mode; all samples are always used.")
    parser.add_argument("--grid-seed", type=int, default=42, help="Random seed for grid-search subset and folds.")
    args = parser.parse_args()

    train_rbf(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_centers=args.max_centers,
        center_selection=args.center_selection,
        sparse_prune_centers=args.sparse_prune_centers,
        rbf_metric=args.rbf_metric,
        anisotropic_scales=args.anisotropic_scales if args.anisotropic_scales else None,
        anisotropic_power=args.anisotropic_power,
        anisotropic_clip_min=args.anisotropic_clip_min,
        anisotropic_clip_max=args.anisotropic_clip_max,
        kmeans_max_iters=args.kmeans_max_iters,
        kmeans_batch_size=args.kmeans_batch_size,
        kmeans_fit_samples=args.kmeans_fit_samples,
        kernel=args.kernel,
        ridge=args.ridge,
        block_size=args.block_size,
        seed=args.seed,
        grid_search=True,
        grid_kernels=args.grid_kernels if args.grid_kernels else None,
        grid_eps_values=args.grid_eps_values if args.grid_eps_values else None,
        grid_ridges=args.grid_ridges if args.grid_ridges else None,
        grid_folds=args.grid_folds,
        grid_max_samples=args.grid_max_samples,
        grid_seed=args.grid_seed,
    )
