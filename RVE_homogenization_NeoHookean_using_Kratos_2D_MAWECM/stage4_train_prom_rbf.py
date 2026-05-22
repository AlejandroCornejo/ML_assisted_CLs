#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 4 (2D-MAWECM): train PROM-RBF from Stage3 dataset.

Features:
- isotropic RBF (scalar epsilon)
- sparse inducing-point subset
- optional grid search over epsilon/smoothing/neighbors/inducing-points
"""

import argparse
import json
import os
import pickle
from typing import Dict, List

import numpy as np
from scipy.interpolate import RBFInterpolator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage4: train PROM-RBF from Stage3 dataset")
    p.add_argument(
        "--dataset-file",
        type=str,
        default="stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    )
    p.add_argument("--out-dir", type=str, default="stage_4_prom_rbf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rbf-input-space",
        type=str,
        default="q_m",
        choices=["q_m", "mu"],
        help="Input space for the decoder map: q_m->target (docs-aligned) or mu->target.",
    )

    # Single-run settings
    p.add_argument(
        "--rbf-kernel",
        type=str,
        default="gaussian",
        choices=[
            "linear",
            "thin_plate_spline",
            "cubic",
            "quintic",
            "multiquadric",
            "inverse_multiquadric",
            "inverse_quadratic",
            "gaussian",
        ],
    )
    p.add_argument("--rbf-epsilon", type=float, default=1.0, help="Isotropic epsilon.")
    p.add_argument("--rbf-smoothing", type=float, default=1e-10)
    p.add_argument("--rbf-neighbors", type=int, default=120, help="0 means global (all).")

    # Sparse inducing points
    p.add_argument(
        "--rbf-sparse-inducing-points",
        type=int,
        default=0,
        help="Number of inducing points. <=0 means all training points.",
    )
    p.add_argument(
        "--rbf-inducing-selector",
        type=str,
        default="structured_then_farthest",
        choices=["random", "farthest", "structured_then_farthest"],
    )

    # Grid search
    p.add_argument("--rbf-grid-search", type=int, default=0, choices=[0, 1])
    p.add_argument("--rbf-grid-epsilons", type=str, default="0.25,0.5,1.0,2.0")
    p.add_argument("--rbf-grid-smoothings", type=str, default="1e-12,1e-10,1e-8,1e-6")
    p.add_argument("--rbf-grid-neighbors", type=str, default="60,80,120,180,0")
    p.add_argument("--rbf-grid-inducing-points", type=str, default="451,900,1800,0")
    p.add_argument(
        "--rbf-grid-metric",
        type=str,
        default="val_rel_l2",
        choices=["val_rel_l2", "val_rmse"],
    )

    return p.parse_args()


def _parse_csv_int_list(text: str, name: str) -> List[int]:
    vals = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(float(t)))
    if len(vals) == 0:
        raise ValueError(f"{name} list is empty.")
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _parse_csv_float_list(text: str, name: str) -> List[float]:
    vals = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    if len(vals) == 0:
        raise ValueError(f"{name} list is empty.")
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    err = y_pred - y_true
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


def _fit_standardizer(x_train: np.ndarray):
    mean = np.mean(np.asarray(x_train, dtype=float), axis=0)
    std = np.std(np.asarray(x_train, dtype=float), axis=0)
    std = np.where(std < 1e-14, 1.0, std)
    return mean, std


def _apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (np.asarray(x, dtype=float) - mean[None, :]) / std[None, :]


def _farthest_point_sampling(x: np.ndarray, k: int, seed: int, init_idx: np.ndarray = None) -> np.ndarray:
    n = x.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))

    if init_idx is None or init_idx.size == 0:
        first = int(rng.integers(0, n))
        selected = [first]
    else:
        uniq = np.unique(np.asarray(init_idx, dtype=np.int64))
        uniq = uniq[(uniq >= 0) & (uniq < n)]
        if uniq.size == 0:
            first = int(rng.integers(0, n))
            selected = [first]
        else:
            selected = uniq.tolist()

    d2 = np.full(n, np.inf, dtype=float)
    for s in selected:
        ds = np.sum((x - x[s]) ** 2, axis=1)
        d2 = np.minimum(d2, ds)

    while len(selected) < k:
        nxt = int(np.argmax(d2))
        selected.append(nxt)
        ds = np.sum((x - x[nxt]) ** 2, axis=1)
        d2 = np.minimum(d2, ds)

    return np.asarray(selected, dtype=np.int64)


def _select_sparse_subset(
    x_train: np.ndarray,
    idx_train_global: np.ndarray,
    grid_node_snapshot_idx: np.ndarray,
    max_points: int,
    selector: str,
    seed: int,
) -> np.ndarray:
    n = x_train.shape[0]
    if max_points <= 0 or max_points >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))

    if selector == "random":
        return np.sort(rng.choice(n, size=max_points, replace=False).astype(np.int64))

    if selector == "farthest":
        return np.sort(_farthest_point_sampling(x_train, max_points, seed=seed))

    global_to_local = {int(g): int(i) for i, g in enumerate(idx_train_global.tolist())}
    seeds_local = []
    for g in np.unique(grid_node_snapshot_idx).tolist():
        li = global_to_local.get(int(g), None)
        if li is not None:
            seeds_local.append(li)
    seeds_local = np.unique(np.asarray(seeds_local, dtype=np.int64)) if len(seeds_local) else np.zeros(0, dtype=np.int64)

    if seeds_local.size >= max_points:
        return np.sort(seeds_local[:max_points])

    chosen = _farthest_point_sampling(x_train, max_points, seed=seed, init_idx=seeds_local)
    return np.sort(chosen)


def _fit_rbf_model(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    kernel: str,
    epsilon: float,
    smoothing: float,
    neighbors: int,
):
    if x_fit.shape[0] < 2:
        raise RuntimeError("RBF fit requires at least 2 inducing points.")
    if float(epsilon) <= 0.0:
        raise ValueError(f"RBF epsilon must be >0. Got {epsilon}")
    if float(smoothing) < 0.0:
        raise ValueError(f"RBF smoothing must be >=0. Got {smoothing}")

    neigh = int(neighbors)
    if neigh <= 0:
        neigh = None
    else:
        neigh = int(min(neigh, x_fit.shape[0]))

    model = RBFInterpolator(
        y=x_fit,
        d=y_fit,
        kernel=str(kernel),
        epsilon=float(epsilon),
        smoothing=float(smoothing),
        neighbors=neigh,
    )
    return model, neigh


def _train_single(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    idx_train_global: np.ndarray,
    grid_node_snapshot_idx: np.ndarray,
    args: argparse.Namespace,
):
    subset_local = _select_sparse_subset(
        x_train=x_train,
        idx_train_global=idx_train_global,
        grid_node_snapshot_idx=grid_node_snapshot_idx,
        max_points=int(args.rbf_sparse_inducing_points),
        selector=str(args.rbf_inducing_selector),
        seed=int(args.seed),
    )

    x_fit = x_train[subset_local]
    y_fit = y_train[subset_local]

    model, neigh_eff = _fit_rbf_model(
        x_fit=x_fit,
        y_fit=y_fit,
        kernel=args.rbf_kernel,
        epsilon=float(args.rbf_epsilon),
        smoothing=float(args.rbf_smoothing),
        neighbors=int(args.rbf_neighbors),
    )

    y_pred_all = model(x_all)

    info = {
        "mode": "single",
        "kernel": str(args.rbf_kernel),
        "epsilon": float(args.rbf_epsilon),
        "smoothing": float(args.rbf_smoothing),
        "neighbors": None if neigh_eff is None else int(neigh_eff),
        "inducing_points": int(x_fit.shape[0]),
        "inducing_selector": str(args.rbf_inducing_selector),
        "subset_local_indices": subset_local,
    }
    return model, np.asarray(y_pred_all, dtype=float), info


def _train_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    y_phys: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    idx_train_global: np.ndarray,
    idx_val: np.ndarray,
    grid_node_snapshot_idx: np.ndarray,
    args: argparse.Namespace,
):
    eps_list = _parse_csv_float_list(args.rbf_grid_epsilons, "rbf-grid-epsilons")
    smooth_list = _parse_csv_float_list(args.rbf_grid_smoothings, "rbf-grid-smoothings")
    neigh_list = _parse_csv_int_list(args.rbf_grid_neighbors, "rbf-grid-neighbors")
    inducing_list = _parse_csv_int_list(args.rbf_grid_inducing_points, "rbf-grid-inducing-points")

    x_val = x_all[idx_val]
    y_val_true = y_phys[idx_val]

    def _denorm(yz: np.ndarray) -> np.ndarray:
        return yz * y_std[None, :] + y_mean[None, :]

    candidate_rows = []
    best = None

    total_candidates = len(eps_list) * len(smooth_list) * len(neigh_list) * len(inducing_list)
    print(f"  [RBF-GRID] candidates = {total_candidates}")

    for m in inducing_list:
        subset_local = _select_sparse_subset(
            x_train=x_train,
            idx_train_global=idx_train_global,
            grid_node_snapshot_idx=grid_node_snapshot_idx,
            max_points=int(m),
            selector=str(args.rbf_inducing_selector),
            seed=int(args.seed),
        )

        x_fit = x_train[subset_local]
        y_fit = y_train[subset_local]

        for eps in eps_list:
            for smooth in smooth_list:
                for neigh in neigh_list:
                    row = {
                        "inducing_points_request": int(m),
                        "inducing_points_effective": int(x_fit.shape[0]),
                        "epsilon": float(eps),
                        "smoothing": float(smooth),
                        "neighbors_request": int(neigh),
                        "ok": 0,
                        "error": "",
                    }

                    try:
                        model, neigh_eff = _fit_rbf_model(
                            x_fit=x_fit,
                            y_fit=y_fit,
                            kernel=args.rbf_kernel,
                            epsilon=float(eps),
                            smoothing=float(smooth),
                            neighbors=int(neigh),
                        )
                        y_val_pred_z = model(x_val)
                        y_val_pred = _denorm(np.asarray(y_val_pred_z, dtype=float))
                        m_val = _compute_metrics(y_val_true, y_val_pred)

                        row["ok"] = 1
                        row["neighbors_effective"] = None if neigh_eff is None else int(neigh_eff)
                        row["val_rel_l2"] = float(m_val["rel_l2"])
                        row["val_rmse"] = float(m_val["rmse"])
                        row["val_max_abs"] = float(m_val["max_abs"])

                        score = row["val_rel_l2"] if args.rbf_grid_metric == "val_rel_l2" else row["val_rmse"]
                        row["score"] = float(score)

                        if best is None or score < best["score"]:
                            best = {
                                "score": float(score),
                                "epsilon": float(eps),
                                "smoothing": float(smooth),
                                "neighbors": int(neigh),
                                "neighbors_effective": None if neigh_eff is None else int(neigh_eff),
                                "inducing_points_request": int(m),
                                "inducing_points_effective": int(x_fit.shape[0]),
                                "subset_local_indices": subset_local.copy(),
                            }
                    except Exception as exc:
                        row["error"] = str(exc)

                    candidate_rows.append(row)

    if best is None:
        raise RuntimeError("RBF grid-search failed: no candidate completed successfully.")

    x_fit_best = x_train[best["subset_local_indices"]]
    y_fit_best = y_train[best["subset_local_indices"]]
    model, neigh_eff = _fit_rbf_model(
        x_fit=x_fit_best,
        y_fit=y_fit_best,
        kernel=args.rbf_kernel,
        epsilon=float(best["epsilon"]),
        smoothing=float(best["smoothing"]),
        neighbors=int(best["neighbors"]),
    )
    y_pred_all = model(x_all)

    info = {
        "mode": "grid_search",
        "kernel": str(args.rbf_kernel),
        "grid_metric": str(args.rbf_grid_metric),
        "inducing_selector": str(args.rbf_inducing_selector),
        "candidates_total": int(total_candidates),
        "candidates_ok": int(sum(int(r.get("ok", 0)) for r in candidate_rows)),
        "best": {
            "epsilon": float(best["epsilon"]),
            "smoothing": float(best["smoothing"]),
            "neighbors": int(best["neighbors"]),
            "neighbors_effective": None if neigh_eff is None else int(neigh_eff),
            "inducing_points_request": int(best["inducing_points_request"]),
            "inducing_points_effective": int(best["inducing_points_effective"]),
            "score": float(best["score"]),
        },
        "subset_local_indices": best["subset_local_indices"],
        "candidate_rows": candidate_rows,
    }
    return model, np.asarray(y_pred_all, dtype=float), info


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.dataset_file, allow_pickle=True)

    mu_phys = np.asarray(data["mu_all"], dtype=float)
    q_m_phys = np.asarray(data["q_m_all"], dtype=float)
    y_z = np.asarray(data["y_all_z"], dtype=float)
    y_mean = np.asarray(data["y_mean"], dtype=float)
    y_std = np.asarray(data["y_std"], dtype=float)
    y_phys = np.asarray(data["y_all"], dtype=float)

    idx_train = np.asarray(data["idx_train"], dtype=np.int64)
    idx_val = np.asarray(data["idx_val"], dtype=np.int64)
    idx_test = np.asarray(data["idx_test"], dtype=np.int64)

    grid_node_snapshot_idx = np.asarray(data["grid_node_snapshot_idx"], dtype=np.int64)

    input_space = str(args.rbf_input_space).strip().lower()
    if input_space == "q_m":
        x_phys = q_m_phys
    elif input_space == "mu":
        x_phys = mu_phys
    else:
        raise RuntimeError(f"Unsupported rbf-input-space='{args.rbf_input_space}'.")

    x_mean, x_std = _fit_standardizer(x_phys[idx_train])
    x_z = _apply_standardizer(x_phys, x_mean, x_std)

    x_train = x_z[idx_train]
    y_train = y_z[idx_train]

    print("=" * 72)
    print("Stage 4: PROM-RBF training")
    print("=" * 72)
    print(f"dataset_file : {args.dataset_file}")
    print(f"out_dir      : {args.out_dir}")
    print(f"train/val/test sizes: {idx_train.size}/{idx_val.size}/{idx_test.size}")
    print(f"input_space={input_space}, dim(input)={x_z.shape[1]}, dim(target)={y_z.shape[1]}")

    def _denorm(yz: np.ndarray) -> np.ndarray:
        return yz * y_std[None, :] + y_mean[None, :]

    if int(args.rbf_grid_search) == 1:
        model, y_pred_z, rbf_info = _train_grid(
            x_train=x_train,
            y_train=y_train,
            x_all=x_z,
            y_phys=y_phys,
            y_mean=y_mean,
            y_std=y_std,
            idx_train_global=idx_train,
            idx_val=idx_val,
            grid_node_snapshot_idx=grid_node_snapshot_idx,
            args=args,
        )
    else:
        model, y_pred_z, rbf_info = _train_single(
            x_train=x_train,
            y_train=y_train,
            x_all=x_z,
            idx_train_global=idx_train,
            grid_node_snapshot_idx=grid_node_snapshot_idx,
            args=args,
        )

    np.save(
        os.path.join(args.out_dir, "rbf_subset_local_indices.npy"),
        np.asarray(rbf_info["subset_local_indices"], dtype=np.int64),
    )

    if rbf_info.get("mode") == "grid_search":
        with open(os.path.join(args.out_dir, "rbf_grid_results.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "kernel": rbf_info["kernel"],
                    "grid_metric": rbf_info["grid_metric"],
                    "inducing_selector": rbf_info["inducing_selector"],
                    "candidates_total": rbf_info["candidates_total"],
                    "candidates_ok": rbf_info["candidates_ok"],
                    "best": rbf_info["best"],
                    "candidates": rbf_info["candidate_rows"],
                },
                f,
                indent=2,
            )

    y_pred = _denorm(y_pred_z)

    m_train = _compute_metrics(y_phys[idx_train], y_pred[idx_train])
    m_val = _compute_metrics(y_phys[idx_val], y_pred[idx_val])
    m_test = _compute_metrics(y_phys[idx_test], y_pred[idx_test])

    print(
        f"[INFO] relL2 train/val/test = "
        f"{m_train['rel_l2']:.3e} / {m_val['rel_l2']:.3e} / {m_test['rel_l2']:.3e}"
    )

    with open(os.path.join(args.out_dir, "rbf_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    target_space = str(np.ravel(data["target_space"])[0]) if "target_space" in data else "both"
    mu_space = str(np.ravel(data["mu_space"])[0]) if "mu_space" in data else "gx_gxy"
    mapping = str(np.ravel(data["mapping"])[0]) if "mapping" in data else "green_lagrange_upper"
    np.savez(
        os.path.join(args.out_dir, "rbf_model_meta.npz"),
        input_space=np.array([input_space], dtype=object),
        x_mean=np.asarray(x_mean, dtype=float),
        x_std=np.asarray(x_std, dtype=float),
        y_mean=np.asarray(y_mean, dtype=float),
        y_std=np.asarray(y_std, dtype=float),
        q_m_dim=np.array([int(q_m_phys.shape[1])], dtype=np.int64),
        q_s_dim=np.array([int(np.asarray(data["q_s_all"]).shape[1])], dtype=np.int64) if "q_s_all" in data else np.array([-1], dtype=np.int64),
        q_pod_dim=np.array([int(np.asarray(data["q_pod_all"]).shape[1])], dtype=np.int64),
        target_space=np.array([target_space], dtype=object),
        mu_space=np.array([mu_space], dtype=object),
        mapping=np.array([mapping], dtype=object),
    )

    np.savez(
        os.path.join(args.out_dir, "predictions_rbf.npz"),
        y_pred_all=y_pred,
        y_true_all=y_phys,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        y_pred_train=y_pred[idx_train],
        y_pred_val=y_pred[idx_val],
        y_pred_test=y_pred[idx_test],
        y_true_train=y_phys[idx_train],
        y_true_val=y_phys[idx_val],
        y_true_test=y_phys[idx_test],
    )

    settings = {
        "mode": rbf_info["mode"],
        "kernel": str(args.rbf_kernel),
        "inducing_selector": rbf_info.get("inducing_selector", str(args.rbf_inducing_selector)),
        "inducing_points": int(
            rbf_info["best"]["inducing_points_effective"]
            if rbf_info.get("mode") == "grid_search"
            else rbf_info["inducing_points"]
        ),
    }
    if rbf_info.get("mode") == "grid_search":
        settings.update(
            {
                "grid_metric": rbf_info["grid_metric"],
                "best": rbf_info["best"],
                "candidates_total": int(rbf_info["candidates_total"]),
                "candidates_ok": int(rbf_info["candidates_ok"]),
            }
        )
    else:
        settings.update(
            {
                "epsilon": float(rbf_info["epsilon"]),
                "smoothing": float(rbf_info["smoothing"]),
                "neighbors": rbf_info["neighbors"],
            }
        )

    summary = {
        "dataset_file": args.dataset_file,
        "seed": int(args.seed),
        "sizes": {
            "train": int(idx_train.size),
            "val": int(idx_val.size),
            "test": int(idx_test.size),
        },
        "dim": {
            "input": int(x_z.shape[1]),
            "target": int(y_z.shape[1]),
        },
        "input_space": input_space,
        "rbf": {
            "train": m_train,
            "val": m_val,
            "test": m_test,
            "settings": settings,
        },
    }

    with open(os.path.join(args.out_dir, "stage4_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "stage4_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[OK] Stage 4 outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
