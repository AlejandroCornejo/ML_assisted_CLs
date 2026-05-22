#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 3 (2D-MAWECM): build PROM regression dataset from Stage2 outputs.

Purpose
- Prepare a clean dataset to train PROM surrogates (RBF) before MAW-ECM.
- Keep Joaquín-style parameterization as default: mu = [Gx, Gxy].
- Preserve structured Stage0 topology (nodes/quads/graph/Laplacian).

Inputs
- Stage2a POD data (q_pod, applied strain, snapshot trajectory ids)
- Stage2b LS master data (mu, q_m, master operators)
- Stage0 structured mesh bundle (for topology + node matching)

Outputs (out-dir)
- stage3_prom_regression_dataset.npz
- stage3_summary.json
- optional plots
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage3: PROM regression dataset builder (2D-MAWECM)")
    p.add_argument("--stage2a-dir", type=str, default="stage_2a_pod_data")
    p.add_argument("--stage2b-dir", type=str, default="stage_2b_ls_master")
    p.add_argument("--stage0-file", type=str, default="stage_0_trajectory/stage_0_trajectories.npz")
    p.add_argument("--out-dir", type=str, default="stage_3_prom_regression_dataset")

    p.add_argument(
        "--target-space",
        type=str,
        default="q_s",
        choices=["q_m", "q_s", "q_pod", "both", "both_ms"],
        help="Primary regression targets to export for downstream training.",
    )
    p.add_argument(
        "--mu-space",
        type=str,
        default="gx_gxy",
        choices=["gx_gxy", "f11_f12"],
        help="Input parameter space for regression dataset.",
    )

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--structured-match-tol",
        type=float,
        default=1e-8,
        help="Maximum accepted nearest-neighbor distance when mapping structured nodes to snapshots.",
    )
    p.add_argument("--save-plots", type=int, default=1, choices=[0, 1])
    return p.parse_args()


def _load_required_arrays(stage2a_dir: str, stage2b_dir: str, mu_space: str):
    q_pod = np.load(os.path.join(stage2a_dir, "q_pod_train.npy"))
    eps = np.load(os.path.join(stage2a_dir, "applied_strain_train.npy"))

    traj_ids_file = os.path.join(stage2a_dir, "snapshot_trajectory_ids.npy")
    if os.path.exists(traj_ids_file):
        traj_ids = np.load(traj_ids_file).astype(np.int64)
    else:
        traj_ids = np.zeros(q_pod.shape[0], dtype=np.int64)

    if mu_space == "gx_gxy":
        mu = np.load(os.path.join(stage2b_dir, "mu_train_gx_gxy.npy"))
    else:
        mu = np.load(os.path.join(stage2b_dir, "mu_train_f11_f12.npy"))

    q_m = np.load(os.path.join(stage2b_dir, "q_m_train.npy"))
    q_s_file = os.path.join(stage2b_dir, "q_s_train.npy")
    if not os.path.exists(q_s_file):
        raise FileNotFoundError(
            f"{q_s_file} not found. Rebuild Stage2b with master/slave outputs."
        )
    q_s = np.load(q_s_file)
    a_m = np.load(os.path.join(stage2b_dir, "A_m.npy"))
    c_m = np.load(os.path.join(stage2b_dir, "C_m.npy"))
    c_s = np.load(os.path.join(stage2b_dir, "C_s.npy"))

    if (
        q_pod.shape[0] != mu.shape[0]
        or q_m.shape[0] != mu.shape[0]
        or q_s.shape[0] != mu.shape[0]
        or eps.shape[0] != mu.shape[0]
    ):
        raise RuntimeError(
            "Sample count mismatch across stage2 outputs: "
            f"q_pod={q_pod.shape[0]}, q_m={q_m.shape[0]}, q_s={q_s.shape[0]}, "
            f"mu={mu.shape[0]}, eps={eps.shape[0]}"
        )

    return q_pod, q_m, q_s, mu, eps, traj_ids, a_m, c_m, c_s


def _check_split_args(train_frac: float, val_frac: float) -> float:
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train-frac must be in (0,1). Got {train_frac}")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError(f"val-frac must be in [0,1). Got {val_frac}")
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0.0:
        raise ValueError(
            f"Invalid fractions: train+val must be < 1. Got train={train_frac}, val={val_frac}"
        )
    return test_frac


def _random_split(n: int, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)

    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise RuntimeError(
            f"Split sizes must be positive. Got n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )

    idx_train = np.sort(perm[:n_train])
    idx_val = np.sort(perm[n_train:n_train + n_val])
    idx_test = np.sort(perm[n_train + n_val:])

    return idx_train, idx_val, idx_test


def _fit_standardizer(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-14, 1.0, std)
    return mean, std


def _apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, :]) / std[None, :]


def _nearest_neighbors(query: np.ndarray, cloud: np.ndarray):
    # Pure NumPy nearest-neighbor (small problem sizes, deterministic, no extra deps).
    # query: [Nq,d], cloud: [Ns,d]
    d2 = np.sum((query[:, None, :] - cloud[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(query.shape[0]), idx])
    return idx.astype(np.int64), dist.astype(float)


def _load_stage0_topology(stage0_file: str):
    if not os.path.exists(stage0_file):
        raise FileNotFoundError(f"Stage0 file not found: {stage0_file}")

    s0 = np.load(stage0_file, allow_pickle=True)
    required = [
        "grid_nodes_param",
        "grid_cells_quad",
        "grid_graph_edges",
        "grid_graph_laplacian",
        "structured_mesh_shape",
    ]
    missing = [k for k in required if k not in s0]
    if missing:
        raise RuntimeError(f"Stage0 file missing keys: {missing}")

    mapping = str(np.ravel(s0["mapping"])[0]) if "mapping" in s0 else "unknown"

    out = {
        "grid_nodes_param": np.asarray(s0["grid_nodes_param"], dtype=float),
        "grid_cells_quad": np.asarray(s0["grid_cells_quad"], dtype=np.int64),
        "grid_graph_edges": np.asarray(s0["grid_graph_edges"], dtype=np.int64),
        "grid_graph_laplacian": np.asarray(s0["grid_graph_laplacian"], dtype=float),
        "structured_mesh_shape": np.asarray(s0["structured_mesh_shape"], dtype=np.int64),
        "mapping": mapping,
    }

    # Optional trajectory-param paths from Stage0 (still in param space).
    for k in ("trajectory_param_1", "trajectory_param_2", "trajectory_labels"):
        if k in s0:
            out[k] = np.asarray(s0[k])

    return out


def _slice_targets(target_space: str, q_pod: np.ndarray, q_m: np.ndarray, q_s: np.ndarray):
    if target_space == "q_m":
        return q_m
    if target_space == "q_s":
        return q_s
    if target_space == "q_pod":
        return q_pod
    if target_space == "both_ms":
        return np.hstack([q_m, q_s])
    return np.hstack([q_m, q_pod])


def _save_plots(out_dir: str, mu: np.ndarray, idx_train, idx_val, idx_test, grid_nodes, node_dist):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib unavailable. Skipping Stage3 plots.")
        return

    if mu.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.scatter(mu[idx_train, 0], mu[idx_train, 1], s=5, alpha=0.45, label="train")
    ax.scatter(mu[idx_val, 0], mu[idx_val, 1], s=8, alpha=0.55, label="val")
    ax.scatter(mu[idx_test, 0], mu[idx_test, 1], s=8, alpha=0.55, label="test")
    ax.scatter(grid_nodes[:, 0], grid_nodes[:, 1], s=10, marker="x", alpha=0.45, label="structured nodes")
    ax.set_xlabel("mu_1")
    ax.set_ylabel("mu_2")
    ax.set_title("Stage3 dataset in parameter space")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage3_param_space_splits.png"), dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
    ax2.hist(node_dist, bins=40, alpha=0.8)
    ax2.set_xlabel("nearest snapshot distance")
    ax2.set_ylabel("count")
    ax2.set_title("Structured-node matching distances")
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "stage3_structured_match_distance_hist.png"), dpi=180)
    plt.close(fig2)


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    test_frac = _check_split_args(args.train_frac, args.val_frac)

    print("=" * 76)
    print("Stage 3: PROM regression dataset")
    print("=" * 76)
    print(f"stage2a_dir  : {args.stage2a_dir}")
    print(f"stage2b_dir  : {args.stage2b_dir}")
    print(f"stage0_file  : {args.stage0_file}")
    print(f"mu_space     : {args.mu_space}")
    print(f"target_space : {args.target_space}")
    print(f"split        : train={args.train_frac:.3f}, val={args.val_frac:.3f}, test={test_frac:.3f}")

    q_pod, q_m, q_s, mu, eps, traj_ids, a_m, c_m, c_s = _load_required_arrays(
        stage2a_dir=args.stage2a_dir,
        stage2b_dir=args.stage2b_dir,
        mu_space=args.mu_space,
    )
    y = _slice_targets(args.target_space, q_pod=q_pod, q_m=q_m, q_s=q_s)

    n_samples = mu.shape[0]
    idx_train, idx_val, idx_test = _random_split(
        n=n_samples,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        seed=int(args.seed),
    )

    mu_mean, mu_std = _fit_standardizer(mu[idx_train])
    y_mean, y_std = _fit_standardizer(y[idx_train])

    mu_z = _apply_standardizer(mu, mu_mean, mu_std)
    y_z = _apply_standardizer(y, y_mean, y_std)

    stage0 = _load_stage0_topology(args.stage0_file)
    grid_nodes = stage0["grid_nodes_param"]
    node_snap_idx, node_dist = _nearest_neighbors(grid_nodes, mu)

    max_node_dist = float(np.max(node_dist)) if node_dist.size else 0.0
    unique_hits = int(np.unique(node_snap_idx).size)
    if max_node_dist > float(args.structured_match_tol):
        print(
            "[WARN] Structured-node matching max distance exceeds tolerance: "
            f"max={max_node_dist:.3e} > tol={args.structured_match_tol:.3e}"
        )

    dataset_npz = os.path.join(args.out_dir, "stage3_prom_regression_dataset.npz")
    np.savez(
        dataset_npz,
        # Core supervised arrays
        mu_all=mu,
        y_all=y,
        q_m_all=q_m,
        q_s_all=q_s,
        q_pod_all=q_pod,
        A_m=a_m,
        C_m=c_m,
        C_s=c_s,
        eps_all=eps,
        traj_ids_all=traj_ids,
        # Normalized arrays
        mu_all_z=mu_z,
        y_all_z=y_z,
        mu_mean=mu_mean,
        mu_std=mu_std,
        y_mean=y_mean,
        y_std=y_std,
        # Split indices
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        # Structured mesh topology from Stage0
        grid_nodes_param=stage0["grid_nodes_param"],
        grid_cells_quad=stage0["grid_cells_quad"],
        grid_graph_edges=stage0["grid_graph_edges"],
        grid_graph_laplacian=stage0["grid_graph_laplacian"],
        structured_mesh_shape=stage0["structured_mesh_shape"],
        # Structured-node -> snapshot mapping
        grid_node_snapshot_idx=node_snap_idx,
        grid_node_snapshot_dist=node_dist,
        grid_node_mu=mu[node_snap_idx],
        grid_node_y=y[node_snap_idx],
        grid_node_q_m=q_m[node_snap_idx],
        grid_node_q_s=q_s[node_snap_idx],
        grid_node_q_pod=q_pod[node_snap_idx],
        # Metadata
        mu_space=np.array([args.mu_space], dtype="U16"),
        target_space=np.array([args.target_space], dtype="U16"),
        mapping=np.array([stage0["mapping"]], dtype="U64"),
    )

    summary = {
        "stage2a_dir": args.stage2a_dir,
        "stage2b_dir": args.stage2b_dir,
        "stage0_file": args.stage0_file,
        "mu_space": args.mu_space,
        "target_space": args.target_space,
        "n_samples": int(n_samples),
        "dim_mu": int(mu.shape[1]),
        "dim_q_m": int(q_m.shape[1]),
        "dim_q_s": int(q_s.shape[1]),
        "dim_q_pod": int(q_pod.shape[1]),
        "dim_y": int(y.shape[1]),
        "split": {
            "train": int(idx_train.size),
            "val": int(idx_val.size),
            "test": int(idx_test.size),
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "test_frac": float(test_frac),
            "seed": int(args.seed),
        },
        "structured_mesh": {
            "n_nodes": int(stage0["grid_nodes_param"].shape[0]),
            "n_quads": int(stage0["grid_cells_quad"].shape[0]),
            "shape": [int(x) for x in stage0["structured_mesh_shape"].tolist()],
            "node_match_unique_snapshot_hits": unique_hits,
            "node_match_max_distance": float(max_node_dist),
            "node_match_mean_distance": float(np.mean(node_dist) if node_dist.size else 0.0),
        },
    }

    # Include optional Stage0 trajectories in separate files for convenience.
    for k in ("trajectory_param_1", "trajectory_param_2", "trajectory_labels"):
        if k in stage0:
            np.save(os.path.join(args.out_dir, f"stage0_{k}.npy"), stage0[k])
            summary[f"saved_{k}"] = True

    with open(os.path.join(args.out_dir, "stage3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if int(args.save_plots) == 1:
        _save_plots(
            out_dir=args.out_dir,
            mu=mu,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            grid_nodes=grid_nodes,
            node_dist=node_dist,
        )

    print(f"[INFO] dataset file: {dataset_npz}")
    print(
        "[INFO] samples: "
        f"total={n_samples}, train={idx_train.size}, val={idx_val.size}, test={idx_test.size}"
    )
    print(
        "[INFO] structured-node matching: "
        f"nodes={grid_nodes.shape[0]}, unique_hits={unique_hits}, max_dist={max_node_dist:.3e}"
    )
    print(f"[OK] Stage 3 output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
