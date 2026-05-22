#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 5 (2D-MAWECM): compare PROM-RBF vs FOM on one trajectory.

This is an offline trajectory comparison stage (no nonlinear solve here).
It evaluates the trained RBF surrogate on the trajectory samples and compares
against FOM-derived reduced quantities stored in Stage3.
"""

import argparse
import json
import os
import pickle

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage5: PROM-RBF vs FOM comparison on one trajectory")
    p.add_argument(
        "--dataset-file",
        type=str,
        default="stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    )
    p.add_argument("--rbf-dir", type=str, default="stage_4_prom_rbf_grid")
    p.add_argument(
        "--trajectory-index",
        type=str,
        default="auto_test",
        help="Integer trajectory id, or 'auto_test' to pick the trajectory with most test samples.",
    )
    p.add_argument(
        "--subset",
        type=str,
        default="test",
        choices=["all", "train", "val", "test"],
        help="Subset of Stage3 split used for the comparison inside the selected trajectory.",
    )
    p.add_argument("--out-dir", type=str, default="stage_5_prom_rbf_vs_fom")
    p.add_argument("--save-plots", type=int, default=1, choices=[0, 1])
    return p.parse_args()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_pred - y_true
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(y_true), 1e-30))
    rmse = float(np.sqrt(np.mean(err * err)))
    max_abs = float(np.max(np.abs(err)))
    return {
        "rel_l2": rel_l2,
        "rmse": rmse,
        "max_abs": max_abs,
    }


def _compute_metrics_per_component(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_pred - y_true
    rel_l2 = np.linalg.norm(err, axis=0) / np.maximum(np.linalg.norm(y_true, axis=0), 1e-30)
    rmse = np.sqrt(np.mean(err * err, axis=0))
    max_abs = np.max(np.abs(err), axis=0)
    return {
        "rel_l2": rel_l2,
        "rmse": rmse,
        "max_abs": max_abs,
    }


def _split_masks_for_indices(indices: np.ndarray, idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray):
    set_train = set(idx_train.tolist())
    set_val = set(idx_val.tolist())
    set_test = set(idx_test.tolist())

    in_train = np.array([i in set_train for i in indices], dtype=bool)
    in_val = np.array([i in set_val for i in indices], dtype=bool)
    in_test = np.array([i in set_test for i in indices], dtype=bool)
    return in_train, in_val, in_test


def _extract_q_predictions(
    target_space: str,
    y_pred: np.ndarray,
    q_m_dim: int,
    q_s_dim: int,
    q_pod_dim: int,
    q_m_ref: np.ndarray,
    a_m: np.ndarray,
    c_m: np.ndarray,
    c_s: np.ndarray,
):
    q_m_pred = None
    q_s_pred = None
    q_pod_pred = None

    if target_space == "q_m":
        q_m_pred = y_pred
    elif target_space == "q_s":
        q_s_pred = y_pred
    elif target_space == "q_pod":
        q_pod_pred = y_pred
    elif target_space == "both_ms":
        q_m_pred = y_pred[:, :q_m_dim]
        q_s_pred = y_pred[:, q_m_dim:q_m_dim + q_s_dim]
    elif target_space == "both":
        q_m_pred = y_pred[:, :q_m_dim]
        q_pod_pred = y_pred[:, q_m_dim:q_m_dim + q_pod_dim]
    else:
        raise RuntimeError(f"Unsupported target_space in Stage3 dataset: {target_space}")

    if q_m_pred is None:
        q_m_pred = np.asarray(q_m_ref, dtype=float)

    if q_pod_pred is None:
        if q_s_pred is None:
            raise RuntimeError("Cannot reconstruct q_pod: missing q_s prediction.")
        q_m_block = (a_m @ q_m_pred.T).T
        q_pod_pred = (c_m @ q_m_block.T).T + (c_s @ q_s_pred.T).T

    return q_m_pred, q_s_pred, q_pod_pred


def _maybe_plot(
    out_dir: str,
    mu: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q_pod_true: np.ndarray,
    q_pod_pred: np.ndarray,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available. Skipping plots.")
        return

    n_steps = y_true.shape[0]
    k_show = min(3, y_true.shape[1])
    x = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    if mu.shape[1] >= 2:
        ax.plot(mu[:, 0], mu[:, 1], "-", lw=1.5)
        ax.scatter([mu[0, 0]], [mu[0, 1]], s=25, label="start")
        ax.scatter([mu[-1, 0]], [mu[-1, 1]], s=25, label="end")
        ax.set_xlabel("mu_1")
        ax.set_ylabel("mu_2")
        ax.set_title("Trajectory in parameter space")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "trajectory_param_path.png"), dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(k_show, 1, figsize=(9.0, 2.6 * k_show), sharex=True)
    if k_show == 1:
        axs = [axs]
    for j in range(k_show):
        axs[j].plot(x, y_true[:, j], "-", lw=1.4, label="FOM")
        axs[j].plot(x, y_pred[:, j], "--", lw=1.3, label="PROM-RBF")
        axs[j].set_ylabel(f"y[{j}]")
        axs[j].grid(True, alpha=0.25)
    axs[0].legend(loc="best")
    axs[-1].set_xlabel("snapshot index")
    fig.suptitle("Target-space comparison (first components)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_components_comparison.png"), dpi=180)
    plt.close(fig)

    rel_err_step = np.linalg.norm(y_pred - y_true, axis=1) / np.maximum(np.linalg.norm(y_true, axis=1), 1e-30)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.plot(x, rel_err_step, "-", lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("snapshot index")
    ax.set_ylabel("relative error")
    ax.set_title("Per-snapshot relative error in target space")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_rel_error_per_snapshot.png"), dpi=180)
    plt.close(fig)

    if q_pod_true is not None and q_pod_pred is not None:
        kq = min(3, q_pod_true.shape[1])
        fig, axs = plt.subplots(kq, 1, figsize=(9.0, 2.6 * kq), sharex=True)
        if kq == 1:
            axs = [axs]
        for j in range(kq):
            axs[j].plot(x, q_pod_true[:, j], "-", lw=1.4, label="FOM q_pod")
            axs[j].plot(x, q_pod_pred[:, j], "--", lw=1.3, label="PROM q_pod")
            axs[j].set_ylabel(f"q_pod[{j}]")
            axs[j].grid(True, alpha=0.25)
        axs[0].legend(loc="best")
        axs[-1].set_xlabel("snapshot index")
        fig.suptitle("q_pod comparison (first components)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "qpod_components_comparison.png"), dpi=180)
        plt.close(fig)


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.rbf_dir, "rbf_model.pkl"), "rb") as f:
        rbf_model = pickle.load(f)

    meta_file = os.path.join(args.rbf_dir, "rbf_model_meta.npz")
    meta = np.load(meta_file, allow_pickle=True) if os.path.exists(meta_file) else None

    data = np.load(args.dataset_file, allow_pickle=True)

    mu_all = np.asarray(data["mu_all"], dtype=float)
    y_all = np.asarray(data["y_all"], dtype=float)
    q_m_all = np.asarray(data["q_m_all"], dtype=float)
    q_s_all = np.asarray(data["q_s_all"], dtype=float) if "q_s_all" in data else None
    q_pod_all = np.asarray(data["q_pod_all"], dtype=float)
    traj_ids = np.asarray(data["traj_ids_all"], dtype=np.int64)

    if meta is not None:
        input_space = str(np.ravel(meta["input_space"])[0]).strip().lower()
        x_mean = np.asarray(meta["x_mean"], dtype=float)
        x_std = np.asarray(meta["x_std"], dtype=float)
        y_mean = np.asarray(meta["y_mean"], dtype=float)
        y_std = np.asarray(meta["y_std"], dtype=float)
    else:
        input_space = "mu"
        x_mean = np.asarray(data["mu_mean"], dtype=float)
        x_std = np.asarray(data["mu_std"], dtype=float)
        y_mean = np.asarray(data["y_mean"], dtype=float)
        y_std = np.asarray(data["y_std"], dtype=float)

    idx_train = np.asarray(data["idx_train"], dtype=np.int64)
    idx_val = np.asarray(data["idx_val"], dtype=np.int64)
    idx_test = np.asarray(data["idx_test"], dtype=np.int64)

    target_space = str(np.ravel(data["target_space"])[0]) if "target_space" in data else "both"
    mu_space = str(np.ravel(data["mu_space"])[0]) if "mu_space" in data else "gx_gxy"

    if args.subset == "train":
        subset_global_mask = np.zeros_like(traj_ids, dtype=bool)
        subset_global_mask[idx_train] = True
    elif args.subset == "val":
        subset_global_mask = np.zeros_like(traj_ids, dtype=bool)
        subset_global_mask[idx_val] = True
    elif args.subset == "test":
        subset_global_mask = np.zeros_like(traj_ids, dtype=bool)
        subset_global_mask[idx_test] = True
    else:
        subset_global_mask = np.ones_like(traj_ids, dtype=bool)

    if str(args.trajectory_index).lower() in ("auto_test", "test", "auto"):
        best_traj = None
        best_count = -1
        for t in np.unique(traj_ids):
            c = int(np.sum((traj_ids == t) & subset_global_mask))
            if c > best_count:
                best_count = c
                best_traj = int(t)
        if best_traj is None or best_count <= 0:
            raise RuntimeError(f"No samples available for subset='{args.subset}'.")
        selected_traj = best_traj
    else:
        selected_traj = int(args.trajectory_index)

    mask = (traj_ids == selected_traj) & subset_global_mask
    if not np.any(mask):
        raise RuntimeError(
            f"No samples found for trajectory-index={selected_traj} in subset='{args.subset}'."
        )

    ids = np.nonzero(mask)[0].astype(np.int64)

    mu_t = mu_all[ids]
    y_true = y_all[ids]
    q_m_true = q_m_all[ids]
    q_s_true = q_s_all[ids] if q_s_all is not None else None
    q_pod_true = q_pod_all[ids]

    a_m = np.asarray(data["A_m"], dtype=float) if "A_m" in data else None
    c_m = np.asarray(data["C_m"], dtype=float) if "C_m" in data else None
    c_s = np.asarray(data["C_s"], dtype=float) if "C_s" in data else None
    if a_m is None or c_m is None or c_s is None:
        # Fallback to Stage2b artifacts colocated in default folder layout.
        ds_dir = os.path.dirname(os.path.abspath(args.dataset_file))
        root_dir = os.path.dirname(ds_dir)
        s2b = os.path.join(root_dir, "stage_2b_ls_master")
        a_m = np.asarray(np.load(os.path.join(s2b, "A_m.npy")), dtype=float)
        c_m = np.asarray(np.load(os.path.join(s2b, "C_m.npy")), dtype=float)
        c_s = np.asarray(np.load(os.path.join(s2b, "C_s.npy")), dtype=float)

    if input_space == "q_m":
        x_phys = q_m_true
    elif input_space == "mu":
        x_phys = mu_t
    else:
        raise RuntimeError(f"Unsupported RBF input_space='{input_space}'.")

    xz = (x_phys - x_mean[None, :]) / x_std[None, :]
    yz_pred = np.asarray(rbf_model(xz), dtype=float)
    y_pred = yz_pred * y_std[None, :] + y_mean[None, :]

    q_m_dim = q_m_all.shape[1]
    q_s_dim = int(q_s_all.shape[1]) if q_s_all is not None else int(c_s.shape[1])
    q_pod_dim = q_pod_all.shape[1]
    q_m_pred, q_s_pred, q_pod_pred = _extract_q_predictions(
        target_space=target_space,
        y_pred=y_pred,
        q_m_dim=q_m_dim,
        q_s_dim=q_s_dim,
        q_pod_dim=q_pod_dim,
        q_m_ref=q_m_true,
        a_m=a_m,
        c_m=c_m,
        c_s=c_s,
    )

    metrics_y = _compute_metrics(y_true, y_pred)
    metrics_y_comp = _compute_metrics_per_component(y_true, y_pred)

    metrics_qm = _compute_metrics(q_m_true, q_m_pred) if q_m_pred is not None else None
    metrics_qs = _compute_metrics(q_s_true, q_s_pred) if (q_s_true is not None and q_s_pred is not None) else None
    metrics_qpod = _compute_metrics(q_pod_true, q_pod_pred) if q_pod_pred is not None else None

    in_train, in_val, in_test = _split_masks_for_indices(ids, idx_train, idx_val, idx_test)

    summary = {
        "trajectory_index": int(selected_traj),
        "subset": str(args.subset),
        "n_samples": int(ids.size),
        "mu_space": mu_space,
        "rbf_input_space": input_space,
        "target_space": target_space,
        "membership": {
            "train": int(np.sum(in_train)),
            "val": int(np.sum(in_val)),
            "test": int(np.sum(in_test)),
        },
        "metrics_target": metrics_y,
        "metrics_target_first_components": {
            "rel_l2": [float(x) for x in metrics_y_comp["rel_l2"][: min(5, y_true.shape[1])]],
            "rmse": [float(x) for x in metrics_y_comp["rmse"][: min(5, y_true.shape[1])]],
            "max_abs": [float(x) for x in metrics_y_comp["max_abs"][: min(5, y_true.shape[1])]],
        },
        "metrics_q_m": metrics_qm,
        "metrics_q_s": metrics_qs,
        "metrics_q_pod": metrics_qpod,
    }

    np.savez(
        os.path.join(args.out_dir, f"trajectory_{selected_traj}_subset_{args.subset}_prom_rbf_vs_fom.npz"),
        sample_indices=ids,
        mu=mu_t,
        y_true=y_true,
        y_pred=y_pred,
        q_m_true=q_m_true,
        q_m_pred=(q_m_pred if q_m_pred is not None else np.zeros((0, 0), dtype=float)),
        q_pod_true=q_pod_true,
        q_pod_pred=(q_pod_pred if q_pod_pred is not None else np.zeros((0, 0), dtype=float)),
        in_train=in_train.astype(np.uint8),
        in_val=in_val.astype(np.uint8),
        in_test=in_test.astype(np.uint8),
    )

    with open(
        os.path.join(args.out_dir, f"trajectory_{selected_traj}_subset_{args.subset}_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    if int(args.save_plots) == 1:
        _maybe_plot(
            out_dir=args.out_dir,
            mu=mu_t,
            y_true=y_true,
            y_pred=y_pred,
            q_pod_true=q_pod_true,
            q_pod_pred=q_pod_pred,
        )

    print("=" * 72)
    print("Stage 5: PROM-RBF vs FOM (trajectory comparison)")
    print("=" * 72)
    print(f"trajectory-index: {selected_traj}")
    print(f"subset          : {args.subset}")
    print(f"n_samples       : {ids.size}")
    print(
        "membership      : "
        f"train={np.sum(in_train)}, val={np.sum(in_val)}, test={np.sum(in_test)}"
    )
    print(
        "target relL2    : "
        f"{metrics_y['rel_l2']:.3e} (rmse={metrics_y['rmse']:.3e}, max_abs={metrics_y['max_abs']:.3e})"
    )
    if metrics_qm is not None:
        print(f"q_m relL2       : {metrics_qm['rel_l2']:.3e}")
    if metrics_qpod is not None:
        print(f"q_pod relL2     : {metrics_qpod['rel_l2']:.3e}")
    print(f"[OK] Outputs in : {args.out_dir}")


if __name__ == "__main__":
    main()
