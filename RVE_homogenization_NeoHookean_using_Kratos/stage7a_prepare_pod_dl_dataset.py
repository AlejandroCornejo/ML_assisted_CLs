#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 7a (POD-DL): build q snapshots from Stage-1 displacements and Stage-2 POD basis.
"""

import os
import argparse
import numpy as np

from stage7a_prepare_ann_rbf_dataset import (
    _build_affine_lifting_helpers,
    _compute_affine_free_displacement,
)


def collect_pod_dl_q_dataset(fom_dir="stage_1_training_set_fom", basis_dir="stage_2_pod_rve", q_dim=9):
    basis = np.asarray(np.load(os.path.join(basis_dir, "pod_basis_free.npy")), dtype=np.float64)
    free_dofs = np.asarray(np.load(os.path.join(basis_dir, "free_dofs.npy")), dtype=np.int64)
    dir_dofs = np.asarray(np.load(os.path.join(basis_dir, "dirichlet_dofs.npy")), dtype=np.int64)
    eq_map = np.asarray(np.load(os.path.join(basis_dir, "eq_map.npy")), dtype=np.int64)

    n_total_basis = int(basis.shape[1])
    q_dim = int(q_dim)
    if q_dim < 1 or q_dim > n_total_basis:
        raise ValueError(f"q_dim must satisfy 1 <= q_dim <= {n_total_basis}. Got {q_dim}.")
    phi_q = basis[:, :q_dim].copy()

    n_total_dofs = int(len(free_dofs) + len(dir_dofs))
    n_total_runtime, dof_x, dof_y, is_x_dof, eq_map_runtime = _build_affine_lifting_helpers()
    if n_total_runtime != n_total_dofs:
        raise RuntimeError(
            f"Runtime/model DOF mismatch: runtime={n_total_runtime}, expected={n_total_dofs}."
        )
    if eq_map_runtime.shape == eq_map.shape and not np.array_equal(eq_map_runtime, eq_map):
        raise RuntimeError("eq_map mismatch between runtime model and Stage 2 metadata.")

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    traj_dirs = [
        d
        for d in os.listdir(fom_dir)
        if os.path.isdir(os.path.join(fom_dir, d)) and d.startswith("trajectory_")
    ]
    traj_dirs.sort(key=lambda x: int(x.split("_")[1]))
    if not traj_dirs:
        raise RuntimeError(f"No trajectory_* folders found in {fom_dir}.")

    q_list = []
    steps_per_traj = []
    used_traj = []

    for d in traj_dirs:
        idx = int(d.split("_")[1])
        traj_dir = os.path.join(fom_dir, d)
        u_file = os.path.join(traj_dir, f"trajectory_{idx}_U.npy")
        e_file = os.path.join(traj_dir, f"trajectory_{idx}_applied_strain.npy")
        if not (os.path.exists(u_file) and os.path.exists(e_file)):
            continue

        U = np.asarray(np.load(u_file), dtype=np.float64)
        E_hist = np.asarray(np.load(e_file), dtype=np.float64)
        if U.ndim != 2 or E_hist.ndim != 2 or E_hist.shape[1] != 3:
            continue

        if U.shape[1] == n_total_dofs:
            U_free = U[:, free_dofs]
        elif U.shape[0] == n_total_dofs:
            U_free = U[free_dofs, :].T
        else:
            continue

        n_steps = min(int(U_free.shape[0]), int(E_hist.shape[0]))
        if n_steps <= 0:
            continue
        U_free = U_free[:n_steps, :]
        E_hist = E_hist[:n_steps, :]

        q_traj = np.empty((n_steps, q_dim), dtype=np.float64)
        for k in range(n_steps):
            u_aff = _compute_affine_free_displacement(E_hist[k], x_free, y_free, is_x_free)
            w_free = U_free[k] - u_aff
            q_traj[k, :] = w_free @ phi_q

        q_list.append(q_traj)
        steps_per_traj.append(n_steps)
        used_traj.append(idx)

    if not q_list:
        raise RuntimeError("No valid snapshots found to build POD-DL dataset.")

    q_data = np.vstack(q_list)
    return q_data, phi_q, used_traj, steps_per_traj


def prepare_pod_dl_dataset(
    fom_dir="stage_1_training_set_fom",
    basis_dir="stage_2_pod_rve",
    out_dir="stage_7_pod_dl_data",
    q_dim=9,
):
    os.makedirs(out_dir, exist_ok=True)
    q_data, phi_q, used_traj, steps_per_traj = collect_pod_dl_q_dataset(
        fom_dir=fom_dir,
        basis_dir=basis_dir,
        q_dim=q_dim,
    )

    np.save(os.path.join(out_dir, "q_dataset.npy"), q_data)
    np.save(os.path.join(out_dir, "phi_q.npy"), phi_q)
    np.savez(
        os.path.join(out_dir, "pod_dl_dataset_metadata.npz"),
        q_dim=np.array([int(q_dim)], dtype=np.int64),
        n_samples=np.array([int(q_data.shape[0])], dtype=np.int64),
        used_trajectories=np.asarray(used_traj, dtype=np.int64),
        steps_per_trajectory=np.asarray(steps_per_traj, dtype=np.int64),
        fom_dir=np.array([str(fom_dir)]),
        basis_dir=np.array([str(basis_dir)]),
    )

    print("=" * 70)
    print("Stage 7a (POD-DL): dataset ready")
    print("=" * 70)
    print(f"q_data shape: {q_data.shape}")
    print(f"phi_q shape: {phi_q.shape}")
    print(f"used trajectories: {used_traj}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 7a (POD-DL): build q dataset for POD-DL training.")
    p.add_argument("--fom-dir", type=str, default="stage_1_training_set_fom", help="Stage-1 FOM data directory.")
    p.add_argument("--basis-dir", type=str, default="stage_2_pod_rve", help="Stage-2 POD data directory.")
    p.add_argument("--out-dir", type=str, default="stage_7_pod_dl_data", help="Output directory for q dataset.")
    p.add_argument("--q-dim", type=int, default=9, help="Number of POD coordinates used for POD-DL.")
    args = p.parse_args()

    prepare_pod_dl_dataset(
        fom_dir=args.fom_dir,
        basis_dir=args.basis_dir,
        out_dir=args.out_dir,
        q_dim=args.q_dim,
    )

