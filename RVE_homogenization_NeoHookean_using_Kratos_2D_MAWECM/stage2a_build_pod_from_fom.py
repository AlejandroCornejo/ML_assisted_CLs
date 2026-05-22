#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 2a (2D-MAWECM): build POD basis from Stage-1 FOM snapshots.

This script is intentionally POD-only:
- reconstructs DOF partition (free/Dirichlet)
- subtracts exact finite-deformation affine lifting per snapshot
- computes POD on free fluctuations
- stores POD basis + reduced coordinates + metadata for Stage 2b (LS master)
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np


KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    ExtractDirichletBoundaryConditions,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    setup_kratos_parameters,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2a: POD from FOM snapshots (2D-MAWECM)")
    p.add_argument("--fom-dir", type=str, default="stage_1_training_set_fom")
    p.add_argument("--out-dir", type=str, default="stage_2a_pod_data")
    p.add_argument("--mesh", type=str, default="rve_geometry")
    p.add_argument("--pod-energy-loss", type=float, default=1e-8)
    p.add_argument(
        "--pod-rank",
        type=int,
        default=0,
        help="If >0, force rank. If 0, choose from --pod-energy-loss.",
    )
    p.add_argument("--save-w-free", type=int, default=0, choices=[0, 1])
    p.add_argument("--save-plots", type=int, default=1, choices=[0, 1])
    return p.parse_args()


def _discover_trajectory_dirs(root: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"FOM directory not found: {root}")
    out: List[Tuple[int, str]] = []
    pat = re.compile(r"^trajectory_(\d+)$")
    for name in os.listdir(root):
        m = pat.match(name)
        if m is None:
            continue
        idx = int(m.group(1))
        path = os.path.join(root, name)
        if os.path.isdir(path):
            out.append((idx, path))
    out.sort(key=lambda x: x[0])
    if len(out) == 0:
        raise RuntimeError(f"No trajectory_<i> folders found in {root}")
    return out


def _read_stage1_arrays(traj_dir: str, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    u_file = os.path.join(traj_dir, f"trajectory_{idx}_U.npy")
    e_file = os.path.join(traj_dir, f"trajectory_{idx}_applied_strain.npy")
    if not os.path.exists(u_file):
        raise FileNotFoundError(f"Missing displacement snapshots: {u_file}")
    if not os.path.exists(e_file):
        raise FileNotFoundError(f"Missing applied strain snapshots: {e_file}")
    u = np.load(u_file)
    e = np.load(e_file)
    if u.ndim != 2:
        raise RuntimeError(f"Expected 2D array in {u_file}, got shape={u.shape}")
    if e.ndim != 2 or e.shape[1] != 3:
        raise RuntimeError(f"Expected (Ns,3) applied strain in {e_file}, got shape={e.shape}")
    return u, e


def _normalize_u_layout(u: np.ndarray, n_dof: int) -> np.ndarray:
    if u.shape[1] == n_dof:
        return np.asarray(u, dtype=float)
    if u.shape[0] == n_dof:
        return np.asarray(u.T, dtype=float)
    raise RuntimeError(
        f"Snapshot DOF mismatch: expected one dimension = {n_dof}, got {u.shape}"
    )


def _build_runtime_geometry(mesh: str):
    params = setup_kratos_parameters(mesh)
    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, params)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()

    n_dof, eq_map, _ = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    dir_dofs, _ = ExtractDirichletBoundaryConditions(mp)

    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[dir_dofs] = False
    free_dofs = np.nonzero(free_mask)[0].astype(np.int64)

    sim._InitializeDomainCenterIfNeeded(mp)
    x0c = float(sim._x0c)
    y0c = float(sim._y0c)

    dof_x = np.zeros(n_dof, dtype=float)
    dof_y = np.zeros(n_dof, dtype=float)
    is_x_dof = np.zeros(n_dof, dtype=bool)

    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        ix = int(eq_map[i, 0])
        iy = int(eq_map[i, 1])
        if 0 <= ix < n_dof:
            dof_x[ix] = xr
            dof_y[ix] = yr
            is_x_dof[ix] = True
        if 0 <= iy < n_dof:
            dof_x[iy] = xr
            dof_y[iy] = yr
            is_x_dof[iy] = False

    return {
        "sim": sim,
        "n_dof": int(n_dof),
        "eq_map": np.asarray(eq_map, dtype=np.int64),
        "dir_dofs": np.asarray(dir_dofs, dtype=np.int64),
        "free_dofs": free_dofs,
        "x0c": x0c,
        "y0c": y0c,
        "dof_x": dof_x,
        "dof_y": dof_y,
        "is_x_dof": is_x_dof,
    }


def _affine_displacement_from_E(
    E: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    is_x: np.ndarray,
) -> np.ndarray:
    F = DeformationGradientFromGreenLagrange2D(E)
    ux = (F[0, 0] - 1.0) * x_coords + F[0, 1] * y_coords
    uy = F[1, 0] * x_coords + (F[1, 1] - 1.0) * y_coords
    return np.where(is_x, ux, uy)


def _collect_fluctuations(
    fom_dir: str,
    n_dof: int,
    free_dofs: np.ndarray,
    dir_dofs: np.ndarray,
    dof_x: np.ndarray,
    dof_y: np.ndarray,
    is_x_dof: np.ndarray,
):
    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    x_dir = dof_x[dir_dofs]
    y_dir = dof_y[dir_dofs]
    is_x_dir = is_x_dof[dir_dofs]

    traj_info = _discover_trajectory_dirs(fom_dir)

    w_blocks: List[np.ndarray] = []
    e_blocks: List[np.ndarray] = []
    traj_ids_per_sample: List[np.ndarray] = []
    traj_offsets: List[Tuple[int, int, int]] = []
    dir_res_norms: List[float] = []

    cursor = 0
    for idx, tdir in traj_info:
        U_raw, E_raw = _read_stage1_arrays(tdir, idx)
        U = _normalize_u_layout(U_raw, n_dof)

        n_steps = min(U.shape[0], E_raw.shape[0])
        if n_steps <= 0:
            print(f"[WARN] trajectory_{idx}: no valid snapshots after alignment")
            continue

        U = U[:n_steps, :]
        E_hist = np.asarray(E_raw[:n_steps, :], dtype=float)

        W = np.empty((n_steps, free_dofs.size), dtype=float)
        for k in range(n_steps):
            Ek = E_hist[k]
            u_aff_free = _affine_displacement_from_E(Ek, x_free, y_free, is_x_free)
            W[k, :] = U[k, free_dofs] - u_aff_free

            if dir_dofs.size > 0:
                u_aff_dir = _affine_displacement_from_E(Ek, x_dir, y_dir, is_x_dir)
                w_dir = U[k, dir_dofs] - u_aff_dir
                dir_res_norms.append(float(np.linalg.norm(w_dir)))

        w_blocks.append(W)
        e_blocks.append(E_hist)
        traj_ids_per_sample.append(np.full(n_steps, idx, dtype=np.int64))
        traj_offsets.append((idx, cursor, cursor + n_steps))
        cursor += n_steps

        print(f"[INFO] trajectory_{idx}: snapshots={n_steps}")

    if len(w_blocks) == 0:
        raise RuntimeError("No valid fluctuation snapshots were collected from Stage 1.")

    W_all = np.vstack(w_blocks)
    E_all = np.vstack(e_blocks)
    traj_ids = np.concatenate(traj_ids_per_sample)

    dir_stats = {
        "max": float(np.max(dir_res_norms)) if len(dir_res_norms) else 0.0,
        "mean": float(np.mean(dir_res_norms)) if len(dir_res_norms) else 0.0,
        "count": int(len(dir_res_norms)),
    }

    return W_all, E_all, traj_ids, traj_offsets, dir_stats


def _compute_pod_basis(W: np.ndarray, pod_energy_loss: float, pod_rank: int):
    D = W.T  # [n_free, Ns]
    U, S, _ = np.linalg.svd(D, full_matrices=False)
    energy = S * S
    total_energy = float(np.sum(energy))
    if total_energy <= 0.0:
        raise RuntimeError("Degenerate snapshot set: zero POD energy.")

    cumulative = np.cumsum(energy) / total_energy
    if pod_rank > 0:
        rank = int(min(max(1, pod_rank), U.shape[1]))
    else:
        target = float(1.0 - pod_energy_loss)
        target = min(max(target, 0.0), 1.0)
        rank = int(np.searchsorted(cumulative, target) + 1)
        rank = int(min(max(1, rank), U.shape[1]))

    phi = np.asarray(U[:, :rank], dtype=float)
    q = np.asarray(W @ phi, dtype=float)

    W_rec = q @ phi.T
    rel_rec = float(np.linalg.norm(W - W_rec) / max(np.linalg.norm(W), 1e-30))

    return {
        "phi": phi,
        "singular_values": S,
        "cumulative_energy": cumulative,
        "rank": rank,
        "q": q,
        "relative_reconstruction_error": rel_rec,
    }


def _save_diagnostics(out_dir: str, singular_values: np.ndarray, cumulative: np.ndarray):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib unavailable. Skipping POD diagnostics plot.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(singular_values, "o-", markersize=3)
    axs[0].set_yscale("log")
    axs[0].set_title("Singular value decay")
    axs[0].set_xlabel("mode")
    axs[0].set_ylabel("sigma")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(1.0 - cumulative, "o-", markersize=3)
    axs[1].set_yscale("log")
    axs[1].set_title("Residual energy")
    axs[1].set_xlabel("mode")
    axs[1].set_ylabel("1 - cumulative")
    axs[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(out_dir, "pod_diagnostics.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved diagnostics: {out_png}")


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 72)
    print("Stage 2a: POD from FOM")
    print("=" * 72)
    print(f"fom_dir      : {args.fom_dir}")
    print(f"out_dir      : {args.out_dir}")
    print(f"mesh         : {args.mesh}")
    print(f"pod_rank     : {args.pod_rank}")
    print(f"energy_loss  : {args.pod_energy_loss:.3e}")

    runtime = _build_runtime_geometry(args.mesh)
    sim = runtime["sim"]

    n_dof = runtime["n_dof"]
    free_dofs = runtime["free_dofs"]
    dir_dofs = runtime["dir_dofs"]

    print(
        f"[INFO] runtime DOFs: total={n_dof}, free={free_dofs.size}, dirichlet={dir_dofs.size}"
    )

    W, E_all, traj_ids, traj_offsets, dir_stats = _collect_fluctuations(
        fom_dir=args.fom_dir,
        n_dof=n_dof,
        free_dofs=free_dofs,
        dir_dofs=dir_dofs,
        dof_x=runtime["dof_x"],
        dof_y=runtime["dof_y"],
        is_x_dof=runtime["is_x_dof"],
    )

    print(f"[INFO] collected snapshots: Ns={W.shape[0]}, n_free={W.shape[1]}")
    print(
        f"[INFO] Dirichlet fluctuation check: max={dir_stats['max']:.3e}, mean={dir_stats['mean']:.3e}"
    )

    pod = _compute_pod_basis(W, args.pod_energy_loss, args.pod_rank)
    phi = pod["phi"]
    q = pod["q"]

    print(f"[INFO] POD rank: {pod['rank']} / {phi.shape[0]}")
    print(f"[INFO] POD relative reconstruction error: {pod['relative_reconstruction_error']:.3e}")

    np.save(os.path.join(args.out_dir, "pod_basis_free.npy"), phi)
    np.save(os.path.join(args.out_dir, "pod_singular_values.npy"), pod["singular_values"])
    np.save(os.path.join(args.out_dir, "pod_cumulative_energy.npy"), pod["cumulative_energy"])
    np.save(os.path.join(args.out_dir, "q_pod_train.npy"), q)
    np.save(os.path.join(args.out_dir, "applied_strain_train.npy"), E_all)

    if int(args.save_w_free) == 1:
        np.save(os.path.join(args.out_dir, "w_free_train.npy"), W)

    np.save(os.path.join(args.out_dir, "free_dofs.npy"), free_dofs)
    np.save(os.path.join(args.out_dir, "dirichlet_dofs.npy"), dir_dofs)
    np.save(os.path.join(args.out_dir, "eq_map.npy"), runtime["eq_map"])
    np.save(
        os.path.join(args.out_dir, "domain_center.npy"),
        np.array([runtime["x0c"], runtime["y0c"]], dtype=float),
    )

    traj_offsets_arr = np.asarray(traj_offsets, dtype=np.int64)
    np.save(os.path.join(args.out_dir, "snapshot_trajectory_ids.npy"), traj_ids)
    np.save(os.path.join(args.out_dir, "snapshot_trajectory_offsets.npy"), traj_offsets_arr)

    summary = {
        "fom_dir": args.fom_dir,
        "mesh": args.mesh,
        "n_total_dofs": int(n_dof),
        "n_free_dofs": int(free_dofs.size),
        "n_dirichlet_dofs": int(dir_dofs.size),
        "n_snapshots": int(W.shape[0]),
        "pod_rank": int(pod["rank"]),
        "pod_energy_loss": float(args.pod_energy_loss),
        "pod_reconstruction_error_rel": float(pod["relative_reconstruction_error"]),
        "dirichlet_fluctuation_max": float(dir_stats["max"]),
        "dirichlet_fluctuation_mean": float(dir_stats["mean"]),
        "trajectory_offsets": [
            {"trajectory": int(t), "start": int(a), "end": int(b)}
            for (t, a, b) in traj_offsets
        ],
    }

    with open(os.path.join(args.out_dir, "stage2a_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if int(args.save_plots) == 1:
        _save_diagnostics(
            out_dir=args.out_dir,
            singular_values=pod["singular_values"],
            cumulative=pod["cumulative_energy"],
        )

    sim.Finalize()
    print(f"[OK] Stage 2a data saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
