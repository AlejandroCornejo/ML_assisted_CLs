#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 8a (2D-MAWECM): build structured MAW-ECM dataset.

This dataset is designed for MAW training on structured latent/master nodes
from Stage3 grid mapping. Each node contributes:
- one local residual block A_res,j and target b_res,j
- optionally one local homogenization block A_hom,j and target b_hom,j
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    setup_kratos_parameters,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage8a: structured residual-only MAW dataset")
    p.add_argument("--fom-dir", type=str, default="stage_1_training_set_fom")
    p.add_argument("--pod-dir", type=str, default="stage_2a_pod_data")
    p.add_argument(
        "--stage3-dataset-file",
        type=str,
        default="stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    )
    p.add_argument("--stage0-file", type=str, default="stage_0_trajectory/stage_0_trajectories.npz")
    p.add_argument("--mesh", type=str, default="rve_geometry")
    p.add_argument("--out-dir", type=str, default="stage_8a_mawecm_res_dataset")
    p.add_argument(
        "--include-homogenization",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, also stores structured homogenization blocks C_hom/b_hom for MAW-single experiments.",
    )
    return p.parse_args()


def _build_free_map(n_dof: int, free_dofs: np.ndarray) -> np.ndarray:
    g2f = -np.ones(int(n_dof), dtype=int)
    for i, gdof in enumerate(np.asarray(free_dofs, dtype=np.int64).reshape(-1)):
        g2f[int(gdof)] = int(i)
    return g2f


def _to_voigt3(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    ncomp = int(arr.shape[1]) if arr.ndim >= 2 else 0
    if ncomp == 3:
        return arr
    if ncomp >= 4:
        return arr[:, [0, 1, 3]]
    out = np.zeros((arr.shape[0], 3), dtype=float)
    if ncomp > 0:
        out[:, : min(3, ncomp)] = arr[:, : min(3, ncomp)]
    return out


def _build_residual_projection_cache(elements, process_info, map_g2f: np.ndarray, phi_f: np.ndarray):
    cache = []
    active = 0
    for elem in elements:
        ids = np.asarray(elem.EquationIdVector(process_info), dtype=int)
        local_pos = np.flatnonzero(ids >= 0)
        if local_pos.size == 0:
            cache.append(None)
            continue

        local_dofs = ids[local_pos]
        rows = map_g2f[local_dofs]
        valid = rows >= 0
        if not np.any(valid):
            cache.append(None)
            continue

        rhs_pick = local_pos[valid].astype(int, copy=False)
        v_e_t = np.ascontiguousarray(phi_f[rows[valid], :].T)
        cache.append((rhs_pick, v_e_t))
        active += 1
    return cache, active


def _traj_offsets_to_lookup(offsets: np.ndarray):
    lookup = []
    for row in np.asarray(offsets):
        traj_id = int(row[0])
        i0 = int(row[1])
        i1 = int(row[2])
        lookup.append((traj_id, i0, i1))
    lookup.sort(key=lambda t: t[1])
    return lookup


def _global_to_traj_local(g_idx: int, offsets_lookup):
    gi = int(g_idx)
    for traj_id, i0, i1 in offsets_lookup:
        if i0 <= gi < i1:
            return traj_id, gi - i0
    raise RuntimeError(f"Global snapshot index {gi} not found in trajectory offsets.")


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for p in (
        args.stage3_dataset_file,
        args.stage0_file,
        os.path.join(args.pod_dir, "pod_basis_free.npy"),
        os.path.join(args.pod_dir, "free_dofs.npy"),
        os.path.join(args.pod_dir, "snapshot_trajectory_offsets.npy"),
    ):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    s3 = np.load(args.stage3_dataset_file, allow_pickle=True)
    s0 = np.load(args.stage0_file, allow_pickle=True)

    if (
        "grid_node_snapshot_idx" not in s3
        or "grid_node_q_m" not in s3
        or "grid_node_mu" not in s3
    ):
        raise RuntimeError(
            "Stage3 dataset missing grid-node keys. Required: "
            "grid_node_snapshot_idx, grid_node_q_m, grid_node_mu."
        )

    snap_idx_nodes = np.asarray(s3["grid_node_snapshot_idx"], dtype=np.int64).reshape(-1)
    q_m_nodes = np.asarray(s3["grid_node_q_m"], dtype=float)
    mu_nodes = np.asarray(s3["grid_node_mu"], dtype=float)
    grid_nodes_param = np.asarray(s3["grid_nodes_param"], dtype=float) if "grid_nodes_param" in s3 else None

    n_nodes = int(snap_idx_nodes.size)
    if q_m_nodes.shape[0] != n_nodes or mu_nodes.shape[0] != n_nodes:
        raise RuntimeError("grid node arrays length mismatch in Stage3 dataset.")

    phi_f = np.asarray(np.load(os.path.join(args.pod_dir, "pod_basis_free.npy")), dtype=float)
    free_dofs = np.asarray(np.load(os.path.join(args.pod_dir, "free_dofs.npy")), dtype=np.int64)
    offsets = np.asarray(np.load(os.path.join(args.pod_dir, "snapshot_trajectory_offsets.npy")), dtype=np.int64)
    offsets_lookup = _traj_offsets_to_lookup(offsets)

    n_qm = int(q_m_nodes.shape[1])
    nq = int(phi_f.shape[1])

    params = setup_kratos_parameters(args.mesh)
    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, params)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    pi = mp.ProcessInfo
    n_dof, eq_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    elements = list(mp.Elements)
    n_elem = int(len(elements))
    if n_elem <= 0:
        raise RuntimeError("Mesh has no elements.")

    map_g2f = _build_free_map(n_dof, free_dofs)
    residual_cache, n_active = _build_residual_projection_cache(elements, pi, map_g2f, phi_f)
    rhs_vectors = [KM.Vector() for _ in range(n_elem)]
    q_block = np.zeros((nq, n_elem), dtype=float)

    q_path = os.path.join(args.out_dir, "Q_ecm.dat")
    b_path = os.path.join(args.out_dir, "b_full.dat")
    c_path = os.path.join(args.out_dir, "C_hom.dat")
    bh_path = os.path.join(args.out_dir, "b_hom.dat")
    for p in (q_path, b_path, c_path, bh_path):
        if os.path.exists(p):
            os.remove(p)
    q_ecm = np.memmap(q_path, dtype=np.float64, mode="w+", shape=(nq * n_nodes, n_elem))
    b_full = np.memmap(b_path, dtype=np.float64, mode="w+", shape=(nq * n_nodes,))
    include_hom = bool(int(args.include_homogenization))
    if include_hom:
        c_hom = np.memmap(c_path, dtype=np.float64, mode="w+", shape=(6 * n_nodes, n_elem))
        b_hom = np.memmap(bh_path, dtype=np.float64, mode="w+", shape=(6 * n_nodes,))
    else:
        c_hom = None
        b_hom = None

    sample_ids = np.zeros((n_nodes, 2), dtype=np.int64)
    c_block = np.zeros((6, n_elem), dtype=float)
    area_e = np.zeros(n_elem, dtype=float)
    for e, elem in enumerate(elements):
        area_e[e] = float(elem.GetGeometry().Area())

    # Cache trajectory arrays lazily
    u_cache: Dict[int, np.ndarray] = {}

    print("=" * 78)
    print("Stage 8a: MAW-ECM residual structured dataset")
    print("=" * 78)
    print(f"nodes / nq / n_elem : {n_nodes} / {nq} / {n_elem}")
    print(f"q_m_dim             : {n_qm}")
    print(f"active residual proj: {n_active}/{n_elem}")
    print(f"include hom blocks  : {int(include_hom)}")

    for j in range(n_nodes):
        gidx = int(snap_idx_nodes[j])
        traj_id, local_idx = _global_to_traj_local(gidx, offsets_lookup)
        sample_ids[j, 0] = traj_id
        sample_ids[j, 1] = local_idx

        if traj_id not in u_cache:
            u_file = os.path.join(args.fom_dir, f"trajectory_{traj_id}", f"trajectory_{traj_id}_U.npy")
            if not os.path.exists(u_file):
                raise FileNotFoundError(u_file)
            u_cache[traj_id] = np.load(u_file, mmap_mode="r")

        u_all = u_cache[traj_id]
        if not (0 <= local_idx < int(u_all.shape[0])):
            raise RuntimeError(
                f"Local index {local_idx} out of range for trajectory_{traj_id} with {u_all.shape[0]} snapshots."
            )

        u_snap = np.asarray(u_all[local_idx, :], dtype=float)
        SetDisplacementFromEquationVector(u_snap, eq_map, ta)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)

        q_block.fill(0.0)
        for e, elem in enumerate(elements):
            cached = residual_cache[e]
            if cached is None:
                continue
            rhs_pick, v_e_t = cached
            rhs = rhs_vectors[e]
            elem.CalculateRightHandSide(rhs, pi)
            rhs_arr = np.asarray(rhs, dtype=float)
            q_block[:, e] = v_e_t @ rhs_arr[rhs_pick]

        r0, r1 = nq * j, nq * (j + 1)
        q_ecm[r0:r1, :] = q_block
        b_full[r0:r1] = np.sum(q_block, axis=1)
        if include_hom:
            for e, elem in enumerate(elements):
                eps = _to_voigt3(elem.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, pi))
                sig = _to_voigt3(elem.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, pi))
                a_e = float(area_e[e])
                c_block[0:3, e] = a_e * np.mean(eps, axis=0)
                c_block[3:6, e] = a_e * np.mean(sig, axis=0)
            h0, h1 = 6 * j, 6 * (j + 1)
            c_hom[h0:h1, :] = c_block
            b_hom[h0:h1] = np.sum(c_block, axis=1)

        if (j + 1) % 50 == 0 or (j + 1) == n_nodes:
            print(f"  processed nodes: {j + 1}/{n_nodes}")

    q_ecm.flush()
    b_full.flush()
    if include_hom:
        c_hom.flush()
        b_hom.flush()

    # Structured graph metadata from Stage0
    if "grid_cells_quad" not in s0:
        raise RuntimeError("Stage0 file missing grid_cells_quad required for structured MAW graph.")
    cells = np.asarray(s0["grid_cells_quad"], dtype=np.int64)
    grid_shape = np.asarray(s0["structured_mesh_shape"], dtype=np.int64) if "structured_mesh_shape" in s0 else None

    # reference measure from undeformed mesh (sum of element areas)
    a0_ref = float(np.sum(area_e))

    np.save(os.path.join(args.out_dir, "q_m_res.npy"), np.asarray(q_m_nodes, dtype=float))
    np.save(os.path.join(args.out_dir, "mu_res.npy"), np.asarray(mu_nodes, dtype=float))
    np.save(os.path.join(args.out_dir, "sample_ids_res.npy"), sample_ids)
    if grid_nodes_param is not None:
        np.save(os.path.join(args.out_dir, "grid_nodes_param.npy"), grid_nodes_param)
    np.save(os.path.join(args.out_dir, "structured_mesh_cells_res.npy"), cells)
    if grid_shape is not None:
        np.save(os.path.join(args.out_dir, "structured_mesh_grid_shape_res.npy"), grid_shape)

    np.savez(
        os.path.join(args.out_dir, "meta.npz"),
        nq=np.array([nq], dtype=np.int64),
        n_elem=np.array([n_elem], dtype=np.int64),
        N_s_res=np.array([n_nodes], dtype=np.int64),
        N_s_hom=np.array([n_nodes if include_hom else 0], dtype=np.int64),
        stage3_dataset_file=np.array([str(args.stage3_dataset_file)]),
        stage0_file=np.array([str(args.stage0_file)]),
        pod_dir=np.array([str(args.pod_dir)]),
        fom_dir=np.array([str(args.fom_dir)]),
        include_homogenization=np.array([int(include_hom)], dtype=np.int64),
        A_total=np.array([a0_ref], dtype=float),
        A0_ref=np.array([a0_ref], dtype=float),
        hom_reference_measure=np.array([a0_ref], dtype=float),
    )

    print("[OK] Stage8a dataset built")
    print(f"  Q_ecm shape: ({nq * n_nodes}, {n_elem})")
    print(f"  b_full shape: ({nq * n_nodes},)")
    if include_hom:
        print(f"  C_hom shape: ({6 * n_nodes}, {n_elem})")
        print(f"  b_hom shape: ({6 * n_nodes},)")
    print(f"  A0_ref: {a0_ref:.6e}")


if __name__ == "__main__":
    main()
