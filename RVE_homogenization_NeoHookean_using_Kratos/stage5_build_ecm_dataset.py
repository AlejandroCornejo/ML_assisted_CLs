#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5a: Build ECM dataset with independent sampling for:
  - residual projection (Q_ecm, b_full)
  - homogenization targets (C_hom, b_hom)
"""

import os
import sys
import argparse
import numpy as np

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    PrecomputeElementIntegrationWeights,
    SetInputMeshFilename,
    StripMdpaExtension,
    DetectMaterialSubModelParts,
    ConfigureElementModelerForMaterialParts,
)
from ecm_sampling_utils import get_stratified_indices, get_param_aware_indices

# ============================================================
# CONFIGURATION
# ============================================================
SNAPSHOTS_DIR = "stage_1_training_set_fom"
MODEL_DIR = "stage_2_pod_rve"
OUT_DIR = "stage_5_ecm_dataset"
SNAPSHOT_PERCENT_RES = 2.0
SNAPSHOT_PERCENT_HOM = 2.0
SEED = 42
SAMPLING_MODE = "param_aware"  # options: "param_aware", "stratified"
PARAM_AWARE_TIME_WEIGHT = 0.20

# ============================================================
# UTILITIES
# ============================================================

def BuildFreeMap(n_dof, free_dofs):
    map_g2f = -np.ones(n_dof, dtype=int)
    for i, gdof in enumerate(free_dofs):
        map_g2f[int(gdof)] = i
    return map_g2f


def _to_voigt3_array(values):
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


def BuildResidualProjectionCache(elements, process_info, map_g2f, phi_f):
    """
    Precompute element-local projection data for q_e = V_e^T r_e.
    Each entry is either None (no free DOFs contribute) or:
      (rhs_pick_positions, V_e_T)
    """
    cache = []
    active = 0
    for elem in elements:
        ids = np.array(elem.EquationIdVector(process_info), dtype=int)
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
        V_e_T = np.ascontiguousarray(phi_f[rows[valid], :].T)
        cache.append((rhs_pick, V_e_T))
        active += 1
    return cache, active


def FillHomogenizationBlockFromKratos(elements, process_info, area_e, c_block):
    """
    Build per-element homogenization columns:
      c_block[0:3, e] = A_e * mean_gp(eps_e)
      c_block[3:6, e] = A_e * mean_gp(sig_e)
    """
    for e, elem in enumerate(elements):
        eps = _to_voigt3_array(
            elem.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
        )
        sig = _to_voigt3_array(
            elem.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info)
        )
        A_e = float(area_e[e])
        c_block[0:3, e] = A_e * np.mean(eps, axis=0)
        c_block[3:6, e] = A_e * np.mean(sig, axis=0)

def AllocateMemmaps(out_dir, nq, N_s_res, N_s_hom, n_elem):
    os.makedirs(out_dir, exist_ok=True)
    Q_path = os.path.join(out_dir, "Q_ecm.dat")
    b_path = os.path.join(out_dir, "b_full.dat")
    C_path = os.path.join(out_dir, "C_hom.dat")
    bh_path = os.path.join(out_dir, "b_hom.dat")

    # Wipe previous
    for p in (Q_path, b_path, C_path, bh_path):
        if os.path.exists(p): os.remove(p)

    Q_ecm = np.memmap(Q_path, dtype="float64", mode="w+", shape=(nq * N_s_res, n_elem))
    b_full = np.memmap(b_path, dtype="float64", mode="w+", shape=(nq * N_s_res,))
    C_hom = np.memmap(C_path, dtype="float64", mode="w+", shape=(6 * N_s_hom, n_elem))
    b_hom = np.memmap(bh_path, dtype="float64", mode="w+", shape=(6 * N_s_hom,))

    return Q_ecm, b_full, C_hom, b_hom


def ParseArgs():
    p = argparse.ArgumentParser(
        description="Stage 5a: build ECM dataset with separate residual/hom sampling."
    )
    p.add_argument(
        "--snapshots-dir",
        type=str,
        default=SNAPSHOTS_DIR,
        help="Input Stage-1 snapshot root directory.",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help="Input ROM model directory (Stage-2).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=OUT_DIR,
        help="Output directory for ECM dataset files.",
    )
    p.add_argument(
        "--snapshot-percent-res",
        type=float,
        default=SNAPSHOT_PERCENT_RES,
        help="Percent of snapshots used for residual dataset Q_ecm/b_full.",
    )
    p.add_argument(
        "--snapshot-percent-hom",
        type=float,
        default=SNAPSHOT_PERCENT_HOM,
        help="Percent of snapshots used for homogenization dataset C_hom/b_hom.",
    )
    p.add_argument(
        "--sampling-mode",
        type=str,
        default=SAMPLING_MODE,
        choices=["param_aware", "stratified"],
        help="Snapshot sampling mode.",
    )
    p.add_argument("--seed", type=int, default=SEED, help="Random seed for sampling.")
    p.add_argument(
        "--param-aware-time-weight",
        type=float,
        default=PARAM_AWARE_TIME_WEIGHT,
        help="Time weight for param-aware sampling.",
    )
    p.add_argument(
        "--trajectory-indices",
        type=str,
        default="",
        help=(
            "Optional comma-separated trajectory indices to include "
            "(e.g. '2' or '1,2,5'). Empty means all trajectories."
        ),
    )
    p.add_argument(
        "--first-n-steps",
        type=int,
        default=0,
        help="If >0, only first N snapshots per selected trajectory are considered.",
    )
    return p.parse_args()


def PickSnapshotIndices(E_traj, n_steps, n_pick, mode, seed, time_weight):
    n_pick = int(max(1, min(int(n_steps), int(n_pick))))
    if str(mode).strip().lower() == "param_aware":
        return get_param_aware_indices(
            E_traj[:n_steps, :3],
            n_pick,
            seed=int(seed),
            time_weight=float(time_weight),
        )
    return get_stratified_indices(int(n_steps), n_pick, seed=int(seed))

# ============================================================
# MAIN
# ============================================================

def main():
    args = ParseArgs()
    print("--- Stage 5a: Building ECM Dataset ---")
    pct_res = float(args.snapshot_percent_res)
    pct_hom = float(args.snapshot_percent_hom)
    if pct_res <= 0.0 or pct_hom <= 0.0:
        raise ValueError("snapshot percentages must be > 0.")
    if pct_res > 100.0 or pct_hom > 100.0:
        raise ValueError("snapshot percentages must be <= 100.")
    
    snapshots_dir = str(args.snapshots_dir)
    model_dir = str(args.model_dir)
    out_dir = str(args.out_dir)
    first_n_steps = int(args.first_n_steps)
    if first_n_steps < 0:
        raise ValueError("--first-n-steps must be >= 0.")

    selected_traj_ids = None
    if str(args.trajectory_indices).strip():
        try:
            selected_traj_ids = sorted(
                set(int(v.strip()) for v in str(args.trajectory_indices).split(",") if v.strip())
            )
        except ValueError as exc:
            raise ValueError("--trajectory-indices must be comma-separated integers.") from exc
        if any(v <= 0 for v in selected_traj_ids):
            raise ValueError("--trajectory-indices must be >= 1.")

    # 1. Load PROM Basis
    phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy")).astype(int)
    nq = phi_f.shape[1]

    # 2. Identify Snapshots
    trajectories = sorted([d for d in os.listdir(snapshots_dir) if d.startswith("trajectory_")])
    if selected_traj_ids is not None:
        selected_names = {f"trajectory_{i}" for i in selected_traj_ids}
        trajectories = [t for t in trajectories if t in selected_names]
    if not trajectories: 
        raise FileNotFoundError(f"No trajectory folders found in {snapshots_dir} for selection={selected_traj_ids}.")

    # Count total valid snapshots for each dataset
    total_snapshots_res = 0
    total_snapshots_hom = 0
    all_tasks = []
    frac_res = pct_res / 100.0
    frac_hom = pct_hom / 100.0

    for traj in trajectories:
        u_file = os.path.join(snapshots_dir, traj, f"{traj}_U.npy")
        e_file = os.path.join(snapshots_dir, traj, f"{traj}_applied_strain.npy")
        if not (os.path.exists(u_file) and os.path.exists(e_file)):
            print(f"  [Skip] {traj}: missing U or applied_strain file")
            continue
        
        # Load one to get count (not efficient for disk but safer)
        U_meta = np.load(u_file, mmap_mode='r')
        E_meta = np.load(e_file, mmap_mode='r')
        if E_meta.ndim != 2 or E_meta.shape[1] < 3:
            print(f"  [Skip] {traj}: invalid applied_strain shape {E_meta.shape}")
            continue
        n_traj_snaps = min(int(U_meta.shape[0]), int(E_meta.shape[0]))
        if first_n_steps > 0:
            n_traj_snaps = min(n_traj_snaps, first_n_steps)
        if n_traj_snaps <= 0:
            print(f"  [Skip] {traj}: no snapshots after first-n-steps filter")
            continue
        
        n_pick_res = int(np.ceil(frac_res * n_traj_snaps))
        n_pick_hom = int(np.ceil(frac_hom * n_traj_snaps))
        idx_res = PickSnapshotIndices(
            E_meta,
            n_steps=n_traj_snaps,
            n_pick=n_pick_res,
            mode=args.sampling_mode,
            seed=int(args.seed) + 2 * len(all_tasks),
            time_weight=args.param_aware_time_weight,
        )
        idx_hom = PickSnapshotIndices(
            E_meta,
            n_steps=n_traj_snaps,
            n_pick=n_pick_hom,
            mode=args.sampling_mode,
            seed=int(args.seed) + 2 * len(all_tasks) + 1,
            time_weight=args.param_aware_time_weight,
        )

        total_snapshots_res += len(idx_res)
        total_snapshots_hom += len(idx_hom)
        all_tasks.append((traj, u_file, idx_res, idx_hom))

    if total_snapshots_res <= 0:
        raise RuntimeError("No residual snapshots selected.")
    if total_snapshots_hom <= 0:
        raise RuntimeError("No homogenization snapshots selected.")

    print(f"[Info] Sampling mode: {args.sampling_mode}")
    print(
        f"[Info] Target residual snapshots: {pct_res}% across {len(all_tasks)} trajectories."
    )
    print(
        f"[Info] Target homogenization snapshots: {pct_hom}% across {len(all_tasks)} trajectories."
    )
    if selected_traj_ids is not None:
        print(f"[Info] Selected trajectory indices: {selected_traj_ids}")
    if first_n_steps > 0:
        print(f"[Info] First-N filter per trajectory: {first_n_steps}")
    print(f"[Info] Total residual snapshots      : {total_snapshots_res}")
    print(f"[Info] Total homogenization snapshots: {total_snapshots_hom}")

    # 3. Setup Kratos (same init as FOM driver)
    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())
    SetInputMeshFilename(parameters, "rve_geometry")

    mesh_base = parameters["modelers"][0]["parameters"]["input_filename"].GetString()
    mdpa_path = f"{StripMdpaExtension(mesh_base)}.mdpa"

    if os.path.exists(mdpa_path):
        material_parts = DetectMaterialSubModelParts(mdpa_path)
        parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
        parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(
            "StructuralMaterials.json"
        )
        print(f"[Info] Material parts: {material_parts}")

    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    pi = mp.ProcessInfo

    n_dof, eq_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    elements = list(mp.Elements)
    n_elem = len(elements)
    _, area_e = PrecomputeElementIntegrationWeights(elements)
    map_g2f = BuildFreeMap(n_dof, free_dofs)

    # Precompute local projection operators for residual assembly
    residual_cache, n_active_elems = BuildResidualProjectionCache(elements, pi, map_g2f, phi_f)
    rhs_vectors = [KM.Vector() for _ in range(n_elem)]
    q_block = np.zeros((nq, n_elem), dtype=float)
    c_block = np.zeros((6, n_elem), dtype=float)
    print(f"[Info] Residual projection active on {n_active_elems}/{n_elem} elements.")

    # 4. Allocate memmaps
    Q_ecm, b_full, C_hom, b_hom = AllocateMemmaps(
        out_dir, nq, total_snapshots_res, total_snapshots_hom, n_elem
    )

    s_res_global = 0
    s_hom_global = 0
    for traj_name, u_path, idx_res, idx_hom in all_tasks:
        idx_res = np.asarray(idx_res, dtype=int).reshape(-1)
        idx_hom = np.asarray(idx_hom, dtype=int).reshape(-1)
        idx_res = np.unique(idx_res)
        idx_hom = np.unique(idx_hom)
        idx_union = np.union1d(idx_res, idx_hom)
        set_res = set(int(v) for v in idx_res.tolist())
        set_hom = set(int(v) for v in idx_hom.tolist())

        print(
            f"  > Processing {traj_name} "
            f"(res={len(idx_res)} steps, hom={len(idx_hom)} steps)..."
        )
        U_all = np.load(u_path, mmap_mode="r")

        for k in idx_union:
            ks = int(k)
            u_snap = np.asarray(U_all[ks, :], dtype=float)

            # Apply displacement snapshot to Kratos model once.
            SetDisplacementFromEquationVector(u_snap, eq_map, ta)
            UpdateCurrentCoordinatesFromDisplacement(mp)

            if ks in set_res:
                q_block.fill(0.0)
                for i, elem in enumerate(elements):
                    cached = residual_cache[i]
                    if cached is None:
                        continue
                    rhs_pick, V_e_T = cached
                    RHS = rhs_vectors[i]
                    elem.CalculateRightHandSide(RHS, pi)
                    rhs_arr = np.asarray(RHS, dtype=float)
                    q_block[:, i] = V_e_T @ rhs_arr[rhs_pick]

                r0, r1 = nq * s_res_global, nq * (s_res_global + 1)
                Q_ecm[r0:r1, :] = q_block
                b_full[r0:r1] = np.sum(q_block, axis=1)
                s_res_global += 1

            if ks in set_hom:
                # Kratos-reference homogenization operator per element:
                #   C_elem = A_e * mean_gp(value_gp)
                # so that global homogenization is:
                #   value_hom = (sum_e C_elem) / (sum_e A_e)
                # This keeps Stage 5 ECM targets consistent with Stage 11/FOM reference.
                FillHomogenizationBlockFromKratos(elements, pi, area_e, c_block)

                h0, h1 = 6 * s_hom_global, 6 * (s_hom_global + 1)
                C_hom[h0:h1, :] = c_block
                b_hom[h0:h1] = np.sum(c_block, axis=1)
                s_hom_global += 1

    if s_res_global != total_snapshots_res:
        raise RuntimeError(
            f"Residual snapshot count mismatch: expected {total_snapshots_res}, got {s_res_global}"
        )
    if s_hom_global != total_snapshots_hom:
        raise RuntimeError(
            f"Homogenization snapshot count mismatch: expected {total_snapshots_hom}, got {s_hom_global}"
        )

    # Finalize
    Q_ecm.flush()
    b_full.flush()
    C_hom.flush()
    b_hom.flush()

    a0_ref = float(np.sum(area_e))
    np.savez(
        os.path.join(out_dir, "meta.npz"),
        nq=nq,
        n_elem=n_elem,
        N_s_res=total_snapshots_res,
        N_s_hom=total_snapshots_hom,
        snapshot_percent_res=pct_res,
        snapshot_percent_hom=pct_hom,
        sampling_mode=np.array([args.sampling_mode]),
        param_aware_time_weight=np.array([args.param_aware_time_weight]),
        snapshots_dir=np.array([snapshots_dir]),
        model_dir=np.array([model_dir]),
        first_n_steps=np.array([first_n_steps], dtype=np.int64),
        trajectory_indices=np.array(selected_traj_ids if selected_traj_ids is not None else [], dtype=np.int64),
        A_total=a0_ref,
        A0_ref=np.array([a0_ref], dtype=float),
        hom_reference_measure=np.array([a0_ref], dtype=float),
    )
    with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{a0_ref:.16e}\n")
    print("\n[DONE] Dataset generation complete.")
    print(f"      - Q_ecm shape: {Q_ecm.shape}")
    print(f"      - C_hom shape: {C_hom.shape}")
    print(f"      - Datasets saved to: {out_dir}")

if __name__ == "__main__":
    main()
