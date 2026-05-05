#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5a: Build ECM Residual Dataset
Replays FOM snapshots to collect element-wise internal forces and projects them onto the POD basis.
"""

import os
import sys
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
    ExtractDirichletBoundaryConditions,
    PrecomputeElementIntegrationWeights,
    EvaluateGaussPointData,
    SetInputMeshFilename,
    StripMdpaExtension,
    DetectMaterialSubModelParts,
    ConfigureElementModelerForMaterialParts,
)

# ============================================================
# CONFIGURATION
# ============================================================
SNAPSHOTS_DIR = "stage_1_training_set_fom"
MODEL_DIR = "stage_2_pod_rve"
OUT_DIR = "stage_5_ecm_dataset"
SNAPSHOT_PERCENT = 2.0 # Target 10% of total snapshots
SEED = 42

# ============================================================
# UTILITIES
# ============================================================

def GetStratifiedIndices(n_total, n_pick, seed=SEED):
    """Pick n_pick indices from n_total with random-but-spread coverage."""
    if n_pick <= 0: return np.array([], dtype=int)
    if n_pick >= n_total: return np.arange(n_total)
    
    rng = np.random.default_rng(seed)
    picks = np.zeros(n_pick, dtype=int)
    edges = np.linspace(0, n_total, n_pick + 1, dtype=int)
    for i in range(n_pick):
        i0, i1 = edges[i], edges[i+1]
        if i1 <= i0: picks[i] = i0
        else: picks[i] = rng.integers(i0, i1)
    return np.sort(np.unique(picks))

def BuildFreeMap(n_dof, free_dofs):
    map_g2f = -np.ones(n_dof, dtype=int)
    for i, gdof in enumerate(free_dofs):
        map_g2f[int(gdof)] = i
    return map_g2f

def AllocateMemmaps(out_dir, nq, N_s, n_elem):
    os.makedirs(out_dir, exist_ok=True)
    Q_path = os.path.join(out_dir, "Q_ecm.dat")
    b_path = os.path.join(out_dir, "b_full.dat")
    C_path = os.path.join(out_dir, "C_hom.dat")
    bh_path = os.path.join(out_dir, "b_hom.dat")

    # Wipe previous
    for p in (Q_path, b_path, C_path, bh_path):
        if os.path.exists(p): os.remove(p)

    Q_ecm = np.memmap(Q_path, dtype="float64", mode="w+", shape=(nq * N_s, n_elem))
    b_full = np.memmap(b_path, dtype="float64", mode="w+", shape=(nq * N_s,))
    C_hom = np.memmap(C_path, dtype="float64", mode="w+", shape=(6 * N_s, n_elem))
    b_hom = np.memmap(bh_path, dtype="float64", mode="w+", shape=(6 * N_s,))

    return Q_ecm, b_full, C_hom, b_hom

# ============================================================
# MAIN
# ============================================================

def main():
    print("--- Stage 5a: Building ECM Dataset ---")
    
    # 1. Load PROM Basis
    phi_f = np.load(os.path.join(MODEL_DIR, "pod_basis_free.npy"))
    phi_d = np.load(os.path.join(MODEL_DIR, "phi_affine_lifting.npy"))
    free_dofs = np.load(os.path.join(MODEL_DIR, "free_dofs.npy")).astype(int)
    dir_dofs_pod = np.load(os.path.join(MODEL_DIR, "dirichlet_dofs.npy")).astype(int)
    nq = phi_f.shape[1]

    # 2. Identify Snapshots
    trajectories = sorted([d for d in os.listdir(SNAPSHOTS_DIR) if d.startswith("trajectory_")])
    if not trajectories: 
        raise FileNotFoundError(f"No trajectory folders found in {SNAPSHOTS_DIR}")

    # Count total valid snapshots with stratified sampling
    total_snapshots = 0
    all_tasks = []
    pct = SNAPSHOT_PERCENT / 100.0

    for traj in trajectories:
        u_file = os.path.join(SNAPSHOTS_DIR, traj, f"{traj}_U.npy")
        e_file = os.path.join(SNAPSHOTS_DIR, traj, f"{traj}_applied_strain.npy")
        if not os.path.exists(u_file):
            print(f"  [Skip] {traj}: missing U file")
            continue
        
        # Load one to get count (not efficient for disk but safer)
        U_meta = np.load(u_file, mmap_mode='r')
        n_traj_snaps = U_meta.shape[0]
        
        n_pick = int(np.ceil(pct * n_traj_snaps))
        idx = GetStratifiedIndices(n_traj_snaps, n_pick)
        
        total_snapshots += len(idx)
        all_tasks.append((traj, u_file, e_file, idx))

    print(f"[Info] Target: {SNAPSHOT_PERCENT}% snapshots stratified across {len(all_tasks)} trajectories.")
    print(f"[Info] Total snapshots to process: {total_snapshots}")

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
    w_gp, area_e = PrecomputeElementIntegrationWeights(elements)
    map_g2f = BuildFreeMap(n_dof, free_dofs)

    # Pre-extract Element Dof Maps for speed
    elem_dofs = []
    elem_masks = []
    for elem in elements:
        ids = np.array(elem.EquationIdVector(pi), dtype=int)
        mask = ids >= 0
        elem_dofs.append(ids[mask])
        elem_masks.append(mask)

    # 4. Allocate memmaps
    Q_ecm, b_full, C_hom, b_hom = AllocateMemmaps(OUT_DIR, nq, total_snapshots, n_elem)

    s_global = 0
    for traj_name, u_path, e_path, snapshot_indices in all_tasks:
        print(f"  > Processing {traj_name} ({len(snapshot_indices)} steps)...")
        U_all = np.load(u_path)
        E_all = np.load(e_path)

        for k in snapshot_indices:
            u_snap = U_all[k, :].astype(float)
            e_macro = E_all[k, :].astype(float)

            # Apply displacement snapshot to Kratos model
            SetDisplacementFromEquationVector(u_snap, eq_map, ta)
            UpdateCurrentCoordinatesFromDisplacement(mp)

            # Element-wise residuals
            q_block = np.zeros((nq, n_elem))
            c_block = np.zeros((6, n_elem))

            for i, elem in enumerate(elements):
                RHS = KM.Vector()
                elem.CalculateRightHandSide(RHS, pi)
                rhs_arr = np.array(RHS, dtype=float)

                local_mask = elem_masks[i]
                local_dofs = elem_dofs[i]

                # Project RHS onto POD
                # find indices of local_dofs in free_dofs
                rows = map_g2f[local_dofs]
                valid = rows >= 0
                if np.any(valid):
                    V_e = phi_f[rows[valid], :]
                    r_e = rhs_arr[local_mask][valid]
                    q_block[:, i] = V_e.T @ r_e

            # Homogenized components (for weighted homogenization)
            eps_gp, sig_gp, _ = EvaluateGaussPointData(elements, mp)
            # eps_gp: (n_elem, n_gauss, 3)
            # Reduction for Homogenization dataset: 
            # C_elem = Integral( [eps, sig] ) = Sum_g( w_g * [eps_g, sig_g] )
            # We want Sum_e ( we * C_elem ) = Total Macro Stress/Strain
            # Here C_hom stores the per-element contribution.
            
            # Kratos-reference homogenization operator per element:
            #   C_elem = A_e * mean_gp(value_gp)
            # so that global homogenization is:
            #   value_hom = (sum_e C_elem) / (sum_e A_e)
            # This keeps Stage 5 ECM targets consistent with Stage 11/FOM reference.
            c_block[0:3, :] = (area_e[:, None] * np.mean(eps_gp, axis=1)).T
            c_block[3:6, :] = (area_e[:, None] * np.mean(sig_gp, axis=1)).T

            # Save to memmaps
            r0, r1 = nq * s_global, nq * (s_global + 1)
            Q_ecm[r0:r1, :] = q_block
            b_full[r0:r1] = np.sum(q_block, axis=1)

            h0, h1 = 6 * s_global, 6 * (s_global + 1)
            C_hom[h0:h1, :] = c_block
            b_hom[h0:h1] = np.sum(c_block, axis=1)

            s_global += 1

    # Finalize
    Q_ecm.flush()
    b_full.flush()
    C_hom.flush()
    b_hom.flush()

    np.savez(
        os.path.join(OUT_DIR, "meta.npz"),
        nq=nq, N_s=total_snapshots, n_elem=n_elem,
        snapshot_percent=SNAPSHOT_PERCENT,
        A_total=np.sum(area_e)
    )
    print("\n[DONE] Dataset generation complete.")
    print(f"      - Q_ecm shape: {Q_ecm.shape}")
    print(f"      - Datasets saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
