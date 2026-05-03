#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 — POD basis for free fluctuations (Strong Dirichlet Finite-Deformation)

This script:
1. Reconstructs the FOM DOF structure (Mesh, Equation IDs, Dirichlet/Free split).
2. Loads total displacement snapshots (U) and applied strains (E).
3. Subtracts the exact non-linear affine lifting: u_aff = (F - I)(X - Xc).
4. Verifies that the residual on Dirichlet DOFs is near zero.
5. Computes the POD basis (Phi_f) on the free fluctuation snapshots.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import save_npz

# Add Kratos path if not in PYTHONPATH (User specific, but following project style)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    ExtractDirichletBoundaryConditions,
    StripMdpaExtension,
    DetectMaterialSubModelParts,
    ConfigureElementModelerForMaterialParts,
    RVEHomogenizationDatasetGenerator,
)

# ============================================================
# SETTINGS
# ============================================================
SNAPSHOTS_ROOT = "stage_1_training_set_fom" # Root folder containing trajectory_n subfolders
PROJECT_PARAMETERS = "ProjectParameters.json"
OUT_DIR = "stage_2_pod_rve"

U_PREFIX = "trajectory_"
STRAIN_PREFIX = "trajectory_"
APPLIED_STRAIN_SUFFIX = "_applied_strain.npy"
U_SUFFIX = "_U.npy"

# POD truncation
EPS2_TOL = 1e-8 # Squared energy loss tolerance (99.9999% captured energy)

SAVE_DIAGNOSTICS = True

# ============================================================
# Helpers
# ============================================================

def get_affine_lifting(mp, E, Xc, Yc, eq_map, n_dof):
    """Computes the exact (F-I)(X-Xc) displacement vector for all DOFs."""
    F = DeformationGradientFromGreenLagrange2D(E)
    u_aff = np.zeros(n_dof, dtype=float)
    
    for i, node in enumerate(mp.Nodes):
        xr = node.X0 - Xc
        yr = node.Y0 - Yc
        
        # u = (F-I)X
        ux = (F[0, 0] - 1.0) * xr + F[0, 1] * yr
        uy = F[1, 0] * xr + (F[1, 1] - 1.0) * yr
        
        idx_x = eq_map[i, 0]
        idx_y = eq_map[i, 1]
        
        if idx_x < n_dof: u_aff[idx_x] = ux
        if idx_y < n_dof: u_aff[idx_y] = uy
        
    return u_aff

def build_indices_from_folders(root):
    if not os.path.exists(root):
        return []
    indices = []
    for d in os.listdir(root):
        if d.startswith("trajectory_"):
            try:
                indices.append(int(d.split("_")[1]))
            except:
                pass
    return sorted(indices)

# ============================================================
# Main POD Workflow
# ============================================================

def run_stage2_pod():
    print("--- Stage 2: POD with Dirichlet Consistency ---")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Reconstruction
    with open(PROJECT_PARAMETERS, "r") as f:
        parameters = KM.Parameters(f.read())
        
    mesh_base = parameters["modelers"][0]["parameters"]["input_filename"].GetString()
    mdpa_path = f"{StripMdpaExtension(mesh_base)}.mdpa"
    
    if os.path.exists(mdpa_path):
        material_parts = DetectMaterialSubModelParts(mdpa_path)
        parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
        parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(
            "StructuralMaterials.json"
        )
    
    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    
    n_dof, eq_map, _ = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    dir_dofs, _ = ExtractDirichletBoundaryConditions(mp)
    all_dofs = np.arange(n_dof)
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[dir_dofs] = False
    free_dofs = all_dofs[free_mask]
    
    sim._InitializeDomainCenterIfNeeded(mp)
    Xc, Yc = sim._x0c, sim._y0c
    
    print(f"Mesh: {mesh_base}")
    print(f"DOFs: {n_dof} (Free: {len(free_dofs)}, Dirichlet: {len(dir_dofs)})")
    print(f"Center: ({Xc:.4f}, {Yc:.4f})")
    
    # 2. Snapshot Collection
    traj_indices = build_indices_from_folders(SNAPSHOTS_ROOT)
    if not traj_indices:
        print(f"[ERROR] No trajectory folders found in {SNAPSHOTS_ROOT}")
        return

    W_f_list = []
    dirichlet_residuals = []
    total_snapshots = 0

    for idx in traj_indices:
        traj_dir = os.path.join(SNAPSHOTS_ROOT, f"trajectory_{idx}")
        u_file = os.path.join(traj_dir, f"trajectory_{idx}_U.npy")
        e_file = os.path.join(traj_dir, f"trajectory_{idx}_applied_strain.npy")
        
        if not (os.path.exists(u_file) and os.path.exists(e_file)):
            print(f" [Skip] Trajectory {idx}: missing U or applied_strain")
            continue
            
        U_hist = np.load(u_file)
        E_hist = np.load(e_file)
        
        n_snaps = U_hist.shape[0]
        print(f" Processing Trajectory {idx} ({n_snaps} snapshots)...")
        
        for k in range(n_snaps):
            Uk = U_hist[k]
            Ek = E_hist[k]
            
            # Substract affine lifting
            u_aff_k = get_affine_lifting(mp, Ek, Xc, Yc, eq_map, n_dof)
            wk = Uk - u_aff_k
            
            # Dirichlet Check (Strong consistency test)
            wD = wk[dir_dofs]
            res_norm = np.linalg.norm(wD)
            dirichlet_residuals.append(res_norm)
            
            # Collect free fluctuation
            W_f_list.append(wk[free_dofs])
            total_snapshots += 1

    if not W_f_list:
        print("[ERROR] No snapshots collected.")
        return

    Wf = np.column_stack(W_f_list)
    max_wD = np.max(dirichlet_residuals)
    mean_wD = np.mean(dirichlet_residuals)
    
    print(f"Total Snapshots: {total_snapshots}")
    print(f"Strong BC Consistency Check: Max ||w_D|| = {max_wD:.3e}, Mean = {mean_wD:.3e}")
    
    if max_wD > 1e-8:
        print("[WARNING] Dirichlet fluctuations are higher than expected. Check lifting logic.")

    # 3. SVD / POD
    print(" Computing SVD...")
    U_pod, S, _ = np.linalg.svd(Wf, full_matrices=False)
    
    # Selection based on epsilon^2 energy
    energy = S**2
    cumulative_energy = np.cumsum(energy) / np.sum(energy)
    rank = np.searchsorted(cumulative_energy, 1.0 - EPS2_TOL) + 1
    rank = int(min(rank, U_pod.shape[1]))
    
    Phi_f = U_pod[:, :rank].copy()
    
    print(f" POD Rank chosen: {rank} (Energy captured: {cumulative_energy[rank-1]:.6%})")

    # 4. Save Basis and Metadata
    np.save(os.path.join(OUT_DIR, "pod_basis_free.npy"), Phi_f)
    np.save(os.path.join(OUT_DIR, "free_dofs.npy"), free_dofs)
    np.save(os.path.join(OUT_DIR, "dirichlet_dofs.npy"), dir_dofs)
    np.save(os.path.join(OUT_DIR, "eq_map.npy"), eq_map)
    np.save(os.path.join(OUT_DIR, "domain_center.npy"), np.array([Xc, Yc]))
    print(f" Saved POD basis and metadata to {OUT_DIR}")

    # 5. Compute and Save Affine Lifting Basis (Phi_d)
    # u_lift = Phi_d * [Exx, Eyy, Gxy]
    phi_d = np.zeros((n_dof, 3))
    for i, node in enumerate(mp.Nodes):
        dx = node.X - Xc
        dy = node.Y - Yc
        
        ix = eq_map[i, 0]
        iy = eq_map[i, 1]
        
        # Exx column
        if ix >= 0: phi_d[ix, 0] = dx
        # Eyy column
        if iy >= 0: phi_d[iy, 1] = dy
        # Gxy column
        if ix >= 0: phi_d[ix, 2] = 0.5 * dy
        if iy >= 0: phi_d[iy, 2] = 0.5 * dx
        
    np.save(os.path.join(OUT_DIR, "phi_affine_lifting.npy"), phi_d)
    print(f" Saved Affine Lifting Basis to {os.path.join(OUT_DIR, 'phi_affine_lifting.npy')}")

    # 6. Diagnostics
    if SAVE_DIAGNOSTICS:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(S, 'o-', markersize=3)
        plt.yscale('log')
        plt.title("Singular Value Decay")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(1.0 - cumulative_energy, 'o-', markersize=3)
        plt.yscale('log')
        plt.title("Residual Energy (1 - CumSum)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "pod_diagnostics.png"))
        print(f" Diagnostics plot saved as {os.path.join(OUT_DIR, 'pod_diagnostics.png')}")

if __name__ == "__main__":
    run_stage2_pod()
