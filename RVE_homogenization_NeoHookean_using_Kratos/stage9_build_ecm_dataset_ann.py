#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 9a: Build ECM dataset for HPROM-ANN.

This collects residuals and homogenized data using the ANN manifold tangent:
  J_m = phi_p + phi_s * d(ANN)/dq_p
"""

import os
import sys
import numpy as np
import torch
import tqdm

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    PrecomputeElementIntegrationWeights,
    EvaluateGaussPointData,
    SetInputMeshFilename,
    StripMdpaExtension,
    setup_kratos_parameters,
)
from stage7b_train_ann_manifold import ManifoldANN

SNAPSHOTS_DIR = "stage_1_training_set_fom"
BASIS_DIR = "stage_2_pod_rve"
ANN_DIR = "stage_7_ann_data"
OUT_DIR = "stage_9_ecm_dataset_ann"
SNAPSHOT_PERCENT = 1.0
SEED = 42

def _get_stratified_indices(n_total, n_pick, seed=SEED):
    if n_pick <= 0: return np.array([], dtype=int)
    if n_pick >= n_total: return np.arange(n_total, dtype=int)
    rng = np.random.default_rng(int(seed))
    picks = np.zeros(n_pick, dtype=int)
    edges = np.linspace(0, n_total, n_pick + 1, dtype=int)
    for i in range(n_pick):
        i0, i1 = int(edges[i]), int(edges[i + 1])
        picks[i] = int(rng.integers(i0, i1)) if i1 > i0 else i0
    return np.sort(np.unique(picks))

def _build_free_map(n_dof, free_dofs):
    g2f = -np.ones(int(n_dof), dtype=int)
    for i, gdof in enumerate(np.asarray(free_dofs, dtype=np.int64)):
        g2f[int(gdof)] = int(i)
    return g2f

def _allocate_memmaps(out_dir, nq, n_snapshots, n_elem):
    os.makedirs(out_dir, exist_ok=True)
    q_path = os.path.join(out_dir, "Q_ecm.dat")
    b_path = os.path.join(out_dir, "b_full.dat")
    c_path = os.path.join(out_dir, "C_hom.dat")
    bh_path = os.path.join(out_dir, "b_hom.dat")
    for p in (q_path, b_path, c_path, bh_path):
        if os.path.exists(p): os.remove(p)
    q_ecm = np.memmap(q_path, dtype="float64", mode="w+", shape=(int(nq) * int(n_snapshots), int(n_elem)))
    b_full = np.memmap(b_path, dtype="float64", mode="w+", shape=(int(nq) * int(n_snapshots),))
    c_hom = np.memmap(c_path, dtype="float64", mode="w+", shape=(6 * int(n_snapshots), int(n_elem)))
    b_hom = np.memmap(bh_path, dtype="float64", mode="w+", shape=(6 * int(n_snapshots),))
    return q_ecm, b_full, c_hom, b_hom

def _load_ann_for_hprom(ann_dir, device):
    meta = np.load(os.path.join(ann_dir, "manifold_ann_metadata.npz"))
    d_meta = np.load(os.path.join(ann_dir, "ann_dataset_metadata.npz"))
    in_dim = int(np.ravel(d_meta["input_dim"])[0])
    out_dim = int(np.ravel(d_meta["n_secondary"])[0])
    model = ManifoldANN(meta["x_mean"], meta["x_std"], meta["y_mean"], meta["y_std"], in_dim=in_dim, out_dim=out_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(ann_dir, "manifold_ann.pt"), map_location=device))
    model.eval()
    return model, bool(int(np.ravel(d_meta["include_macro_strain_input"])[0]))

def main():
    print("--- Stage 9a: Building ECM Dataset for HPROM-ANN ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann_model, include_macro = _load_ann_for_hprom(ANN_DIR, device)
    
    phi_p = np.load(os.path.join(ANN_DIR, "phi_p.npy"))
    phi_s = np.load(os.path.join(ANN_DIR, "phi_s.npy"))
    free_dofs = np.load(os.path.join(BASIS_DIR, "free_dofs.npy"))
    eq_map_ref = np.load(os.path.join(BASIS_DIR, "eq_map.npy"))
    
    n_primary = int(phi_p.shape[1])
    n_secondary = int(phi_s.shape[1])
    n_total_dofs_ref = int(len(free_dofs) + len(np.load(os.path.join(BASIS_DIR, "dirichlet_dofs.npy"))))

    trajectories = sorted([d for d in os.listdir(SNAPSHOTS_DIR) if d.startswith("trajectory_")])
    pct = float(SNAPSHOT_PERCENT) / 100.0
    all_tasks = []
    total_snapshots = 0
    for traj in trajectories:
        u_file = os.path.join(SNAPSHOTS_DIR, traj, f"{traj}_U.npy")
        e_file = os.path.join(SNAPSHOTS_DIR, traj, f"{traj}_applied_strain.npy")
        if not (os.path.exists(u_file) and os.path.exists(e_file)): continue
        U_meta = np.load(u_file, mmap_mode="r")
        E_meta = np.load(e_file, mmap_mode="r")
        n_steps = min(int(U_meta.shape[0]), int(E_meta.shape[0])) if U_meta.shape[1] == n_total_dofs_ref else min(int(U_meta.shape[1]), int(E_meta.shape[0]))
        idx = _get_stratified_indices(n_steps, int(np.ceil(pct * n_steps)))
        total_snapshots += len(idx)
        all_tasks.append((traj, u_file, e_file, idx))

    print(f"[Info] Total snapshots to process: {total_snapshots}")
    
    # Use the robust parameter setup that handles materials/MDPA automatically
    parameters = setup_kratos_parameters("rve_geometry")
    
    model_kratos = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model_kratos, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    pi = mp.ProcessInfo
    
    n_dof, eq_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    if int(n_dof) != n_total_dofs_ref:
        raise RuntimeError(f"DOF mismatch: runtime={n_dof}, expected={n_total_dofs_ref}")

    elements = list(mp.Elements)
    w_gp, _ = PrecomputeElementIntegrationWeights(elements)
    g2f = _build_free_map(n_dof, free_dofs)
    
    sim._InitializeDomainCenterIfNeeded(mp)
    x0c, y0c = float(sim._x0c), float(sim._y0c)
    
    # Pre-build coordinate maps for affine lifting
    dof_x = np.zeros(n_dof, dtype=float)
    dof_y = np.zeros(n_dof, dtype=float)
    is_x_dof = np.zeros(n_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr, yr = float(node.X0) - x0c, float(node.Y0) - y0c
        ix, iy = int(eq_map[i, 0]), int(eq_map[i, 1])
        if 0 <= ix < n_dof:
            dof_x[ix], dof_y[ix], is_x_dof[ix] = xr, yr, True
        if 0 <= iy < n_dof:
            dof_x[iy], dof_y[iy], is_x_dof[iy] = xr, yr, False
            
    x_free, y_free, is_x_free = dof_x[free_dofs], dof_y[free_dofs], is_x_dof[free_dofs]

    def _affine_free(e_vec):
        F = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (F[0,0]-1.0)*x_free + F[0,1]*y_free
        uy = F[1,0]*x_free + (F[1,1]-1.0)*y_free
        return np.where(is_x_free, ux, uy)

    Q_ecm, b_full, C_hom, b_hom = _allocate_memmaps(OUT_DIR, n_primary, total_snapshots, len(elements))
    s_global = 0

    def compute_j_ann(qp_val, e_macro):
        qp_t = torch.from_numpy(qp_val.astype(np.float32)).unsqueeze(0).to(device)
        e_t = torch.from_numpy(e_macro.astype(np.float32)).unsqueeze(0).to(device)
        with torch.enable_grad():
            qp_in = qp_t.clone().detach().requires_grad_(True)
            def ann_wrap(q_loc):
                inp = torch.cat([q_loc, e_t], dim=1) if include_macro else q_loc
                return ann_model.output_unscaler(ann_model.mlp(ann_model.input_scaler(inp)))
            jac = torch.autograd.functional.jacobian(ann_wrap, qp_in).reshape(n_secondary, n_primary)
        return jac.cpu().numpy()

    for traj_name, u_path, e_path, snapshot_indices in all_tasks:
        U_all, E_all = np.load(u_path), np.load(e_path)
        for k in tqdm.tqdm(snapshot_indices, desc=traj_name):
            u_snap = U_all[k] if U_all.shape[1] == n_dof else U_all[:,k]
            e_macro = E_all[k]
            SetDisplacementFromEquationVector(u_snap, eq_map, ta)
            UpdateCurrentCoordinatesFromDisplacement(mp)
            q_p = (u_snap[free_dofs] - _affine_free(e_macro)) @ phi_p
            J_m = phi_p + phi_s @ compute_j_ann(q_p, e_macro)
            
            q_block = np.zeros((n_primary, len(elements)))
            for i, elem in enumerate(elements):
                RHS = KM.Vector()
                elem.CalculateRightHandSide(RHS, mp.ProcessInfo)
                r_e = np.array(RHS)
                e_dofs = np.array(elem.EquationIdVector(mp.ProcessInfo))
                valid = e_dofs >= 0
                rows = g2f[e_dofs[valid]]
                v2 = rows >= 0
                if np.any(v2):
                    q_block[:, i] = J_m[rows[v2]].T @ r_e[valid][v2]
            
            eps_gp, sig_gp, _ = EvaluateGaussPointData(elements, mp)
            c_block = np.zeros((6, len(elements)))
            c_block[0:3] = np.sum(w_gp[..., None] * eps_gp, axis=1).T
            c_block[3:6] = np.sum(w_gp[..., None] * sig_gp, axis=1).T
            
            Q_ecm[n_primary*s_global : n_primary*(s_global+1), :] = q_block
            b_full[n_primary*s_global : n_primary*(s_global+1)] = np.sum(q_block, axis=1)
            C_hom[6*s_global : 6*(s_global+1), :] = c_block
            b_hom[6*s_global : 6*(s_global+1)] = np.sum(c_block, axis=1)
            s_global += 1

    Q_ecm.flush(); b_full.flush(); C_hom.flush(); b_hom.flush()
    np.savez(os.path.join(OUT_DIR, "meta.npz"), n_primary=n_primary, n_secondary=n_secondary, n_elem=len(elements), N_s=total_snapshots)
    print(f"[DONE] Dataset saved to {OUT_DIR}")

if __name__ == "__main__": main()
