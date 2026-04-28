#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 9a: Build ECM dataset for HPROM-RBF.

This mirrors Stage 5a sampling/assembly flow, but the residual projection
uses the RBF manifold tangent:
  J_m = phi_p + phi_s * dN_rbf/dq_p
instead of the fixed linear POD basis.
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
    DeformationGradientFromGreenLagrange2D,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    PrecomputeElementIntegrationWeights,
    EvaluateGaussPointData,
    SetInputMeshFilename,
    StripMdpaExtension,
    DetectMaterialSubModelParts,
    ConfigureElementModelerForMaterialParts,
    WriteRuntimeMaterialsFile,
)
from rbf_manifold_model import load_rbf_model, evaluate_rbf_map_and_jacobian_qp


SNAPSHOTS_DIR = "stage_1_training_set_fom"
BASIS_DIR = "stage_2_pod_rve"
RBF_DIR = "stage_7_rbf_data"
OUT_DIR = "stage_9_ecm_dataset_rbf"
SNAPSHOT_PERCENT = 1.0
SEED = 42


def _get_stratified_indices(n_total, n_pick, seed=SEED):
    if n_pick <= 0:
        return np.array([], dtype=int)
    if n_pick >= n_total:
        return np.arange(n_total, dtype=int)

    rng = np.random.default_rng(int(seed))
    picks = np.zeros(n_pick, dtype=int)
    edges = np.linspace(0, n_total, n_pick + 1, dtype=int)
    for i in range(n_pick):
        i0, i1 = int(edges[i]), int(edges[i + 1])
        if i1 <= i0:
            picks[i] = i0
        else:
            picks[i] = int(rng.integers(i0, i1))
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
        if os.path.exists(p):
            os.remove(p)

    q_ecm = np.memmap(q_path, dtype="float64", mode="w+", shape=(int(nq) * int(n_snapshots), int(n_elem)))
    b_full = np.memmap(b_path, dtype="float64", mode="w+", shape=(int(nq) * int(n_snapshots),))
    c_hom = np.memmap(c_path, dtype="float64", mode="w+", shape=(6 * int(n_snapshots), int(n_elem)))
    b_hom = np.memmap(bh_path, dtype="float64", mode="w+", shape=(6 * int(n_snapshots),))
    return q_ecm, b_full, c_hom, b_hom


def _extract_snapshot_u(U_all, step_idx, n_dof):
    if U_all.ndim != 2:
        raise ValueError(f"Invalid snapshot array shape {U_all.shape}. Expected 2D.")
    if U_all.shape[1] == n_dof:
        return np.asarray(U_all[step_idx, :], dtype=float)
    if U_all.shape[0] == n_dof:
        return np.asarray(U_all[:, step_idx], dtype=float)
    raise ValueError(
        f"Cannot infer displacement layout from shape {U_all.shape} with n_dof={n_dof}."
    )


def main():
    print("--- Stage 9a: Building ECM Dataset for HPROM-RBF ---")

    phi_p = np.asarray(np.load(os.path.join(RBF_DIR, "phi_p.npy")), dtype=float)
    phi_s = np.asarray(np.load(os.path.join(RBF_DIR, "phi_s.npy")), dtype=float)
    free_dofs = np.asarray(np.load(os.path.join(BASIS_DIR, "free_dofs.npy")), dtype=np.int64)
    dir_dofs = np.asarray(np.load(os.path.join(BASIS_DIR, "dirichlet_dofs.npy")), dtype=np.int64)
    eq_map_ref = np.asarray(np.load(os.path.join(BASIS_DIR, "eq_map.npy")), dtype=np.int64)
    rbf_model = load_rbf_model(os.path.join(RBF_DIR, "rbf_model.npz"))

    n_primary = int(phi_p.shape[1])
    n_secondary = int(phi_s.shape[1])
    n_free = int(phi_p.shape[0])
    if int(phi_s.shape[0]) != n_free:
        raise ValueError("phi_p and phi_s row count mismatch.")
    if int(rbf_model["n_primary"]) != n_primary:
        raise ValueError("RBF model n_primary does not match phi_p.")
    if int(rbf_model["n_secondary"]) != n_secondary:
        raise ValueError("RBF model n_secondary does not match phi_s.")

    include_macro = bool(rbf_model["include_macro_strain_input"])
    n_total_dofs_ref = int(len(free_dofs) + len(dir_dofs))
    print(f"[Info] n_primary={n_primary}, n_secondary={n_secondary}, include_macro_strain_input={int(include_macro)}")

    trajectories = sorted([d for d in os.listdir(SNAPSHOTS_DIR) if d.startswith("trajectory_")])
    if not trajectories:
        raise FileNotFoundError(f"No trajectory folders found in {SNAPSHOTS_DIR}")

    pct = float(SNAPSHOT_PERCENT) / 100.0
    total_snapshots = 0
    all_tasks = []

    for traj in trajectories:
        u_file = os.path.join(SNAPSHOTS_DIR, traj, f"{traj}_U.npy")
        e_file = os.path.join(SNAPSHOTS_DIR, traj, f"{traj}_applied_strain.npy")
        if not (os.path.exists(u_file) and os.path.exists(e_file)):
            continue

        U_meta = np.load(u_file, mmap_mode="r")
        E_meta = np.load(e_file, mmap_mode="r")
        if U_meta.ndim != 2 or E_meta.ndim != 2 or E_meta.shape[1] != 3:
            continue

        if U_meta.shape[1] == n_total_dofs_ref:
            n_u_steps = int(U_meta.shape[0])
        elif U_meta.shape[0] == n_total_dofs_ref:
            n_u_steps = int(U_meta.shape[1])
        else:
            continue

        n_steps = min(n_u_steps, int(E_meta.shape[0]))
        if n_steps <= 0:
            continue

        n_pick = int(np.ceil(pct * n_steps))
        idx = _get_stratified_indices(n_steps, n_pick)
        total_snapshots += int(len(idx))
        all_tasks.append((traj, u_file, e_file, idx))

    if total_snapshots <= 0 or not all_tasks:
        raise RuntimeError("No valid snapshots selected for Stage 9a.")

    print(f"[Info] Target: {SNAPSHOT_PERCENT}% snapshots stratified across {len(all_tasks)} trajectories.")
    print(f"[Info] Total snapshots to process: {total_snapshots}")

    with open("ProjectParameters.json", "r", encoding="utf-8") as f:
        parameters = KM.Parameters(f.read())
    SetInputMeshFilename(parameters, "rve_geometry")

    mesh_base = parameters["modelers"][0]["parameters"]["input_filename"].GetString()
    mdpa_path = f"{StripMdpaExtension(mesh_base)}.mdpa"
    if os.path.exists(mdpa_path):
        material_parts = DetectMaterialSubModelParts(mdpa_path)
        parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
        runtime_materials = WriteRuntimeMaterialsFile(
            material_parts=material_parts, young_mpa=1628.0, poisson=0.4
        )
        parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(runtime_materials)
        print(f"[Info] Material parts: {material_parts}")

    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    pi = mp.ProcessInfo

    n_dof, eq_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    if int(n_dof) != n_total_dofs_ref:
        raise RuntimeError(f"DOF mismatch: runtime={n_dof}, reference={n_total_dofs_ref}.")
    if eq_map.shape == eq_map_ref.shape and not np.array_equal(eq_map, eq_map_ref):
        raise RuntimeError("eq_map mismatch between Stage 2 metadata and runtime model.")

    elements = list(mp.Elements)
    n_elem = len(elements)
    w_gp, area_e = PrecomputeElementIntegrationWeights(elements)
    g2f = _build_free_map(n_dof, free_dofs)

    sim._InitializeDomainCenterIfNeeded(mp)
    x0c, y0c = float(sim._x0c), float(sim._y0c)
    dof_x = np.zeros(n_dof, dtype=float)
    dof_y = np.zeros(n_dof, dtype=float)
    is_x_dof = np.zeros(n_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        idx_x = int(eq_map[i, 0])
        idx_y = int(eq_map[i, 1])
        if 0 <= idx_x < n_dof:
            dof_x[idx_x] = xr
            dof_y[idx_x] = yr
            is_x_dof[idx_x] = True
        if 0 <= idx_y < n_dof:
            dof_x[idx_y] = xr
            dof_y[idx_y] = yr
            is_x_dof[idx_y] = False

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    def _affine_free(e_macro):
        F = DeformationGradientFromGreenLagrange2D(np.asarray(e_macro, dtype=float))
        ux = (F[0, 0] - 1.0) * x_free + F[0, 1] * y_free
        uy = F[1, 0] * x_free + (F[1, 1] - 1.0) * y_free
        return np.where(is_x_free, ux, uy)

    elem_dofs = []
    elem_masks = []
    for elem in elements:
        ids = np.array(elem.EquationIdVector(pi), dtype=int)
        mask = ids >= 0
        elem_dofs.append(ids[mask])
        elem_masks.append(mask)

    Q_ecm, b_full, C_hom, b_hom = _allocate_memmaps(OUT_DIR, n_primary, total_snapshots, n_elem)

    s_global = 0
    for traj_name, u_path, e_path, snapshot_indices in all_tasks:
        print(f"  > Processing {traj_name} ({len(snapshot_indices)} sampled steps)...")
        U_all = np.load(u_path)
        E_all = np.load(e_path)
        for k in snapshot_indices:
            u_snap = _extract_snapshot_u(U_all, int(k), int(n_dof))
            e_macro = np.asarray(E_all[int(k), :], dtype=float)

            SetDisplacementFromEquationVector(u_snap, eq_map, ta)
            UpdateCurrentCoordinatesFromDisplacement(mp)

            u_free = u_snap[free_dofs]
            u_aff_free = _affine_free(e_macro)
            w_free = u_free - u_aff_free
            q_p = w_free @ phi_p

            if include_macro:
                x_in = np.concatenate([q_p, e_macro], axis=0)
            else:
                x_in = q_p
            _, j_qs_qp = evaluate_rbf_map_and_jacobian_qp(x_in, rbf_model, n_primary)
            j_qs_qp = np.asarray(j_qs_qp, dtype=float)
            if j_qs_qp.shape != (n_secondary, n_primary):
                raise RuntimeError(
                    f"Invalid RBF Jacobian shape {j_qs_qp.shape}, expected {(n_secondary, n_primary)}."
                )
            J_m = phi_p + phi_s @ j_qs_qp

            q_block = np.zeros((n_primary, n_elem), dtype=float)
            c_block = np.zeros((6, n_elem), dtype=float)

            for i, elem in enumerate(elements):
                RHS = KM.Vector()
                elem.CalculateRightHandSide(RHS, pi)
                rhs_arr = np.array(RHS, dtype=float)

                local_mask = elem_masks[i]
                local_dofs = elem_dofs[i]
                rows = g2f[local_dofs]
                valid = rows >= 0
                if np.any(valid):
                    j_e = J_m[rows[valid], :]
                    r_e = rhs_arr[local_mask][valid]
                    q_block[:, i] = j_e.T @ r_e

            eps_gp, sig_gp, _ = EvaluateGaussPointData(elements, mp)
            c_block[0:3, :] = np.sum(w_gp[..., None] * eps_gp, axis=1).T
            c_block[3:6, :] = np.sum(w_gp[..., None] * sig_gp, axis=1).T

            r0, r1 = n_primary * s_global, n_primary * (s_global + 1)
            Q_ecm[r0:r1, :] = q_block
            b_full[r0:r1] = np.sum(q_block, axis=1)

            h0, h1 = 6 * s_global, 6 * (s_global + 1)
            C_hom[h0:h1, :] = c_block
            b_hom[h0:h1] = np.sum(c_block, axis=1)
            s_global += 1

    Q_ecm.flush()
    b_full.flush()
    C_hom.flush()
    b_hom.flush()

    np.savez(
        os.path.join(OUT_DIR, "meta.npz"),
        nq=n_primary,
        N_s=total_snapshots,
        n_elem=n_elem,
        snapshot_percent=SNAPSHOT_PERCENT,
        A_total=np.sum(area_e),
        n_primary=n_primary,
        n_secondary=n_secondary,
        include_macro_strain_input=np.array([1 if include_macro else 0], dtype=np.int64),
        rbf_data_dir=np.array([RBF_DIR]),
    )

    sim.Finalize()

    print("\n[DONE] Stage 9a dataset generation complete.")
    print(f"      - Q_ecm shape: {Q_ecm.shape}")
    print(f"      - Datasets saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

