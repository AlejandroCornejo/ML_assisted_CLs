#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POD pure reconstruction test for the RVE J2 problem.

Workflow:
- Load:
    * data_set/all_displacement_snapshots.npz  (U tensor)
    * modes/U_modes_tol_*.npy                 (POD basis Phi)
- Choose a batch (parameter combination) and restrict to the first 51 steps.
- Reconstruct displacements via pure projection:
        U_POD(t) = Phi Phi^T U_FOM(t)
- Rebuild the RVE mesh from ProjectParameters.json, precompute B_all etc.
- Compute homogenized strain & stress histories from:
        - U_FOM(t)
        - U_POD(t)
  using the J2 law.
- Plot FOM vs POD stress–strain curves for that batch.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import importlib

import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage

from j2_plane_stress_plastic_strain_rve_simo_optimized import (
    VonMisesIsotropicPlasticityPlaneStress
)

# ----------------------------------------------------------------------
# User parameters
# ----------------------------------------------------------------------
# 1-based batch index: 1 = first parameter combo, 9 = ninth, etc.
batch_index_user = 8

# Number of time steps to use (including the initial configuration).
# You mentioned 51 steps per training parameter combination.
n_steps_use = 51

# POD basis file to use
basis_file = "modes/U_modes_tol_1e-16.npy"

# Paths to data files
disp_npz_path      = "data_set/all_displacement_snapshots.npz"
project_params_file = "ProjectParameters.json"


# ----------------------------------------------------------------------
# Helpers from your solver: mesh + B-matrices
# ----------------------------------------------------------------------
def build_node_global_map(mp):
    """
    Deterministic map:
      [ux(node0), uy(node0), ux(node1), uy(node1), ...]
    following mp.Nodes iteration.
    """
    idx_ux, idx_uy = {}, {}
    for k, node in enumerate(mp.Nodes):
        idx_ux[node.Id] = 2 * k
        idx_uy[node.Id] = 2 * k + 1
    n_dof = 2 * mp.NumberOfNodes()
    return idx_ux, idx_uy, n_dof


def build_B_from_DNDX(DNDX):
    """
    DNDX: (nnode, 2) with columns [dN/dx, dN/dy] in global coords.
    Returns B (3, 2*nnode) for 2D small strain with ENGINEERING shear (γxy).
    Voigt order: [εxx, εyy, γxy].
    """
    nnode = DNDX.shape[0]
    B = np.zeros((3, 2 * nnode))
    for a in range(nnode):
        Nx, Ny = DNDX[a, 0], DNDX[a, 1]
        # εxx
        B[0, 2 * a]     = Nx
        B[0, 2 * a + 1] = 0.0
        # εyy
        B[1, 2 * a]     = 0.0
        B[1, 2 * a + 1] = Ny
        # γxy
        B[2, 2 * a]     = Ny
        B[2, 2 * a + 1] = Nx
    return B


def precompute_mesh_arrays(mp, idx_ux, idx_uy):
    """
    Precompute element connectivity, B-matrices, GP weights and areas.

    Returns
    -------
    conn      : (n_elem, nd)        DOF ids per element
    B_all     : (n_elem, n_gp, 3, nd)
    w_all     : (n_elem, n_gp)
    area_all  : (n_elem,)
    """
    elems = list(mp.Elements)
    n_elem = len(elems)

    if n_elem == 0:
        raise RuntimeError("No elements found in the computing model part.")

    geom0 = elems[0].GetGeometry()
    nnode = geom0.PointsNumber()
    nd    = 2 * nnode

    Ns   = np.array(geom0.ShapeFunctionsValues())
    n_gp = Ns.shape[0]

    conn     = np.zeros((n_elem, nd), dtype=int)
    B_all    = np.zeros((n_elem, n_gp, 3, nd), dtype=float)
    w_all    = np.zeros((n_elem, n_gp), dtype=float)
    area_all = np.zeros(n_elem, dtype=float)

    for e_idx, elem in enumerate(elems):
        geom = elem.GetGeometry()
        area = geom.Area()
        area_all[e_idx] = area

        # DOF indices
        col_ids = []
        for node in geom:
            col_ids.append(idx_ux[node.Id])
            col_ids.append(idx_uy[node.Id])
        conn[e_idx, :] = np.array(col_ids, dtype=int)

        for igauss in range(n_gp):
            DNDe = np.array(geom.ShapeFunctionDerivatives(1, igauss))
            J    = np.array(geom.Jacobian(igauss))
            DNDX = DNDe @ np.linalg.inv(J)
            B_loc = build_B_from_DNDX(DNDX)
            B_all[e_idx, igauss, :, :] = B_loc
            w_all[e_idx, igauss] = area / n_gp

    return conn, B_all, w_all, area_all


# ----------------------------------------------------------------------
# Kratos: build model + mesh + material
# ----------------------------------------------------------------------
def load_rve_mesh_and_material(params_file):
    """
    Builds the Kratos model from ProjectParameters.json, returns:
      - n_dof
      - conn, B_all, w_all, area_all
      - material object for J2
    """
    if not os.path.isfile(params_file):
        raise FileNotFoundError(f"ProjectParameters.json not found at {params_file}")

    with open(params_file, "r") as parameter_file:
        parameters = KM.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    model = KM.Model()
    simulation = analysis_stage_class(model, parameters)
    simulation.Initialize()

    mp = simulation._GetSolver().GetComputingModelPart()

    idx_ux, idx_uy, n_dof = build_node_global_map(mp)
    conn, B_all, w_all, area_all = precompute_mesh_arrays(mp, idx_ux, idx_uy)

    # Material properties from first element
    elem0 = next(iter(mp.Elements))
    props0 = elem0.Properties
    E        = props0[KM.YOUNG_MODULUS]
    nu       = props0[KM.POISSON_RATIO]
    sigma_y0 = props0[KM.YIELD_STRESS]
    H        = 0.0   # same as in your solver

    mat = VonMisesIsotropicPlasticityPlaneStress(E, nu, sigma_y0, H)

    print("[INFO] RVE mesh and material loaded:")
    print(f"       n_dof       = {n_dof}")
    print(f"       n_elements  = {conn.shape[0]}")
    print(f"       n_gp        = {B_all.shape[1]}")
    print(f"       E           = {E}")
    print(f"       nu          = {nu}")
    print(f"       sigma_y0    = {sigma_y0}")
    print(f"       H           = {H}")

    return n_dof, conn, B_all, w_all, area_all, mat


# ----------------------------------------------------------------------
# J2-based homogenization for a given displacement history
# ----------------------------------------------------------------------
def homogenized_history_from_displacements(U_traj, conn, B_all, w_all, area_all, mat):
    """
    Given a displacement history U_traj (n_steps, n_dof), compute the
    homogenized strain and stress histories using the J2 law and the
    precomputed B_all, w_all, area_all.

    This re-integrates the J2 internal variables (eps_p, alpha) step by step,
    starting from zero plastic strain.
    """
    n_steps, n_dof = U_traj.shape
    n_elem, n_gp   = B_all.shape[0], B_all.shape[1]

    # Plastic state (committed): start from zero
    eps_p_n = np.zeros((n_elem, n_gp, 3), dtype=float)
    alpha_n = np.zeros((n_elem, n_gp), dtype=float)

    strain_hist = []
    stress_hist = []

    RVE_area = np.sum(area_all)
    if RVE_area <= 0.0:
        raise RuntimeError("RVE_area is non-positive, check mesh / areas.")

    for k in range(n_steps):
        u = U_traj[k, :]  # (n_dof,)

        # Element displacements
        u_e = u[conn]  # (n_elem, nd)

        # Strains at all GPs
        eps = np.einsum('egij,ej->egi', B_all, u_e)  # (n_elem, n_gp, 3)

        # J2 return mapping: update plastic state
        eps_flat     = eps.reshape(-1, 3)
        eps_p_n_flat = eps_p_n.reshape(-1, 3)
        alpha_n_flat = alpha_n.reshape(-1)

        sigma_flat, eps_p_flat, alpha_flat, _ = mat._return_mapping_batch(
            eps_flat, eps_p_n_flat, alpha_n_flat
        )

        sigma = sigma_flat.reshape(n_elem, n_gp, 3)
        eps_p_n = eps_p_flat.reshape(n_elem, n_gp, 3)
        alpha_n = alpha_flat.reshape(n_elem, n_gp)

        # Homogenization (area-weighted average)
        w = w_all[..., None]  # (n_elem, n_gp, 1)
        hom_strain_raw = np.sum(w * eps, axis=(0, 1))    # (3,)
        hom_stress_raw = np.sum(w * sigma, axis=(0, 1))  # (3,)

        hom_strain = hom_strain_raw / RVE_area
        hom_stress = hom_stress_raw / RVE_area

        strain_hist.append(hom_strain)
        stress_hist.append(hom_stress)

    return np.array(strain_hist), np.array(stress_hist)


# ----------------------------------------------------------------------
# Plot FOM vs POD stress–strain
# ----------------------------------------------------------------------
def plot_stress_strain_fom_vs_pod(strain_fom, stress_fom,
                                  strain_pod, stress_pod,
                                  batch_index_user,
                                  save_dir="data_set"):
    os.makedirs(save_dir, exist_ok=True)

    labels_comp = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$"]
    colors      = ["r", "b", "k"]

    plt.figure(figsize=(8, 6))

    for i in range(3):
        plt.plot(
            strain_fom[:, i], stress_fom[:, i],
            color=colors[i], linestyle='-', marker='o',
            label=f"FOM {labels_comp[i]}"
        )
        plt.plot(
            strain_pod[:, i], stress_pod[:, i],
            color=colors[i], linestyle='--', marker='',
            label=f"POD {labels_comp[i]}"
        )

    plt.xlabel("Strain component [-]")
    plt.ylabel("Stress [Pa]")
    plt.title(f"Batch {batch_index_user}: FOM vs POD (homogenized)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(save_dir,
                         f"batch_{batch_index_user}_stress_strain_FOM_vs_POD.png")
    plt.savefig(fname, dpi=300)
    plt.show()
    print(f"[INFO] Saved comparison plot to {fname}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # -----------------------------------------
    # Load FOM displacement data
    # -----------------------------------------
    if not os.path.isfile(disp_npz_path):
        raise FileNotFoundError(disp_npz_path)

    disp_data = np.load(disp_npz_path)
    U_tensor  = disp_data["U"]        # (n_batches, n_steps_total, n_dof)

    n_batches, n_steps_total, n_dof = U_tensor.shape
    print(f"[INFO] U_tensor shape = {U_tensor.shape}")

    # -----------------------------------------
    # Select batch and truncate to first n_steps_use
    # -----------------------------------------
    ibatch = batch_index_user - 1
    if ibatch < 0 or ibatch >= n_batches:
        raise IndexError(f"batch_index_user={batch_index_user} out of range "
                         f"(1..{n_batches})")

    n_steps = min(n_steps_use, n_steps_total)
    print(f"[INFO] Using batch {batch_index_user} (0-based index {ibatch})")
    print(f"[INFO] Using first {n_steps} time steps (of {n_steps_total})")

    U_fom_batch = U_tensor[ibatch, :n_steps, :]        # (n_steps, n_dof)

    # -----------------------------------------
    # Build RVE mesh + material (for J2 evaluation)
    # -----------------------------------------
    n_dof_rve, conn, B_all, w_all, area_all, mat = \
        load_rve_mesh_and_material(project_params_file)

    if n_dof_rve != n_dof:
        raise ValueError(f"n_dof from Kratos ({n_dof_rve}) != n_dof in snapshots ({n_dof}). "
                         "Check consistency between solver and POD script.")

    # -----------------------------------------
    # Homogenized FOM history from displacements
    # -----------------------------------------
    strain_fom, stress_fom = homogenized_history_from_displacements(
        U_fom_batch, conn, B_all, w_all, area_all, mat
    )

    print("[INFO] Homogenized FOM history computed.")
    print(f"       strain_fom shape = {strain_fom.shape}")
    print(f"       stress_fom shape = {stress_fom.shape}")

    # -----------------------------------------
    # Load POD basis Phi
    # -----------------------------------------
    if not os.path.isfile(basis_file):
        raise FileNotFoundError(f"POD basis file not found: {basis_file}")
    Phi = np.load(basis_file)  # (n_dof, n_modes)
    n_modes = Phi.shape[1]

    if Phi.shape[0] != n_dof:
        raise ValueError(f"POD basis size mismatch: Phi has {Phi.shape[0]} rows, "
                         f"but FOM has n_dof = {n_dof}")

    print(f"[INFO] Loaded POD basis from {basis_file}")
    print(f"       Phi shape = {Phi.shape}  (n_dof x n_modes)")
    print(f"       n_modes   = {n_modes}")

    # -----------------------------------------
    # Pure POD reconstruction: U_POD(t) = Phi Phi^T U_FOM(t)
    # -----------------------------------------
    U_fom_T = U_fom_batch.T        # (n_dof, n_steps)
    Q       = Phi.T @ U_fom_T      # (n_modes, n_steps)
    U_pod_T = Phi @ Q              # (n_dof, n_steps)
    U_pod_batch = U_pod_T.T        # (n_steps, n_dof)

    print("[INFO] POD pure reconstruction done.")
    print(f"       U_pod_batch shape = {U_pod_batch.shape}")

    # -----------------------------------------
    # Homogenized history from POD recon
    # -----------------------------------------
    strain_pod, stress_pod = homogenized_history_from_displacements(
        U_pod_batch, conn, B_all, w_all, area_all, mat
    )

    print("[INFO] Homogenized POD history computed.")
    print(f"       strain_pod shape = {strain_pod.shape}")
    print(f"       stress_pod shape = {stress_pod.shape}")

    # -----------------------------------------
    # Plot FOM vs POD
    # -----------------------------------------
    plot_stress_strain_fom_vs_pod(
        strain_fom, stress_fom,
        strain_pod, stress_pod,
        batch_index_user=batch_index_user,
        save_dir="data_set"
    )


if __name__ == "__main__":
    main()
