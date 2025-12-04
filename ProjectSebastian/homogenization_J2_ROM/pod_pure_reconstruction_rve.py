#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POD pure reconstruction test for the RVE J2 problem (per (theta, phi, gamma, psi_deg)).

Workflow:
- Load:
    * training/testing displacement file:
          {data_dir}/U_theta{theta:.1f}_phi{phi:.1f}_gamma{gamma:.3f}_psi{psi_deg:.1f}.npy
      (full U trajectory for the 0 -> E1 -> E2 path)
    * modes/U_modes_tol_*.npy                             (POD basis Phi)
- Optionally restrict to the first n_steps_use time steps (if not None).
- Reconstruct displacements via pure projection:
        U_POD(t) = Phi Phi^T U_FOM(t)
- Rebuild the RVE mesh from ProjectParameters.json, precompute B_all etc.
- Compute homogenized strain & stress histories from:
        - U_FOM(t)
        - U_POD(t)
  using the J2 law.
- Plot FOM vs POD stress–strain curves for that (theta, phi, gamma, psi_deg).
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

# Import mesh helpers from the FOM solver (to avoid duplication)
from fom_solver_rve import build_node_global_map, precompute_mesh_arrays

# ----------------------------------------------------------------------
# User parameters
# ----------------------------------------------------------------------
# Angle combination to test (must match what you used in the FOM runs)
theta   = 50.0      # [deg]
phi     = 50.0      # [deg]
gamma   = 0.553      # radius factor used in that FOM batch
psi_deg = 2.5      # angle between E1 and E2 [deg] used in that FOM batch

# Directory where the FOM files were saved by run_fom_batch
#   e.g. "training_set" or "testing_set"
data_dir = "training_set"

# Number of time steps to use (including the initial configuration).
# If None: use all time steps contained in the file.
# If an integer: use min(n_steps_use, n_steps_total).
n_steps_use = None  # was 51 for old single-path; now we default to "all"

# POD basis file to use
basis_file = "modes/U_modes_tol_1e-16.npy"

# Path to ProjectParameters.json
project_params_file = "ProjectParameters.json"


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
    # precompute_mesh_arrays from fom_solver_rve returns:
    #   conn, B_all, w_all, area_all, pattern_I, pattern_J
    conn, B_all, w_all, area_all, _, _ = precompute_mesh_arrays(mp, idx_ux, idx_uy)

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
                                  theta, phi, gamma, psi_deg,
                                  save_dir):
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
    plt.title(
        "FOM vs POD (homogenized)\n"
        + f"theta = {theta:.1f}°, phi = {phi:.1f}°, "
        + f"gamma = {gamma:.3f}, psi = {psi_deg:.1f}°"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(
        save_dir,
        f"stress_strain_FOM_vs_POD_theta{theta:.1f}_phi{phi:.1f}"
        f"_gamma{gamma:.3f}_psi{psi_deg:.1f}.png"
    )
    plt.savefig(fname, dpi=300)
    plt.show()
    print(f"[INFO] Saved comparison plot to {fname}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # -----------------------------------------
    # Load FOM displacement data for given (theta, phi, gamma, psi)
    # -----------------------------------------
    disp_file = os.path.join(
        data_dir,
        f"U_theta{theta:.1f}_phi{phi:.1f}_gamma{gamma:.3f}_psi{psi_deg:.1f}.npy"
    )
    if not os.path.isfile(disp_file):
        raise FileNotFoundError(
            f"Displacement file not found: {disp_file}\n"
            f"Make sure run_fom_batch(theta={theta}, phi={phi}, "
            f"gamma={gamma}, psi_deg={psi_deg}, "
            f"out_dir='{data_dir}') has been executed."
        )

    U_fom_full = np.load(disp_file)  # expected shape: (n_steps_total, n_dof)
    if U_fom_full.ndim != 2:
        raise ValueError(
            f"{disp_file} has shape {U_fom_full.shape}, expected 2D (n_steps, n_dof)"
        )

    n_steps_total, n_dof = U_fom_full.shape

    if n_steps_use is None:
        n_steps = n_steps_total
    else:
        n_steps = min(n_steps_use, n_steps_total)

    print(f"[INFO] Loaded FOM displacements from {disp_file}")
    print(f"       shape = {U_fom_full.shape}")
    print(f"[INFO] Using first {n_steps} time steps (of {n_steps_total})")

    U_fom_batch = U_fom_full[:n_steps, :]   # (n_steps, n_dof)

    # -----------------------------------------
    # Build RVE mesh + material (for J2 evaluation)
    # -----------------------------------------
    n_dof_rve, conn, B_all, w_all, area_all, mat = \
        load_rve_mesh_and_material(project_params_file)

    if n_dof_rve != n_dof:
        raise ValueError(
            f"n_dof from Kratos ({n_dof_rve}) != n_dof in snapshots ({n_dof}). "
            "Check consistency between solver and POD script."
        )

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
        raise ValueError(
            f"POD basis size mismatch: Phi has {Phi.shape[0]} rows, "
            f"but FOM has n_dof = {n_dof}"
        )

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
        theta=theta, phi=phi, gamma=gamma, psi_deg=psi_deg,
        save_dir=data_dir
    )


if __name__ == "__main__":
    main()

