#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: Verification of PROM vs FOM
Runs the PROM on a specific trajectory and compares it with the FOM result.
Now featuring separate LaTeX-style plots.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()

# Kratos Path handled inside prom_solver_rve.py

from prom_solver_rve import RunPromBatchSimulation
from fom_solver_rve import (
    StripMdpaExtension,
    DetectMaterialSubModelParts,
    ConfigureElementModelerForMaterialParts,
    SetInputMeshFilename,
    LoadStrainWaypointsFromFile,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE
)
import KratosMultiphysics as KM

def run_verification(idx, mesh="rve_geometry", fom_dir="stage_1_training_set_fom"):
    print(f"--- Verification: FOM vs PROM (Trajectory {idx}) ---")
    
    # 1. Load PROM Basis and Metadata
    model_dir = "stage_2_pod_rve"
    phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(model_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(model_dir, "eq_map.npy"))
    Xc, Yc = np.load(os.path.join(model_dir, "domain_center.npy"))

    # 2. Setup Kratos Parameters
    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())
    SetInputMeshFilename(parameters, mesh)
    
    mdpa_path = f"{StripMdpaExtension(mesh)}.mdpa"
    material_parts = DetectMaterialSubModelParts(mdpa_path)
    parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
    parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(
        "StructuralMaterials.json"
    )

    # 3. Load Path
    strain_path, meta = LoadStrainWaypointsFromFile("stage_0_trajectory/stage_0_trajectories.npz", idx)
    
    # 4. Run or Load PROM
    prom_out_dir = "stage_3_verification"
    p_sig_file = os.path.join(prom_out_dir, f"prom_stress_traj_{idx}.npy")
    p_eps_file = os.path.join(prom_out_dir, f"prom_strain_traj_{idx}.npy")
    
    if os.path.exists(p_sig_file):
        print(f"[Stage 3] Found existing PROM results. Loading...")
        p_sig = np.load(p_sig_file)
        p_eps = np.load(p_eps_file)
    else:
        print(f"[Stage 3] Launching PROM Simulation...")
        p_eps, p_sig = RunPromBatchSimulation(
            parameters, phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc,
            out_dir=prom_out_dir,
            strain_path=strain_path,
            trajectory_index=idx,
            reference_amplitude=meta.get("reference_amplitude", 0.10),
            reference_steps=meta.get("ref_steps", REFERENCE_STEPS_FOR_UNIT_AMPLITUDE)
        )
        p_eps, p_sig = np.array(p_eps), np.array(p_sig)
        np.save(p_sig_file, p_sig)
        np.save(p_eps_file, p_eps)

    # 5. Load FOM Ground Truth
    fom_traj_dir = os.path.join(fom_dir, f"trajectory_{idx}")
    f_eps_file = os.path.join(fom_traj_dir, f"trajectory_{idx}_strain.npy")
    f_sig_file = os.path.join(fom_traj_dir, f"trajectory_{idx}_stress.npy")
    
    if not os.path.exists(f_eps_file):
        print(f"[ERROR] FOM data not found in {fom_traj_dir}. Please run Stage 1 first.")
        return

    f_eps = np.load(f_eps_file)
    f_sig = np.load(f_sig_file)

    # Ensure same length
    n = min(len(p_sig), len(f_sig))
    f_sig, p_sig = f_sig[:n], p_sig[:n]
    f_eps, p_eps = f_eps[:n], p_eps[:n]

    # 6. Error Analysis
    err_sig = np.linalg.norm(f_sig - p_sig) / (np.linalg.norm(f_sig) + 1e-10)
    print(f"\n==========================================")
    print(f"VERIFICATION RESULTS (Trajectory {idx})")
    print(f"Relative Stress Error: {err_sig:.4e}")
    print(f"==========================================\n")

    # 7. Plotting (Separate LaTeX-style figures)
    plt.rcParams.update({
        "font.size": 12,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral"
    })
    
    comp_labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "sigma_xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "sigma_yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "sigma_xy")
    ]
    
    for i, label_sig, label_eps, suffix in comp_labels:
        plt.figure(figsize=(7, 6))
        plt.plot(f_eps[:, i], f_sig[:, i], 'k-', label="FOM", linewidth=2)
        plt.plot(p_eps[:, i], p_sig[:, i], 'r--', label="PROM", linewidth=1.5)
        plt.title(f"Comparison: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(prom_out_dir, f"verification_traj_{idx}_{suffix}.png"), dpi=150)
        plt.close()

    # Relative Error per Step
    plt.figure(figsize=(7, 6))
    step_err = np.linalg.norm(f_sig - p_sig, axis=1) / (np.linalg.norm(f_sig, axis=1) + 1e-10)
    plt.plot(step_err, 'b-', label="Rel. Error")
    plt.title("Relative Stress Error per Step")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(prom_out_dir, f"verification_traj_{idx}_error.png"), dpi=150)
    plt.close()

    print(f"Separate verification plots saved to: {prom_out_dir}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=12, help="Trajectory index to verify.")
    args = p.parse_args()
    
    run_verification(args.index)
