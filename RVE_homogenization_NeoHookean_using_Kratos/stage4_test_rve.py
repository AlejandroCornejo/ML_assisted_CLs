#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4: Testing on Unseen Path
Generates a valid non-proportional trajectory (strictly inside training envelope)
and benchmarks FOM vs PROM.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    RunFomBatchSimulation, 
    BuildDynamicSegmentSteps,
    SetInputMeshFilename, 
    StripMdpaExtension, 
    DetectMaterialSubModelParts, 
    ConfigureElementModelerForMaterialParts, 
    WriteRuntimeMaterialsFile,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT
)
from prom_solver_rve import RunPromBatchSimulation

# ============================================================
# TRAJECTORY GENERATION
# ============================================================

def generate_safe_test_path(emax, rel6, domain_type="box"):
    """
    Generates a non-proportional test trajectory strictly inside the
    training domain. Works for both ellipsoid and box domains.
    
    The waypoints are computed as fractions of the actual domain limits
    to guarantee they remain inside the training envelope.
    """
    r = np.array(rel6, dtype=float)
    exx_max = emax * r[0]
    exx_min = -emax * r[1]
    eyy_max = emax * r[2]
    eyy_min = -emax * r[3]
    gxy_max = emax * r[4]
    gxy_min = -emax * r[5]

    # Safety factor: stay at 80% of the domain limits to avoid boundary issues
    sf = 0.80

    # Define test waypoints as fractions of the safe domain
    # These exercise multi-axial, non-proportional loading
    path_nodes = np.array([
        [0.00, 0.00, 0.00],                                    # Home
        [exx_max * sf, 0.00, 0.00],                             # Deep tension X
        [exx_max * 0.3, eyy_max * 0.7 * sf, gxy_max * 0.6],    # Multi-axial (X-unloading, Y-tension, +shear)
        [exx_min * 0.5, eyy_max * 0.5 * sf, gxy_min * 0.75],   # Compress-X, tension-Y, -shear
        [exx_max * 0.2, eyy_min * 0.5, gxy_max * 0.25],        # Unloading multi-axial
        [0.00, 0.00, 0.00]                                      # Home
    ])
    
    # Visualization: Connect with straight segments
    full_path = []
    for i in range(len(path_nodes)-1):
        segment = np.linspace(path_nodes[i], path_nodes[i+1], 50)
        full_path.append(segment)
    full_path = np.vstack(full_path)
    
    return path_nodes, full_path

# ============================================================
# 3D MULTI-VIEW VISUALIZATION
# ============================================================

def plot_path_in_domain(path_nodes, full_path, emax, rel6, domain_type, out_file):
    fig = plt.figure(figsize=(20, 7))
    
    exx_p, exx_n, eyy_p, eyy_n, gxy_p, gxy_n = np.array(rel6) * emax
    cx, cy, cz = 0.5*(exx_p-exx_n), 0.5*(eyy_p-eyy_n), 0.5*(gxy_p-gxy_n)
    rx, ry, rz = 0.5*(exx_p+exx_n), 0.5*(eyy_p+eyy_n), 0.5*(gxy_p+gxy_n)
    
    # Scale Gxy by 0.5 for visualization
    p_plot = full_path.copy()
    p_plot[:, 2] *= 0.5
    node_plot = path_nodes.copy()
    node_plot[:, 2] *= 0.5
    
    views = [
        (22, 35, "Perspective View"),
        (90, 0, "Top View (Exx-Eyy)"),
        (0, 0, "Side View (Exx-Gxy/2)")
    ]
    
    from matplotlib.patches import Patch
    domain_label = "Box Domain" if domain_type == "box" else "Ellipsoid Domain"
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='Test Path (unseen)'),
        Patch(facecolor='gray', edgecolor='gray', alpha=0.15, label=domain_label)
    ]

    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        if domain_type == "box":
            _draw_box_3d(ax, cx, cy, cz*0.5, rx, ry, rz*0.5)
        else:
            u = np.linspace(0, 2 * np.pi, 60)
            v = np.linspace(0, np.pi, 30)
            uu, vv = np.meshgrid(u, v)
            xx = cx + rx * np.cos(uu) * np.sin(vv)
            yy = cy + ry * np.sin(uu) * np.sin(vv)
            zz = (cz + rz * np.cos(vv)) * 0.5
            ax.plot_surface(xx, yy, zz, color='gray', alpha=0.10, linewidth=0)
        
        ax.plot(p_plot[:, 0], p_plot[:, 1], p_plot[:, 2], 'r-', linewidth=1.5)
        ax.scatter(node_plot[:, 0], node_plot[:, 1], node_plot[:, 2], color='red', s=20)
        ax.scatter(node_plot[0,0], node_plot[0,1], node_plot[0,2], color='green', s=100, zorder=5)
        
        ax.set_xlabel('Exx'); ax.set_ylabel('Eyy'); ax.set_zlabel('Gxy/2')
        ax.set_title(title); ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((rx if rx > 0 else 1, ry if ry > 0 else 1, rz*0.5 if rz > 0 else 1))
        if i == 0: ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_file, dpi=120)
    print(f"Update: Multi-View Plot saved to: {out_file}")


def _draw_box_3d(ax, cx, cy, cz, rx, ry, rz):
    """Draw a wireframe box in 3D."""
    xs = [cx - rx, cx + rx]
    ys = [cy - ry, cy + ry]
    zs = [cz - rz, cz + rz]
    for i in range(2):
        for j in range(2):
            ax.plot([xs[0], xs[1]], [ys[i], ys[i]], [zs[j], zs[j]], 'k-', lw=0.7, alpha=0.25)
            ax.plot([xs[i], xs[i]], [ys[0], ys[1]], [zs[j], zs[j]], 'k-', lw=0.7, alpha=0.25)
            ax.plot([xs[i], xs[i]], [ys[j], ys[j]], [zs[0], zs[1]], 'k-', lw=0.7, alpha=0.25)


def run_stage4_test(mesh="rve_geometry", run_fom=False, plot_only=False):
    out_dir = "stage_4_test_rve"
    os.makedirs(out_dir, exist_ok=True)
    
    # Defaults (Option 1 fallback)
    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"
    
    bundle_path = "stage_0_trajectory/stage_0_trajectories.npz"
    if os.path.exists(bundle_path):
        print(f"[Info] Loading training parameters from {bundle_path}")
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        # Load emax (new field) or fall back to reference_amplitude (legacy)
        if "emax" in data:
            emax = float(data["emax"])
        else:
            emax = float(data["reference_amplitude"])
            print(f"[Warning] 'emax' not found in bundle, using reference_amplitude={emax}")
        # Load domain_type (new field) or default to box (legacy)
        if "domain_type" in data:
            domain_type = str(data["domain_type"][0])
        else:
            domain_type = "box"
            print(f"[Warning] 'domain_type' not found in bundle, defaulting to box")
    
    print(f"[Info] Domain: {domain_type}, emax={emax}, rel6={rel6}")
    print("--- Stage 4: Multi-View Verification (Test Path) ---")
    control_points, full_path = generate_safe_test_path(emax, rel6, domain_type)
    
    print(f"[Info] Test path waypoints:")
    for i, wp in enumerate(control_points):
        print(f"  wp[{i}]: Exx={wp[0]:+.4f}, Eyy={wp[1]:+.4f}, Gxy={wp[2]:+.4f}")
    
    plot_file = os.path.join(out_dir, "test_path_multi_view.png")
    plot_path_in_domain(control_points, full_path, emax, rel6, domain_type, plot_file)
    
    if plot_only:
        return
    
    # Calculate and report total steps
    seg_steps, _ = BuildDynamicSegmentSteps(
        control_points,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax
    )
    total_steps = int(np.sum(seg_steps))
    print(f"\n[CONFIG] Simulation will take {total_steps} steps.")
    print(f"[CONFIG] Reference Increment Level: {REFERENCE_STEPS_FOR_UNIT_AMPLITUDE}")

    # Kratos Parameters
    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())
    SetInputMeshFilename(parameters, mesh)
    
    mdpa_path = f"{StripMdpaExtension(mesh)}.mdpa"
    material_parts = DetectMaterialSubModelParts(mdpa_path)
    parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
    runtime_materials = WriteRuntimeMaterialsFile(material_parts, 1628.0, 0.4)
    parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(runtime_materials)

    # 1. Run / Load FOM
    fom_sig_file = os.path.join(out_dir, "fom_stress_unseen.npy")
    if run_fom:
        print("\n[Stage 4] Launching FOM Ground Truth...")
        f_eps, f_sig = RunFomBatchSimulation(
            parameters,
            out_dir=out_dir,
            strain_path=control_points,
            trajectory_index=None,
            save_plot=False
        )
        f_eps, f_sig = np.array(f_eps), np.array(f_sig)
        np.save(fom_sig_file, f_sig)
        np.save(os.path.join(out_dir, "fom_strain_unseen.npy"), f_eps)
    else:
        if os.path.exists(fom_sig_file):
            print("\n[Stage 4] Loading existing FOM reference.")
            f_sig = np.load(fom_sig_file)
            f_eps = np.load(os.path.join(out_dir, "fom_strain_unseen.npy"))
        else:
            print("\n[Stage 4] No FOM reference found. Use --run-fom if you want to compare.")
            f_sig, f_eps = None, None

    # 2. Run PROM
    p_sig_file = os.path.join(out_dir, "prom_stress_unseen.npy")
    p_eps_file = os.path.join(out_dir, "prom_strain_unseen.npy")

    if os.path.exists(p_sig_file):
        print("\n[Stage 4] Found existing PROM results. Loading...")
        p_sig = np.load(p_sig_file)
        p_eps = np.load(p_eps_file)
    else:
        print("\n[Stage 4] Launching PROM Prediction...")
        model_dir = "stage_2_pod_rve"
        phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
        free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy"))
        dir_dofs = np.load(os.path.join(model_dir, "dirichlet_dofs.npy"))
        eq_map = np.load(os.path.join(model_dir, "eq_map.npy"))
        Xc, Yc = np.load(os.path.join(model_dir, "domain_center.npy"))

        p_eps, p_sig = RunPromBatchSimulation(
            parameters, phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc,
            out_dir=out_dir,
            strain_path=control_points,
            trajectory_index=None,
            save_plot=False 
        )
        p_eps, p_sig = np.array(p_eps), np.array(p_sig)
        # Save for future one-click viewing
        np.save(p_sig_file, p_sig)
        np.save(p_eps_file, p_eps)

    # 3. Final Comparison (if FOM available)
    if f_sig is not None:
        n = min(len(f_sig), len(p_sig))
        err = np.linalg.norm(f_sig[:n] - p_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-10)
        print(f"\n==========================================")
        print(f"GENERALIZATION ERROR (Unseen Path)")
        print(f"Relative Stress Error: {err:.4e}")
        print(f"==========================================\n")

        # Plotting comparison (Separate LaTeX-style figures)
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
            plt.plot(f_eps[:n, i], f_sig[:n, i], 'k-', label="FOM", linewidth=2)
            plt.plot(p_eps[:n, i], p_sig[:n, i], 'r--', label="PROM", linewidth=1.5)
            plt.title(f"Generalization Comparison: {label_sig}")
            plt.xlabel(f"{label_eps} [-]")
            plt.ylabel(f"{label_sig} [Pa]")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"stage4_comparison_{suffix}.png"), dpi=150)
            plt.close()

        # Error vs Step
        plt.figure(figsize=(7, 6))
        step_err = np.linalg.norm(f_sig[:n] - p_sig[:n], axis=1) / (np.linalg.norm(f_sig[:n], axis=1) + 1e-10)
        plt.plot(step_err, 'b-', label="Rel. Error")
        plt.title("Generalization Stress Error per Step")
        plt.xlabel("Step")
        plt.ylabel("Relative Error [-]")
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"stage4_comparison_error.png"), dpi=150)
        plt.close()

        print(f"Separate generalization comparison plots saved to: {out_dir}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-fom", action="store_true", help="Force run the FOM reference.")
    p.add_argument("--plot-only", action="store_true", help="Only generate the 3D trajectory plot.")
    args = p.parse_args()
    
    run_stage4_test(run_fom=args.run_fom, plot_only=args.plot_only)
