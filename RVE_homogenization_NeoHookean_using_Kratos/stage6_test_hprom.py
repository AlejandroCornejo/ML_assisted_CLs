#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 6: HPROM Verification
Tests the Hyper-Reduced PROM against FOM and PROM on the same unseen trajectory.
All three methods run with identical parameters to ensure consistent step counts.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

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
    MIN_STEPS_PER_SEGMENT,
)
from prom_solver_rve import RunPromBatchSimulation
from hprom_solver_rve import RunHpromBatchSimulation

# ============================================================
# STAGE 6 BENCHMARK TRAJECTORY
# ============================================================

def generate_safe_test_path(emax, rel6, domain_type="box"):
    """
    Stage 6 benchmark trajectory (self-contained).
    Waypoints computed as fractions of the actual domain limits.
    """
    r = np.array(rel6, dtype=float)
    exx_max = emax * r[0]
    exx_min = -emax * r[1]
    eyy_max = emax * r[2]
    eyy_min = -emax * r[3]
    gxy_max = emax * r[4]
    gxy_min = -emax * r[5]

    # Safety factor: stay at 80% of the domain limits
    sf = 0.80

    control_points = np.array([
        [0.00, 0.00, 0.00],                                    # Home
        [exx_max * sf, 0.00, 0.00],                             # Deep tension X
        [exx_max * 0.3, eyy_max * 0.7 * sf, gxy_max * 0.6],    # Multi-axial (+X, +Y, +shear)
        [exx_min * 0.5, eyy_max * 0.5 * sf, gxy_min * 0.75],   # Compress-X, tension-Y, -shear
        [exx_max * 0.2, eyy_min * 0.5, gxy_max * 0.25],        # Unloading multi-axial
        [0.00, 0.00, 0.00]                                      # Home
    ], dtype=float)

    full_path = [control_points[0]]
    for i in range(len(control_points)-1):
        p0, p1 = control_points[i], control_points[i+1]
        n_interp = 50
        for alpha in np.linspace(0, 1, n_interp+1)[1:]:
            full_path.append(p0 + alpha*(p1 - p0))
            
    return control_points, np.array(full_path)


# ============================================================
# SETUP (shared across all 3 methods)
# ============================================================

def setup_kratos_parameters(mesh="rve_geometry"):
    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())
    SetInputMeshFilename(parameters, mesh)
    mdpa_path = f"{StripMdpaExtension(mesh)}.mdpa"
    material_parts = DetectMaterialSubModelParts(mdpa_path)
    parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
    runtime_materials = WriteRuntimeMaterialsFile(material_parts, 1628.0, 0.4)
    parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(runtime_materials)
    return parameters


def load_rom_model(model_dir="stage_2_pod_rve"):
    phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(model_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(model_dir, "eq_map.npy"))
    Xc, Yc = np.load(os.path.join(model_dir, "domain_center.npy"))
    return phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc


# ============================================================
# PLOTTING (Separate LaTeX-style figures)
# ============================================================

def plot_triple_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings=None):
    """Refined plotting suite: Separate FOM vs PROM vs HPROM LaTeX figures."""
    n = min(len(f_sig), len(p_sig), len(h_sig))
    
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
        plt.plot(f_eps[:n, i], f_sig[:n, i], 'k-', label="FOM", linewidth=2.0)
        plt.plot(p_eps[:n, i], p_sig[:n, i], 'r--', label="PROM", linewidth=1.5)
        plt.plot(h_eps[:n, i], h_sig[:n, i], 'b:', label="HPROM", linewidth=1.5)
        plt.title(f"HPROM Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hprom_comp_{suffix}.png"), dpi=150)
        plt.close()

    # Triple-Error history
    plt.figure(figsize=(7, 6))
    fom_norm = np.linalg.norm(f_sig[:n], axis=1) + 1e-30
    err_prom = np.linalg.norm(f_sig[:n] - p_sig[:n], axis=1) / fom_norm
    err_hprom = np.linalg.norm(f_sig[:n] - h_sig[:n], axis=1) / fom_norm
    
    plt.plot(err_prom, 'r-', label="PROM Error", linewidth=1.5)
    plt.plot(err_hprom, 'b-', label="HPROM Error", linewidth=1.5)
    plt.title("Relative Stress Error Comparison")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_comp_error.png"), dpi=150)
    plt.close()

    # Timing Bar Chart
    if timings:
        plt.figure(figsize=(7, 6))
        methods = list(timings.keys())
        run_times = [timings[m] for m in methods]
        colors = {'FOM': 'gray', 'PROM': 'red', 'HPROM': 'blue'}
        bars = plt.bar(methods, run_times, color=[colors.get(m, 'green') for m in methods], alpha=0.8)
        for bar, t in zip(bars, run_times):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{t:.1f}s', ha='center', va='bottom', fontsize=11)
        plt.title("Computational Performance")
        plt.ylabel("Wall-Clock Time [s]")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hprom_performance.png"), dpi=150)
        plt.close()

    print(f"Separate HPROM benchmark plots saved to: {out_dir}")


# ============================================================
# MAIN
# ============================================================

def run_stage6(mesh="rve_geometry", run_fom=False, run_prom=False, run_hprom=True):
    out_dir = "stage_6_hprom_results"
    os.makedirs(out_dir, exist_ok=True)

    # Load domain parameters from stage0 bundle
    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"

    bundle_path = "stage_0_trajectory/stage_0_trajectories.npz"
    if os.path.exists(bundle_path):
        print(f"[Stage 6] Loading training parameters from {bundle_path}")
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        if "emax" in data:
            emax = float(data["emax"])
        else:
            emax = float(data["reference_amplitude"])
            print(f"[Warning] 'emax' not found in bundle, using reference_amplitude={emax}")
        if "domain_type" in data:
            domain_type = str(data["domain_type"][0])
        else:
            domain_type = "box"

    print(f"[Stage 6] Domain: {domain_type}, emax={emax}, rel6={rel6}")

    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)

    # Step count
    seg_steps, _ = BuildDynamicSegmentSteps(
        control_points,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 60)
    print("  Stage 6: HPROM Verification (FOM vs PROM vs HPROM)")
    print("=" * 60)
    print(f"  Trajectory: Segmented Safe Path, {len(control_points)} waypoints")
    print(f"  Total steps: {total_steps}  (+1 initial = {total_steps+1} entries)")
    print(f"  Segments: {seg_steps}")

    parameters = setup_kratos_parameters(mesh)
    timings = {}

    # -----------------------------------------------------------
    # 1. FOM
    # -----------------------------------------------------------
    fom_sig_file = os.path.join(out_dir, "fom_stress.npy")
    fom_eps_file = os.path.join(out_dir, "fom_strain.npy")

    if run_fom or not os.path.exists(fom_sig_file):
        if not os.path.exists(fom_sig_file):
            print(f"\n[Stage 6] No FOM cache found. Running FOM...")
        else:
            print(f"\n[Stage 6] --run-fom: Re-running FOM...")
        t0 = time.perf_counter()
        f_eps, f_sig = RunFomBatchSimulation(
            parameters, out_dir=out_dir, strain_path=waypoints,
            trajectory_index=None, save_plot=False,
            reference_amplitude=emax,
        )
        timings["FOM"] = time.perf_counter() - t0
        f_eps = np.array(f_eps) if not isinstance(f_eps, np.ndarray) else f_eps
        f_sig = np.array(f_sig) if not isinstance(f_sig, np.ndarray) else f_sig
        np.save(fom_sig_file, f_sig)
        np.save(fom_eps_file, f_eps)
        print(f"[Stage 6] FOM done: {f_sig.shape[0]} entries in {timings['FOM']:.1f}s")
    else:
        print(f"\n[Stage 6] Loading cached FOM results.")
        f_sig = np.load(fom_sig_file)
        f_eps = np.load(fom_eps_file)
        print(f"[Stage 6] FOM: {f_sig.shape[0]} entries")

    # -----------------------------------------------------------
    # 2. PROM
    # -----------------------------------------------------------
    prom_sig_file = os.path.join(out_dir, "prom_stress.npy")
    prom_eps_file = os.path.join(out_dir, "prom_strain.npy")

    if run_prom or not os.path.exists(prom_sig_file):
        if not os.path.exists(prom_sig_file):
            print(f"\n[Stage 6] No PROM cache found. Running PROM...")
        else:
            print(f"\n[Stage 6] --run-prom: Re-running PROM...")
        phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc = load_rom_model()
        t0 = time.perf_counter()
        p_eps, p_sig = RunPromBatchSimulation(
            parameters, phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc,
            out_dir=out_dir, strain_path=waypoints,
            trajectory_index=None, save_plot=False,
            reference_amplitude=emax,
        )
        timings["PROM"] = time.perf_counter() - t0
        p_eps = np.array(p_eps) if not isinstance(p_eps, np.ndarray) else p_eps
        p_sig = np.array(p_sig) if not isinstance(p_sig, np.ndarray) else p_sig
        np.save(prom_sig_file, p_sig)
        np.save(prom_eps_file, p_eps)
        print(f"[Stage 6] PROM done: {p_sig.shape[0]} entries in {timings['PROM']:.1f}s")
    else:
        print(f"\n[Stage 6] Loading cached PROM results.")
        p_sig = np.load(prom_sig_file)
        p_eps = np.load(prom_eps_file)
        print(f"[Stage 6] PROM: {p_sig.shape[0]} entries")

    # -----------------------------------------------------------
    # 3. HPROM
    # -----------------------------------------------------------
    hprom_sig_file = os.path.join(out_dir, "hprom_stress.npy")
    hprom_eps_file = os.path.join(out_dir, "hprom_strain.npy")

    if run_hprom or not os.path.exists(hprom_sig_file):
        if not os.path.exists(hprom_sig_file):
            print(f"\n[Stage 6] No HPROM cache found. Running HPROM...")
        else:
            print(f"\n[Stage 6] --run-hprom: Re-running HPROM...")
        phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc = load_rom_model()
        ecm = np.load(os.path.join("stage_5_hprom_data", "ecm_weights_all.npz"))
        ecm_data = {k: ecm[k] for k in ecm.files}

        t0 = time.perf_counter()
        h_eps, h_sig = RunHpromBatchSimulation(
            parameters, phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc,
            ecm_data=ecm_data,
            out_dir=out_dir, strain_path=waypoints,
            trajectory_index=None, save_plot=False,
            reference_amplitude=emax,
        )
        timings["HPROM"] = time.perf_counter() - t0
        np.save(hprom_sig_file, h_sig)
        np.save(hprom_eps_file, h_eps)
        print(f"[Stage 6] HPROM done: {h_sig.shape[0]} entries in {timings['HPROM']:.1f}s")
    else:
        print(f"\n[Stage 6] Loading cached HPROM results.")
        h_sig = np.load(hprom_sig_file)
        h_eps = np.load(hprom_eps_file)
        print(f"[Stage 6] HPROM: {h_sig.shape[0]} entries")

    # -----------------------------------------------------------
    # 4. Consistency check
    # -----------------------------------------------------------
    print(f"\n[Shapes] FOM={f_sig.shape}, PROM={p_sig.shape}, HPROM={h_sig.shape}")
    if f_sig.shape[0] != p_sig.shape[0] or f_sig.shape[0] != h_sig.shape[0]:
        print("[WARNING] Array lengths differ! Results may have been cached with different parameters.")
        print("          Run with --run-fom --run-prom to regenerate all consistently.")

    # -----------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------
    n = min(len(f_sig), len(p_sig), len(h_sig))

    err_prom = np.linalg.norm(f_sig[:n] - p_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-30)
    err_hprom = np.linalg.norm(f_sig[:n] - h_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-30)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  PROM  vs FOM:  Rel. Stress Error = {err_prom:.4e}")
    print(f"  HPROM vs FOM:  Rel. Stress Error = {err_hprom:.4e}")
    for method, t in timings.items():
        print(f"  {method} time: {t:.2f}s")

    # -----------------------------------------------------------
    # 6. Plot
    # -----------------------------------------------------------
    plot_triple_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 6: HPROM Verification")
    p.add_argument("--run-fom", action="store_true", help="Run FOM fresh")
    p.add_argument("--run-prom", action="store_true", help="Run PROM fresh")
    p.add_argument("--skip-hprom", action="store_true", help="Skip HPROM (use cache)")
    args = p.parse_args()

    run_stage6(run_fom=args.run_fom, run_prom=args.run_prom, run_hprom=not args.skip_hprom)
