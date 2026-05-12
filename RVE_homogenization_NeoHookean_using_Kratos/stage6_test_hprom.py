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
from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    RunFomBatchSimulation,
    BuildDynamicSegmentSteps,
    setup_kratos_parameters as setup_kratos_parameters_common,
    SetInputMeshFilename,
    StripMdpaExtension,
    DetectMaterialSubModelParts,
    ConfigureElementModelerForMaterialParts,
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
    return setup_kratos_parameters_common(mesh)


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

def run_stage6(
    mesh="rve_geometry",
    ecm_file=os.path.join("stage_5_hprom_data", "ecm_weights_all.npz"),
    use_hrom_mesh=True,
    run_fom=False,
    run_prom=False,
    run_hprom=True,
    hprom_homogenization_method="ecm_weighted",
    fom_homogenization_method="ecm_weighted_full",
    gappy_operator_file=None,
):
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
            emax = float(np.ravel(data["emax"])[0])
        else:
            emax = float(np.ravel(data["reference_amplitude"])[0])
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
    print(f"  HPROM homogenization method: {hprom_homogenization_method}")
    print(f"  FOM homogenization method:   {fom_homogenization_method}")
    if str(hprom_homogenization_method).strip().lower() == "gappy_pod" and gappy_operator_file is None:
        print(
            "[Stage 6][WARN] gappy_pod requested without --gappy-operator-file; "
            "solver will fallback unless operator keys are embedded in ECM npz."
        )

    fom_hom_mode = str(fom_homogenization_method).strip().lower()
    if fom_hom_mode != "ecm_weighted_full":
        raise RuntimeError(
            "Only 'ecm_weighted_full' is supported for FOM homogenization in Stage 6."
        )

    w_fom_eps = None
    w_fom_sig = None
    a0_ref = None
    if fom_hom_mode == "ecm_weighted_full":
        ecm_meta = np.load(ecm_file, allow_pickle=True)
        if "n_elem" not in ecm_meta.files:
            raise RuntimeError("ECM file missing 'n_elem'; required for ecm_weighted_full FOM mode.")
        n_elem_ref = int(np.ravel(ecm_meta["n_elem"])[0])
        if n_elem_ref <= 0:
            raise RuntimeError(f"Invalid n_elem in ECM file: {n_elem_ref}")
        w_fom_eps = np.ones(n_elem_ref, dtype=float)
        w_fom_sig = np.ones(n_elem_ref, dtype=float)
        for k in ("A0_ref", "hom_reference_measure", "A_total"):
            if k in ecm_meta.files:
                val = float(np.ravel(ecm_meta[k])[0])
                if np.isfinite(val) and val > 0.0:
                    a0_ref = val
                    break
        if a0_ref is not None:
            print(f"  Reference area A0: {a0_ref:.6e}")
        if not run_fom:
            print(
                "[Stage 6] Forcing fresh FOM run with ecm_weighted_full homogenization "
                "(cached FOM may use a different operator)."
            )
            run_fom = True

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
            hom_weights_eps_full=w_fom_eps,
            hom_weights_sig_full=w_fom_sig,
            hom_reference_measure=a0_ref,
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
    prom_q_file = os.path.join(out_dir, "prom_run_q.npy")

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
    method_tag = str(hprom_homogenization_method).strip().lower()
    if method_tag == "ecm_weighted":
        hprom_tag = "hprom"
    else:
        hprom_tag = f"hprom_{method_tag}"
    hprom_sig_file = os.path.join(out_dir, f"{hprom_tag}_stress.npy")
    hprom_eps_file = os.path.join(out_dir, f"{hprom_tag}_strain.npy")
    hprom_q_file = os.path.join(out_dir, "hprom_run_q.npy")

    if run_hprom or not os.path.exists(hprom_sig_file):
        if not os.path.exists(hprom_sig_file):
            print(f"\n[Stage 6] No HPROM cache found. Running HPROM...")
        else:
            print(f"\n[Stage 6] --run-hprom: Re-running HPROM...")
        phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc = load_rom_model()
        ecm = np.load(ecm_file, allow_pickle=True)
        ecm_data = {k: ecm[k] for k in ecm.files}
        hprom_mesh = mesh
        if use_hrom_mesh:
            if "hrom_mesh_base" in ecm_data:
                hprom_mesh = str(np.ravel(ecm_data["hrom_mesh_base"])[0])
                print(f"[Stage 6] Using HROM mesh from ECM file: {hprom_mesh}")
            else:
                print("[Stage 6][WARN] --use-hrom-mesh requested but 'hrom_mesh_base' is missing in ECM file.")
                print(f"[Stage 6][WARN] Falling back to mesh: {mesh}")
        parameters_hprom = setup_kratos_parameters(hprom_mesh)
        gappy_data = None
        if gappy_operator_file is not None:
            gappy_npz = np.load(gappy_operator_file, allow_pickle=True)
            gappy_data = {k: gappy_npz[k] for k in gappy_npz.files}
            if "hom_gappy_train_rel_error_total" in gappy_data:
                print(
                    "[Stage 6] Gappy operator offline train errors: "
                    f"total={float(np.ravel(gappy_data['hom_gappy_train_rel_error_total'])[0]):.3e}, "
                    f"eps={float(np.ravel(gappy_data['hom_gappy_train_rel_error_eps'])[0]):.3e}, "
                    f"sig={float(np.ravel(gappy_data['hom_gappy_train_rel_error_sig'])[0]):.3e}"
                )

        t0 = time.perf_counter()
        h_eps, h_sig = RunHpromBatchSimulation(
            parameters_hprom, phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc,
            ecm_data=ecm_data,
            out_dir=out_dir, strain_path=waypoints,
            trajectory_index=None, save_plot=False,
            reference_amplitude=emax,
            homogenization_method=hprom_homogenization_method,
            homogenization_gappy_data=gappy_data,
        )
        timings["HPROM"] = time.perf_counter() - t0
        np.save(hprom_sig_file, h_sig)
        np.save(hprom_eps_file, h_eps)
        print(
            f"[Stage 6] HPROM ({hprom_homogenization_method}) done: "
            f"{h_sig.shape[0]} entries in {timings['HPROM']:.1f}s"
        )
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

    q_err = None
    q_err_mean = None
    q_err_max = None
    if os.path.exists(prom_q_file) and os.path.exists(hprom_q_file):
        q_prom = np.load(prom_q_file)
        q_hprom = np.load(hprom_q_file)
        nq = min(len(q_prom), len(q_hprom))
        if nq > 0:
            q_err = np.linalg.norm(q_hprom[:nq] - q_prom[:nq]) / (np.linalg.norm(q_prom[:nq]) + 1e-30)
            q_step_den = np.linalg.norm(q_prom[:nq], axis=1)
            q_step_num = np.linalg.norm(q_hprom[:nq] - q_prom[:nq], axis=1)
            valid = q_step_den > 1e-12
            if np.any(valid):
                q_step_err = q_step_num[valid] / q_step_den[valid]
                q_err_mean = float(np.mean(q_step_err))
                q_err_max = float(np.max(q_step_err))
            else:
                q_err_mean = 0.0
                q_err_max = 0.0

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  PROM  vs FOM:  Rel. Stress Error = {err_prom:.4e}")
    print(f"  HPROM vs FOM:  Rel. Stress Error = {err_hprom:.4e}")
    if q_err is not None:
        print(f"  HPROM vs PROM: Rel. q Error      = {q_err:.4e}")
        print(f"  q stepwise: mean={q_err_mean:.4e}, max={q_err_max:.4e}")
    else:
        print("  q error: unavailable (prom_run_q.npy or hprom_run_q.npy missing)")
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
    p.add_argument(
        "--ecm-file",
        type=str,
        default=os.path.join("stage_5_hprom_data", "ecm_weights_all.npz"),
        help="ECM npz file used by HPROM.",
    )
    p.add_argument(
        "--use-hrom-mesh",
        action="store_true",
        help="Use HROM mesh stored in ECM file (hrom_mesh_base) for HPROM run (default: enabled).",
    )
    p.add_argument(
        "--no-use-hrom-mesh",
        dest="use_hrom_mesh",
        action="store_false",
        help="Disable automatic HROM mesh usage and force the base mesh.",
    )
    p.add_argument(
        "--hprom-homogenization-method",
        type=str,
        default="ecm_weighted",
        choices=["ecm_weighted", "gappy_pod", "kratos_reference"],
        help="HPROM homogenization output method.",
    )
    p.add_argument(
        "--fom-homogenization-method",
        type=str,
        default="ecm_weighted_full",
        choices=["ecm_weighted_full"],
        help="FOM homogenization operator used for comparison baseline.",
    )
    p.add_argument(
        "--gappy-operator-file",
        type=str,
        default=None,
        help="Optional Stage5c gappy operator npz (used with --hprom-homogenization-method gappy_pod).",
    )
    p.set_defaults(use_hrom_mesh=True)
    args = p.parse_args()

    run_stage6(
        ecm_file=args.ecm_file,
        use_hrom_mesh=args.use_hrom_mesh,
        run_fom=args.run_fom,
        run_prom=args.run_prom,
        run_hprom=not args.skip_hprom,
        hprom_homogenization_method=args.hprom_homogenization_method,
        fom_homogenization_method=args.fom_homogenization_method,
        gappy_operator_file=args.gappy_operator_file,
    )
