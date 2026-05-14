#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 10: FOM vs PROM-GPR vs HPROM-GPR benchmark.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()

from stage6_test_hprom import generate_safe_test_path
from stage4_test_rve import plot_path_in_domain
from fom_solver_rve import (
    setup_kratos_parameters,
    RunFomBatchSimulation,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from prom_gpr_solver_rve import LoadPromGprModel, RunPromGprBatchSimulation
from hprom_gpr_solver_rve import LoadHpromGprModel, RunHpromGprBatchSimulation


def _load_hrom_mesh_base_from_ecm_dir(hprom_gpr_dir):
    ecm_file = os.path.join(str(hprom_gpr_dir), "ecm_weights_all.npz")
    if not os.path.exists(ecm_file):
        return None
    try:
        ecm = np.load(ecm_file, allow_pickle=True)
    except Exception:
        return None
    if "hrom_mesh_base" not in ecm:
        return None
    try:
        return str(np.ravel(ecm["hrom_mesh_base"])[0])
    except Exception:
        return None


def _compute_equivalent_stress_strain(eps, sig):
    eps = np.asarray(eps, dtype=float)
    sig = np.asarray(sig, dtype=float)
    if eps.ndim != 2 or sig.ndim != 2 or eps.shape[1] < 3 or sig.shape[1] < 3:
        raise ValueError("Expected eps and sig arrays with shape [n_steps, >=3].")

    exx = eps[:, 0]
    eyy = eps[:, 1]
    gxy = eps[:, 2]  # engineering shear strain
    sxx = sig[:, 0]
    syy = sig[:, 1]
    sxy = sig[:, 2]

    sigma_eq = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
    eps_eq = (2.0 / 3.0) * np.sqrt(np.maximum(exx * exx + eyy * eyy - exx * eyy + 0.75 * gxy * gxy, 0.0))
    return eps_eq, sigma_eq


def plot_hprom_gpr_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings=None):
    n = min(len(f_sig), len(p_sig), len(h_sig), len(f_eps), len(p_eps), len(h_eps))

    plt.rcParams.update(
        {
            "font.size": 12,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    comp_labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "sigma_xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "sigma_yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "sigma_xy"),
    ]

    for i, label_sig, label_eps, suffix in comp_labels:
        plt.figure(figsize=(7, 6))
        plt.plot(f_eps[:n, i], f_sig[:n, i], "k-", label="FOM", linewidth=2.0)
        plt.plot(p_eps[:n, i], p_sig[:n, i], "r--", label="PROM-GPR", linewidth=1.5)
        plt.plot(h_eps[:n, i], h_sig[:n, i], "b:", label="HPROM-GPR", linewidth=1.5)
        plt.title(f"HPROM-GPR Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hprom_gpr_comp_{suffix}.png"), dpi=150)
        plt.close()

    f_eps_eq, f_sig_eq = _compute_equivalent_stress_strain(f_eps[:n], f_sig[:n])
    p_eps_eq, p_sig_eq = _compute_equivalent_stress_strain(p_eps[:n], p_sig[:n])
    h_eps_eq, h_sig_eq = _compute_equivalent_stress_strain(h_eps[:n], h_sig[:n])

    plt.figure(figsize=(7, 6))
    plt.plot(f_eps_eq, f_sig_eq, "k-", label="FOM", linewidth=2.0)
    plt.plot(p_eps_eq, p_sig_eq, "r--", label="PROM-GPR", linewidth=1.5)
    plt.plot(h_eps_eq, h_sig_eq, "b:", label="HPROM-GPR", linewidth=1.5)
    plt.title(r"HPROM-GPR Benchmark: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_gpr_comp_sigma_eq.png"), dpi=150)
    plt.close()

    fom_norm = np.linalg.norm(f_sig[:n], axis=1) + 1e-30
    err_prom_rbf = np.linalg.norm(f_sig[:n] - p_sig[:n], axis=1) / fom_norm
    err_hprom_rbf = np.linalg.norm(f_sig[:n] - h_sig[:n], axis=1) / fom_norm

    plt.figure(figsize=(7, 6))
    plt.plot(err_prom_rbf, "r-", label="PROM-GPR Error", linewidth=1.5)
    plt.plot(err_hprom_rbf, "b-", label="HPROM-GPR Error", linewidth=1.5)
    plt.title("Relative Stress Error Comparison")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_gpr_comp_error.png"), dpi=150)
    plt.close()

    if timings:
        plt.figure(figsize=(7, 6))
        methods = list(timings.keys())
        values = [timings[m] for m in methods]
        colors = {"FOM": "gray", "PROM-GPR": "red", "HPROM-GPR": "blue"}
        bars = plt.bar(methods, values, color=[colors.get(m, "green") for m in methods], alpha=0.8)
        for bar, t in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{t:.1f}s", ha="center", va="bottom")
        plt.title("Computational Performance")
        plt.ylabel("Wall-Clock Time [s]")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hprom_gpr_performance.png"), dpi=150)
        plt.close()

    print(f"Stage 10 plots saved to: {out_dir}")


def run_stage10(
    run_fom=False,
    run_prom_gpr=False,
    run_hprom_gpr=False,
    gpr_data_dir="stage_7_gpr_data",
    hprom_gpr_dir="stage_9_hprom_gpr_data",
    out_dir="stage_10_hprom_gpr_results",
):
    os.makedirs(out_dir, exist_ok=True)

    # Load domain parameters from stage0 bundle
    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"

    bundle_path = "stage_0_trajectory/stage_0_trajectories.npz"
    if os.path.exists(bundle_path):
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        if "emax" in data:
            emax = float(np.ravel(data["emax"])[0])
        else:
            emax = float(np.ravel(data["reference_amplitude"])[0])
        if "domain_type" in data:
            domain_type = str(data["domain_type"][0])

    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)

    # Visualization of the test path in the current domain
    _plot_file = os.path.join(out_dir, "hprom_gpr_test_path.png")
    plot_path_in_domain(control_points, waypoints, emax, rel6, domain_type, _plot_file)

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 60)
    print("  Stage 10: HPROM-GPR Benchmark (FOM vs PROM-GPR vs HPROM-GPR)")
    print("=" * 60)
    print(f"  Waypoints: {len(control_points)}")
    print(f"  Strain path points: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  Segments: {seg_steps}")
    print(f"  GPR data dir: {gpr_data_dir}")
    print(f"  HPROM-GPR dir: {hprom_gpr_dir}")
    print(f"  Output dir: {out_dir}")

    mesh_fom_prom = "rve_geometry"
    mesh_hprom = "rve_geometry"
    auto_hrom_mesh = _load_hrom_mesh_base_from_ecm_dir(hprom_gpr_dir)
    if auto_hrom_mesh:
        mesh_hprom = str(auto_hrom_mesh)
    print(f"  FOM/PROM mesh: {mesh_fom_prom}")
    print(f"  HPROM mesh:    {mesh_hprom}")

    parameters = setup_kratos_parameters(mesh_fom_prom)
    parameters_hprom = setup_kratos_parameters(mesh_hprom)
    timings = {}

    fom_eps_file = os.path.join(out_dir, "fom_strain.npy")
    fom_sig_file = os.path.join(out_dir, "fom_stress.npy")
    if run_fom or not (os.path.exists(fom_eps_file) and os.path.exists(fom_sig_file)):
        print("\n[Stage 10] Running FOM...")
        t0 = time.perf_counter()
        f_eps, f_sig = RunFomBatchSimulation(
            parameters,
            out_dir=out_dir,
            strain_path=strain_path,
            trajectory_index=None,
            save_plot=False,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        )
        timings["FOM"] = time.perf_counter() - t0
        f_eps = np.asarray(f_eps, dtype=float)
        f_sig = np.asarray(f_sig, dtype=float)
        np.save(fom_eps_file, f_eps)
        np.save(fom_sig_file, f_sig)
    else:
        print("\n[Stage 10] Loading cached FOM.")
        f_eps = np.load(fom_eps_file)
        f_sig = np.load(fom_sig_file)

    prom_eps_file = os.path.join(out_dir, "prom_gpr_strain.npy")
    prom_sig_file = os.path.join(out_dir, "prom_gpr_stress.npy")
    if run_prom_gpr or not (os.path.exists(prom_eps_file) and os.path.exists(prom_sig_file)):
        print("\n[Stage 10] Running PROM-GPR...")
        phi_p, phi_s, free_dofs, _, _, gpr_model, _include_macro = LoadPromGprModel(
            basis_dir="stage_2_pod_rve", gpr_data_dir=gpr_data_dir
        )
        t0 = time.perf_counter()
        p_eps, p_sig = RunPromGprBatchSimulation(
            parameters,
            phi_p,
            phi_s,
            free_dofs,
            gpr_model,
            strain_path,
            out_dir=out_dir,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        )
        timings["PROM-GPR"] = time.perf_counter() - t0
        p_eps = np.asarray(p_eps, dtype=float)
        p_sig = np.asarray(p_sig, dtype=float)
        np.save(prom_eps_file, p_eps)
        np.save(prom_sig_file, p_sig)
    else:
        print("\n[Stage 10] Loading cached PROM-GPR.")
        p_eps = np.load(prom_eps_file)
        p_sig = np.load(prom_sig_file)

    hprom_eps_file = os.path.join(out_dir, "hprom_gpr_strain.npy")
    hprom_sig_file = os.path.join(out_dir, "hprom_gpr_stress.npy")
    if run_hprom_gpr or not (os.path.exists(hprom_eps_file) and os.path.exists(hprom_sig_file)):
        print("\n[Stage 10] Running HPROM-GPR...")
        (
            phi_p_h,
            phi_s_h,
            free_dofs_h,
            _dir_dofs_h,
            eq_map_h,
            Xc_h,
            Yc_h,
            gpr_model_h,
            ecm_data_h,
            _include_macro_h,
        ) = LoadHpromGprModel(
            basis_dir="stage_2_pod_rve",
            gpr_data_dir=gpr_data_dir,
            hprom_gpr_dir=hprom_gpr_dir,
        )
        t0 = time.perf_counter()
        h_eps, h_sig = RunHpromGprBatchSimulation(
            parameters_hprom,
            phi_p_h,
            phi_s_h,
            free_dofs_h,
            gpr_model_h,
            ecm_data_h,
            out_dir=out_dir,
            strain_path=strain_path,
            trajectory_index=None,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            eq_map_full=eq_map_h,
            Xc=Xc_h,
            Yc=Yc_h,
        )
        timings["HPROM-GPR"] = time.perf_counter() - t0
        h_eps = np.asarray(h_eps, dtype=float)
        h_sig = np.asarray(h_sig, dtype=float)
        np.save(hprom_eps_file, h_eps)
        np.save(hprom_sig_file, h_sig)
    else:
        print("\n[Stage 10] Loading cached HPROM-GPR.")
        h_eps = np.load(hprom_eps_file)
        h_sig = np.load(hprom_sig_file)

    n = min(len(f_sig), len(p_sig), len(h_sig))
    err_prom = np.linalg.norm(f_sig[:n] - p_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-30)
    err_hprom = np.linalg.norm(f_sig[:n] - h_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-30)

    print("\n" + "=" * 60)
    print("  Stage 10 Summary")
    print("=" * 60)
    print(f"  PROM-GPR  vs FOM: Rel. Stress Error = {err_prom:.4e}")
    print(f"  HPROM-GPR vs FOM: Rel. Stress Error = {err_hprom:.4e}")
    for method, t in timings.items():
        print(f"  {method} time: {t:.2f}s")

    plot_hprom_gpr_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 10: HPROM-GPR benchmark")
    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument("--run-prom-gpr", action="store_true", help="Force PROM-GPR recompute.")
    p.add_argument("--run-hprom-gpr", action="store_true", help="Force HPROM-GPR recompute.")
    p.add_argument(
        "--gpr-data-dir",
        type=str,
        default="stage_7_gpr_data",
        help="Directory with sparse-GP model files (sparse_gp_model.npz, phi_p.npy, phi_s.npy).",
    )
    p.add_argument(
        "--hprom-gpr-dir",
        type=str,
        default="stage_9_hprom_gpr_data",
        help="Directory with HPROM-GPR ECM file (ecm_weights_all.npz).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="stage_10_hprom_gpr_results",
        help="Output directory for Stage 10 results.",
    )
    args = p.parse_args()

    run_stage10(
        run_fom=args.run_fom,
        run_prom_gpr=args.run_prom_gpr,
        run_hprom_gpr=args.run_hprom_gpr,
        gpr_data_dir=args.gpr_data_dir,
        hprom_gpr_dir=args.hprom_gpr_dir,
        out_dir=args.out_dir,
    )
