#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 10: FOM vs PROM-RBF vs HPROM-RBF benchmark.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from stage6_test_hprom import generate_safe_test_path
from stage4_test_rve import plot_path_in_domain
from fom_solver_rve import (
    setup_kratos_parameters,
    RunFomBatchSimulation,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from prom_rbf_solver_rve import LoadPromRbfModel, RunPromRbfBatchSimulation
from hprom_rbf_solver_rve import LoadHpromRbfModel, RunHpromRbfBatchSimulation


def plot_hprom_rbf_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings=None):
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
        plt.plot(p_eps[:n, i], p_sig[:n, i], "r--", label="PROM-RBF", linewidth=1.5)
        plt.plot(h_eps[:n, i], h_sig[:n, i], "b:", label="HPROM-RBF", linewidth=1.5)
        plt.title(f"HPROM-RBF Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hprom_rbf_comp_{suffix}.png"), dpi=150)
        plt.close()

    fom_norm = np.linalg.norm(f_sig[:n], axis=1) + 1e-30
    err_prom_rbf = np.linalg.norm(f_sig[:n] - p_sig[:n], axis=1) / fom_norm
    err_hprom_rbf = np.linalg.norm(f_sig[:n] - h_sig[:n], axis=1) / fom_norm

    plt.figure(figsize=(7, 6))
    plt.plot(err_prom_rbf, "r-", label="PROM-RBF Error", linewidth=1.5)
    plt.plot(err_hprom_rbf, "b-", label="HPROM-RBF Error", linewidth=1.5)
    plt.title("Relative Stress Error Comparison")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_rbf_comp_error.png"), dpi=150)
    plt.close()

    if timings:
        plt.figure(figsize=(7, 6))
        methods = list(timings.keys())
        values = [timings[m] for m in methods]
        colors = {"FOM": "gray", "PROM-RBF": "red", "HPROM-RBF": "blue"}
        bars = plt.bar(methods, values, color=[colors.get(m, "green") for m in methods], alpha=0.8)
        for bar, t in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{t:.1f}s", ha="center", va="bottom")
        plt.title("Computational Performance")
        plt.ylabel("Wall-Clock Time [s]")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hprom_rbf_performance.png"), dpi=150)
        plt.close()

    print(f"Stage 10 plots saved to: {out_dir}")


def run_stage10(run_fom=False, run_prom_rbf=False, run_hprom_rbf=False):
    out_dir = "stage_10_hprom_rbf_results"
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
            emax = float(data["emax"])
        else:
            emax = float(data["reference_amplitude"])
        if "domain_type" in data:
            domain_type = str(data["domain_type"][0])

    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)

    # Visualization of the test path in the current domain
    _plot_file = os.path.join(out_dir, "hprom_rbf_test_path.png")
    plot_path_in_domain(control_points, waypoints, emax, rel6, domain_type, _plot_file)

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 60)
    print("  Stage 10: HPROM-RBF Benchmark (FOM vs PROM-RBF vs HPROM-RBF)")
    print("=" * 60)
    print(f"  Waypoints: {len(control_points)}")
    print(f"  Strain path points: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  Segments: {seg_steps}")

    parameters = setup_kratos_parameters("rve_geometry")
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

    prom_eps_file = os.path.join(out_dir, "prom_rbf_strain.npy")
    prom_sig_file = os.path.join(out_dir, "prom_rbf_stress.npy")
    if run_prom_rbf or not (os.path.exists(prom_eps_file) and os.path.exists(prom_sig_file)):
        print("\n[Stage 10] Running PROM-RBF...")
        phi_p, phi_s, free_dofs, _, _, rbf_model, include_macro = LoadPromRbfModel(
            basis_dir="stage_2_pod_rve", rbf_data_dir="stage_7_rbf_data"
        )
        t0 = time.perf_counter()
        p_eps, p_sig = RunPromRbfBatchSimulation(
            parameters,
            phi_p,
            phi_s,
            free_dofs,
            rbf_model,
            strain_path,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            include_macro_strain_input=include_macro,
        )
        timings["PROM-RBF"] = time.perf_counter() - t0
        p_eps = np.asarray(p_eps, dtype=float)
        p_sig = np.asarray(p_sig, dtype=float)
        np.save(prom_eps_file, p_eps)
        np.save(prom_sig_file, p_sig)
    else:
        print("\n[Stage 10] Loading cached PROM-RBF.")
        p_eps = np.load(prom_eps_file)
        p_sig = np.load(prom_sig_file)

    hprom_eps_file = os.path.join(out_dir, "hprom_rbf_strain.npy")
    hprom_sig_file = os.path.join(out_dir, "hprom_rbf_stress.npy")
    if run_hprom_rbf or not (os.path.exists(hprom_eps_file) and os.path.exists(hprom_sig_file)):
        print("\n[Stage 10] Running HPROM-RBF...")
        (
            phi_p_h,
            phi_s_h,
            free_dofs_h,
            _,
            _,
            _,
            _,
            rbf_model_h,
            ecm_data_h,
            include_macro_h,
        ) = LoadHpromRbfModel(
            basis_dir="stage_2_pod_rve",
            rbf_data_dir="stage_7_rbf_data",
            hprom_rbf_dir="stage_9_hprom_rbf_data",
        )
        t0 = time.perf_counter()
        h_eps, h_sig = RunHpromRbfBatchSimulation(
            parameters,
            phi_p_h,
            phi_s_h,
            free_dofs_h,
            rbf_model_h,
            ecm_data_h,
            out_dir=out_dir,
            strain_path=strain_path,
            trajectory_index=None,
            include_macro_strain_input=include_macro_h,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        )
        timings["HPROM-RBF"] = time.perf_counter() - t0
        h_eps = np.asarray(h_eps, dtype=float)
        h_sig = np.asarray(h_sig, dtype=float)
        np.save(hprom_eps_file, h_eps)
        np.save(hprom_sig_file, h_sig)
    else:
        print("\n[Stage 10] Loading cached HPROM-RBF.")
        h_eps = np.load(hprom_eps_file)
        h_sig = np.load(hprom_sig_file)

    n = min(len(f_sig), len(p_sig), len(h_sig))
    err_prom = np.linalg.norm(f_sig[:n] - p_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-30)
    err_hprom = np.linalg.norm(f_sig[:n] - h_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-30)

    print("\n" + "=" * 60)
    print("  Stage 10 Summary")
    print("=" * 60)
    print(f"  PROM-RBF  vs FOM: Rel. Stress Error = {err_prom:.4e}")
    print(f"  HPROM-RBF vs FOM: Rel. Stress Error = {err_hprom:.4e}")
    for method, t in timings.items():
        print(f"  {method} time: {t:.2f}s")

    plot_hprom_rbf_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 10: HPROM-RBF benchmark")
    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument("--run-prom-rbf", action="store_true", help="Force PROM-RBF recompute.")
    p.add_argument("--run-hprom-rbf", action="store_true", help="Force HPROM-RBF recompute.")
    args = p.parse_args()

    run_stage10(
        run_fom=args.run_fom,
        run_prom_rbf=args.run_prom_rbf,
        run_hprom_rbf=args.run_hprom_rbf,
    )
