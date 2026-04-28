#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone benchmark: FOM vs KRATOS on the same Stage-8/10/11 test path.

- Uses the same trajectory logic as Stage 8/10:
  - path from stage_0_trajectory bundle via generate_safe_test_path
  - dynamic increment scaling with reference_amplitude = emax
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
from kratos_solver_rve import RunKratosBatchSimulation


def plot_fom_vs_kratos(f_eps, f_sig, k_eps, k_sig, out_dir, timings=None):
    n = min(len(f_sig), len(k_sig), len(f_eps), len(k_eps))

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
        plt.plot(k_eps[:n, i], k_sig[:n, i], "c--", label="KRATOS", linewidth=1.8)
        plt.title(f"FOM vs KRATOS: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fom_vs_kratos_{suffix}.png"), dpi=150)
        plt.close()

    ref_norm_raw = np.linalg.norm(f_sig[:n], axis=1)
    ref_floor = max(1.0e-30, 1.0e-6 * float(np.max(ref_norm_raw)))
    ref_norm = np.maximum(ref_norm_raw, ref_floor)
    err_kratos = np.linalg.norm(f_sig[:n] - k_sig[:n], axis=1) / ref_norm

    plt.figure(figsize=(7, 6))
    plt.plot(err_kratos, "c-", label="KRATOS Error vs FOM", linewidth=1.8)
    plt.title("Relative Stress Error (regularized): KRATOS vs FOM")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fom_vs_kratos_error.png"), dpi=150)
    plt.close()

    if timings:
        plt.figure(figsize=(6.8, 5.8))
        methods = [m for m in ["FOM", "KRATOS"] if m in timings]
        values = [timings[m] for m in methods]
        colors = {"FOM": "gray", "KRATOS": "c"}
        bars = plt.bar(methods, values, color=[colors.get(m, "tab:gray") for m in methods], alpha=0.85)
        for bar, t in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{t:.1f}s", ha="center", va="bottom")
        plt.title("Wall-Clock Time")
        plt.ylabel("Time [s]")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fom_vs_kratos_timing.png"), dpi=150)
        plt.close()

    print(f"FOM-vs-KRATOS plots saved to: {out_dir}")


def run_fom_vs_kratos(run_fom=False, run_kratos=False):
    out_dir = "stage_fom_vs_kratos_results"
    os.makedirs(out_dir, exist_ok=True)

    # Stage 8/10-compatible path logic
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
            domain_type = str(np.ravel(data["domain_type"])[0])

    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)

    # Visualization of test path
    plot_file = os.path.join(out_dir, "fom_vs_kratos_test_path.png")
    plot_path_in_domain(control_points, waypoints, emax, rel6, domain_type, plot_file)

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 64)
    print("  FOM vs KRATOS Benchmark")
    print("=" * 64)
    print(f"  Waypoints: {len(control_points)}")
    print(f"  Strain path points: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  Step-control reference amplitude: {emax}")
    print(f"  Segments: {seg_steps}")

    parameters = setup_kratos_parameters("rve_geometry")
    timings = {}

    # FOM
    fom_eps_file = os.path.join(out_dir, "fom_strain.npy")
    fom_sig_file = os.path.join(out_dir, "fom_stress.npy")
    if run_fom or not (os.path.exists(fom_eps_file) and os.path.exists(fom_sig_file)):
        print("\n[FOM-vs-KRATOS] Running FOM...")
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
        print("\n[FOM-vs-KRATOS] Loading cached FOM.")
        f_eps = np.load(fom_eps_file)
        f_sig = np.load(fom_sig_file)

    # KRATOS
    kr_eps_file = os.path.join(out_dir, "kratos_strain.npy")
    kr_sig_file = os.path.join(out_dir, "kratos_stress.npy")
    if run_kratos or not (os.path.exists(kr_eps_file) and os.path.exists(kr_sig_file)):
        print("\n[FOM-vs-KRATOS] Running KRATOS...")
        t0 = time.perf_counter()
        k_eps, k_sig, k_u, _ = RunKratosBatchSimulation(
            parameters,
            out_dir=out_dir,
            save_plot=False,
            save_data=False,
            strain_path=strain_path,
            trajectory_index=None,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        )
        timings["KRATOS"] = time.perf_counter() - t0
        k_eps = np.asarray(k_eps, dtype=float)
        k_sig = np.asarray(k_sig, dtype=float)
        k_u = np.asarray(k_u, dtype=float)
        np.save(kr_eps_file, k_eps)
        np.save(kr_sig_file, k_sig)
        np.save(os.path.join(out_dir, "kratos_U.npy"), k_u)
    else:
        print("\n[FOM-vs-KRATOS] Loading cached KRATOS.")
        k_eps = np.load(kr_eps_file)
        k_sig = np.load(kr_sig_file)

    n = min(len(f_sig), len(k_sig))
    err_kratos = np.linalg.norm(f_sig[:n] - k_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-30)
    err_u = None
    f_u_path = os.path.join(out_dir, "single_run_U.npy")
    k_u_path = os.path.join(out_dir, "kratos_U.npy")
    if os.path.exists(f_u_path) and os.path.exists(k_u_path):
        f_u = np.load(f_u_path)
        k_u = np.load(k_u_path)
        nu = min(len(f_u), len(k_u))
        err_u = np.linalg.norm(f_u[:nu] - k_u[:nu]) / (np.linalg.norm(f_u[:nu]) + 1e-30)

    print("\n" + "=" * 64)
    print("  FOM vs KRATOS Summary")
    print("=" * 64)
    print(f"  KRATOS vs FOM: Rel. Stress Error = {err_kratos:.4e} (n={n})")
    if err_u is not None:
        print(f"  KRATOS vs FOM: Rel. U Error      = {err_u:.4e}")
    for method, t in timings.items():
        print(f"  {method:8s} time: {t:.2f}s")

    summary_path = os.path.join(out_dir, "fom_vs_kratos_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FOM vs KRATOS Summary\n")
        f.write("=" * 32 + "\n")
        f.write(f"error_kratos_vs_fom: {err_kratos:.12e}\n")
        f.write(f"n_common: {n}\n")
        f.write(f"steps_total: {total_steps}\n")
        f.write(f"entries_total: {total_steps + 1}\n")
        if err_u is not None:
            f.write(f"error_u_kratos_vs_fom: {err_u:.12e}\n")
        for method in ["FOM", "KRATOS"]:
            if method in timings:
                f.write(f"time_{method}: {timings[method]:.12e}\n")

    plot_fom_vs_kratos(f_eps, f_sig, k_eps, k_sig, out_dir, timings)
    print(f"[FOM-vs-KRATOS] Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="FOM vs KRATOS benchmark on Stage 8/10 path")
    p.add_argument("--run-all", action="store_true", help="Force recomputation of both FOM and KRATOS.")
    p.add_argument("--run-fom", action="store_true", help="Force recomputation of FOM.")
    p.add_argument("--run-kratos", action="store_true", help="Force recomputation of KRATOS.")
    args = p.parse_args()

    if args.run_all:
        args.run_fom = True
        args.run_kratos = True

    run_fom_vs_kratos(run_fom=args.run_fom, run_kratos=args.run_kratos)
