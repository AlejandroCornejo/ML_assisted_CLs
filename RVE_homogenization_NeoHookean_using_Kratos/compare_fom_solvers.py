#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare KRATOS and FOM on the same Stage 8/10/11 trajectory.

Default behavior:
- builds the same safe test path used in stage workflows
- runs/loads 2 solvers on exactly that path
- saves stress/strain arrays, timing summary, and comparison plots

Usage:
    PYTHONPATH=/home/kratos/Kratos_Eigen_Check/bin/Release \
    LD_LIBRARY_PATH=/home/kratos/Kratos_Eigen_Check/bin/Release/libs \
    python3 compare_fom_solvers.py --run-all
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()

from stage6_test_hprom import generate_safe_test_path
from stage4_test_rve import plot_path_in_domain

import fom_solver_rve as fom_solver
from kratos_solver_rve import RunKratosBatchSimulation


METHOD_ORDER = ["KRATOS", "FOM"]


def _load_stage_domain(bundle_path="stage_0_trajectory/stage_0_trajectories.npz"):
    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"
    reference_amplitude_bundle = emax

    if os.path.exists(bundle_path):
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        if "emax" in data:
            emax = float(np.ravel(data["emax"])[0])
        else:
            emax = float(np.ravel(data["reference_amplitude"])[0])

        if "reference_amplitude" in data:
            reference_amplitude_bundle = float(np.ravel(data["reference_amplitude"])[0])
        else:
            reference_amplitude_bundle = float(emax)

        if "domain_type" in data:
            domain_type = str(np.ravel(data["domain_type"])[0])

    return emax, rel6, domain_type, reference_amplitude_bundle


def _rel_l2_vs_ref(sig_ref, sig_pred):
    n = min(len(sig_ref), len(sig_pred))
    num = np.linalg.norm(sig_ref[:n] - sig_pred[:n])
    den = np.linalg.norm(sig_ref[:n]) + 1e-30
    return float(num / den), int(n)


def _compute_equivalent_stress_strain(eps, sig):
    eps = np.asarray(eps, dtype=float)
    sig = np.asarray(sig, dtype=float)

    exx = eps[:, 0]
    eyy = eps[:, 1]
    gxy = eps[:, 2]

    sxx = sig[:, 0]
    syy = sig[:, 1]
    sxy = sig[:, 2]

    sigma_eq = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
    eps_eq = (2.0 / 3.0) * np.sqrt(np.maximum(exx * exx + eyy * eyy - exx * eyy + 0.75 * gxy * gxy, 0.0))
    return eps_eq, sigma_eq


def _plot_comparison(all_eps, all_sig, out_dir, timings, reference_method="FOM"):
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 12,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    styles = {
        "KRATOS": ("--", 1.8, "c"),
        "FOM": ("-", 2.1, "k"),
    }

    methods = [m for m in METHOD_ORDER if m in all_sig and m in all_eps]
    if not methods:
        return

    n_common = min(min(len(all_eps[m]), len(all_sig[m])) for m in methods)

    comp_labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "sigma_xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "sigma_yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "sigma_xy"),
    ]

    for i, label_sig, label_eps, suffix in comp_labels:
        plt.figure(figsize=(7.2, 6.0))
        for m in methods:
            ls, lw, color = styles[m]
            plt.plot(
                all_eps[m][:n_common, i],
                all_sig[m][:n_common, i],
                linestyle=ls,
                color=color,
                linewidth=lw,
                label=m,
            )
        plt.title(f"2-Solver Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"compare3_{suffix}.png"), dpi=150)
        plt.close()

    plt.figure(figsize=(7.2, 6.0))
    for m in methods:
        ls, lw, color = styles[m]
        eps_eq, sig_eq = _compute_equivalent_stress_strain(all_eps[m][:n_common], all_sig[m][:n_common])
        plt.plot(eps_eq, sig_eq, linestyle=ls, color=color, linewidth=lw, label=m)
    plt.title(r"2-Solver Benchmark: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare3_sigma_eq.png"), dpi=150)
    plt.close()

    if reference_method in methods and len(methods) > 1:
        ref_norm_raw = np.linalg.norm(all_sig[reference_method][:n_common], axis=1)
        ref_floor = max(1.0e-30, 1.0e-6 * float(np.max(ref_norm_raw)))
        ref_norm = np.maximum(ref_norm_raw, ref_floor)

        plt.figure(figsize=(7.2, 6.0))
        for m in methods:
            if m == reference_method:
                continue
            ls, lw, color = styles[m]
            err = np.linalg.norm(all_sig[reference_method][:n_common] - all_sig[m][:n_common], axis=1) / ref_norm
            plt.plot(err, linestyle=ls, color=color, linewidth=lw, label=f"{m} vs {reference_method}")

        plt.title(f"Relative Stress Error vs {reference_method}")
        plt.xlabel("Step")
        plt.ylabel("Relative Error [-]")
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compare3_error_vs_step.png"), dpi=150)
        plt.close()

    timing_methods = [m for m in methods if m in timings]
    if timing_methods:
        plt.figure(figsize=(7.2, 6.0))
        vals = [timings[m] for m in timing_methods]
        colors = {
            "KRATOS": "c",
            "FOM": "gray",
        }
        bars = plt.bar(
            timing_methods,
            vals,
            color=[colors.get(m, "tab:gray") for m in timing_methods],
            alpha=0.85,
        )
        for bar, t in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{t:.1f}s",
                ha="center",
                va="bottom",
            )
        plt.title("Wall-Clock Time")
        plt.ylabel("Time [s]")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compare3_timing.png"), dpi=150)
        plt.close()


def _run_kratos(strain_path, reference_amplitude, out_dir):
    parameters = fom_solver.setup_kratos_parameters("rve_geometry")
    t0 = time.perf_counter()
    eps, sig, _, _ = RunKratosBatchSimulation(
        parameters,
        out_dir=out_dir,
        save_plot=False,
        save_data=False,
        strain_path=strain_path,
        trajectory_index=1,
        reference_amplitude=reference_amplitude,
        reference_steps=fom_solver.REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    )
    return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0


def _run_fom(strain_path, reference_amplitude, out_dir):
    parameters = fom_solver.setup_kratos_parameters("rve_geometry")
    t0 = time.perf_counter()
    eps, sig = fom_solver.RunFomBatchSimulation(
        parameters,
        out_dir=out_dir,
        save_plot=False,
        strain_path=strain_path,
        trajectory_index=1,
        reference_amplitude=reference_amplitude,
        reference_steps=fom_solver.REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    )
    return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0


def run_compare(
    out_dir,
    run_kratos=False,
    run_fom=False,
    step_reference_amplitude=None,
    use_bundle_reference_amplitude=False,
):
    os.makedirs(out_dir, exist_ok=True)

    emax, rel6, domain_type, reference_amplitude_bundle = _load_stage_domain()

    if step_reference_amplitude is not None:
        reference_amplitude = float(step_reference_amplitude)
    elif use_bundle_reference_amplitude:
        reference_amplitude = float(reference_amplitude_bundle)
    else:
        reference_amplitude = float(emax)

    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)

    plot_path_in_domain(
        control_points,
        waypoints,
        emax,
        rel6,
        domain_type,
        os.path.join(out_dir, "compare3_test_path.png"),
    )

    seg_steps, _ref_amp_used = fom_solver.BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=fom_solver.REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=fom_solver.MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 70)
    print("  KRATOS vs FOM")
    print("=" * 70)
    print(f"  Control points: {len(control_points)}")
    print(f"  Path points: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  emax: {emax}")
    print(f"  Bundle reference amplitude: {reference_amplitude_bundle}")
    print(f"  Step-control reference amplitude (used): {reference_amplitude}")

    cache = {
        "KRATOS": (
            os.path.join(out_dir, "kratos_strain.npy"),
            os.path.join(out_dir, "kratos_stress.npy"),
        ),
        "FOM": (
            os.path.join(out_dir, "fom_strain.npy"),
            os.path.join(out_dir, "fom_stress.npy"),
        ),
    }

    forces = {
        "KRATOS": bool(run_kratos),
        "FOM": bool(run_fom),
    }

    subdirs = {
        "KRATOS": os.path.join(out_dir, "kratos_run"),
        "FOM": os.path.join(out_dir, "fom_run"),
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    all_eps = {}
    all_sig = {}
    timings = {}

    def run_or_load(method, runner):
        eps_file, sig_file = cache[method]
        if forces[method] or not (os.path.exists(eps_file) and os.path.exists(sig_file)):
            print(f"\n[COMPARE-3] Running {method}...")
            eps, sig, elapsed = runner(strain_path, reference_amplitude, subdirs[method])
            np.save(eps_file, eps)
            np.save(sig_file, sig)
            all_eps[method] = eps
            all_sig[method] = sig
            timings[method] = elapsed
        else:
            print(f"\n[COMPARE-3] Loading cached {method}.")
            all_eps[method] = np.asarray(np.load(eps_file), dtype=float)
            all_sig[method] = np.asarray(np.load(sig_file), dtype=float)

    run_or_load("KRATOS", _run_kratos)
    run_or_load("FOM", _run_fom)

    reference = "FOM"
    errors = {}
    for method in METHOD_ORDER:
        if method == reference:
            continue
        errors[method] = _rel_l2_vs_ref(all_sig[reference], all_sig[method])

    if "FOM" in timings and "KRATOS" in timings:
        timings["speedup_kratos_vs_fom"] = timings["FOM"] / max(timings["KRATOS"], 1e-30)

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    for method in ["KRATOS"]:
        err, n = errors[method]
        print(f"  {method:16s} vs FOM: Rel. Stress Error = {err:.4e} (n={n})")

    for method in METHOD_ORDER:
        if method in timings:
            print(f"  {method:16s} time: {timings[method]:.2f}s")
    if "speedup_kratos_vs_fom" in timings:
        print(f"  Speedup (FOM / KRATOS): {timings['speedup_kratos_vs_fom']:.3f}x")

    summary_path = os.path.join(out_dir, "compare3_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("KRATOS vs FOM\n")
        f.write("=" * 44 + "\n")
        f.write(f"steps_total: {total_steps}\n")
        f.write(f"entries_total: {total_steps + 1}\n")
        f.write(f"emax: {emax:.12e}\n")
        f.write(f"reference_amplitude_bundle: {reference_amplitude_bundle:.12e}\n")
        f.write(f"reference_amplitude_used: {reference_amplitude:.12e}\n")
        for method in ["KRATOS"]:
            err, n = errors[method]
            key = method.lower().replace("-", "_")
            f.write(f"error_{key}_vs_fom: {err:.12e}\n")
            f.write(f"n_{key}_vs_fom: {n}\n")
        for method in METHOD_ORDER:
            if method in timings:
                key = method.lower().replace("-", "_")
                f.write(f"time_{key}: {timings[method]:.12e}\n")
        if "speedup_kratos_vs_fom" in timings:
            f.write(f"speedup_kratos_vs_fom: {timings['speedup_kratos_vs_fom']:.12e}\n")

    _plot_comparison(all_eps, all_sig, out_dir, timings, reference_method=reference)
    print(f"\n[COMPARE-3] Summary saved to: {summary_path}")
    print(f"[COMPARE-3] Plots saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare KRATOS and FOM on Stage 8/10/11 trajectory"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="compare_2solvers_stage_path_results",
        help="Output directory for arrays, plots, and summary.",
    )
    parser.add_argument("--run-all", action="store_true", help="Force recomputation of all methods.")
    parser.add_argument("--run-kratos", action="store_true", help="Force recomputation of KRATOS.")
    parser.add_argument("--run-fom", action="store_true", help="Force recomputation of FOM.")
    parser.add_argument(
        "--step-reference-amplitude",
        type=float,
        default=None,
        help="Override dynamic-step reference amplitude (default: use emax, same as stage 8/10/11).",
    )
    parser.add_argument(
        "--use-bundle-reference-amplitude",
        action="store_true",
        help="Use stage_0 bundle reference_amplitude instead of emax for dynamic-step scaling.",
    )

    args = parser.parse_args()

    if args.run_all:
        args.run_kratos = True
        args.run_fom = True

    run_compare(
        out_dir=args.out_dir,
        run_kratos=args.run_kratos,
        run_fom=args.run_fom,
        step_reference_amplitude=args.step_reference_amplitude,
        use_bundle_reference_amplitude=args.use_bundle_reference_amplitude,
    )
