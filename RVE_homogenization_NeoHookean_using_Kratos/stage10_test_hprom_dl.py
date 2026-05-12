#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 10: FOM vs PROM-POD-DL vs HPROM-POD-DL benchmark.
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
from prom_dl_solver_rve import LoadPromDlModel, RunPromDlBatchSimulation
from hprom_dl_solver_rve import LoadHpromDlModel, RunHpromDlBatchSimulation


def _load_hrom_mesh_base_from_ecm_dir(hprom_dl_dir):
    ecm_file = os.path.join(str(hprom_dl_dir), "ecm_weights_all.npz")
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


def plot_hprom_dl_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings=None):
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
        plt.plot(p_eps[:n, i], p_sig[:n, i], "r--", label="PROM-POD-DL", linewidth=1.5)
        plt.plot(h_eps[:n, i], h_sig[:n, i], "b:", label="HPROM-POD-DL", linewidth=1.5)
        plt.title(f"HPROM-POD-DL Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hprom_dl_comp_{suffix}.png"), dpi=150)
        plt.close()

    f_eps_eq, f_sig_eq = _compute_equivalent_stress_strain(f_eps[:n], f_sig[:n])
    p_eps_eq, p_sig_eq = _compute_equivalent_stress_strain(p_eps[:n], p_sig[:n])
    h_eps_eq, h_sig_eq = _compute_equivalent_stress_strain(h_eps[:n], h_sig[:n])

    plt.figure(figsize=(7, 6))
    plt.plot(f_eps_eq, f_sig_eq, "k-", label="FOM", linewidth=2.0)
    plt.plot(p_eps_eq, p_sig_eq, "r--", label="PROM-POD-DL", linewidth=1.5)
    plt.plot(h_eps_eq, h_sig_eq, "b:", label="HPROM-POD-DL", linewidth=1.5)
    plt.title(r"HPROM-POD-DL Benchmark: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_dl_comp_sigma_eq.png"), dpi=150)
    plt.close()

    fom_norm = np.linalg.norm(f_sig[:n], axis=1) + 1e-30
    err_prom = np.linalg.norm(f_sig[:n] - p_sig[:n], axis=1) / fom_norm
    err_hprom = np.linalg.norm(f_sig[:n] - h_sig[:n], axis=1) / fom_norm

    plt.figure(figsize=(7, 6))
    plt.plot(err_prom, "r-", label="PROM-POD-DL Error", linewidth=1.5)
    plt.plot(err_hprom, "b-", label="HPROM-POD-DL Error", linewidth=1.5)
    plt.title("Relative Stress Error Comparison")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_dl_comp_error.png"), dpi=150)
    plt.close()

    if timings:
        plt.figure(figsize=(7, 6))
        methods = list(timings.keys())
        values = [timings[m] for m in methods]
        colors = {"FOM": "gray", "PROM-POD-DL": "red", "HPROM-POD-DL": "blue"}
        bars = plt.bar(methods, values, color=[colors.get(m, "green") for m in methods], alpha=0.8)
        for bar, t in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{t:.1f}s", ha="center", va="bottom")
        plt.title("Computational Performance")
        plt.ylabel("Wall-Clock Time [s]")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hprom_dl_performance.png"), dpi=150)
        plt.close()

    print(f"Stage 10 plots saved to: {out_dir}")


def run_stage10(
    run_fom=False,
    run_prom_dl=False,
    run_hprom_dl=False,
    pod_dl_data_dir="stage_7_pod_dl_data",
    hprom_dl_dir="stage_9_hprom_pod_dl_data",
    out_dir="stage_10_hprom_dl_results",
    hprom_only=False,
    no_auto_run_missing_baselines=False,
    prom_linear_solver_mode="auto",
    prom_lstsq_rcond=None,
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

    _plot_file = os.path.join(out_dir, "hprom_dl_test_path.png")
    plot_path_in_domain(control_points, waypoints, emax, rel6, domain_type, _plot_file)

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 60)
    print("  Stage 10: HPROM-POD-DL Benchmark (FOM vs PROM-POD-DL vs HPROM-POD-DL)")
    print("=" * 60)
    print(f"  Waypoints: {len(control_points)}")
    print(f"  Strain path points: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  Segments: {seg_steps}")
    print(f"  POD-DL data dir: {pod_dl_data_dir}")
    print(f"  HPROM-POD-DL dir: {hprom_dl_dir}")
    print(f"  Output dir: {out_dir}")
    if hprom_only:
        print("  Mode: HPROM-only (PROM/FOM skipped)")
    else:
        print(
            "  PROM-POD-DL linear solve: "
            f"mode={str(prom_linear_solver_mode).strip().lower()}, "
            f"lstsq_rcond={'default' if prom_lstsq_rcond is None else f'{float(prom_lstsq_rcond):.3e}'}"
        )

    mesh_fom_prom = "rve_geometry"
    mesh_hprom = "rve_geometry"
    auto_hrom_mesh = _load_hrom_mesh_base_from_ecm_dir(hprom_dl_dir)
    if auto_hrom_mesh:
        mesh_hprom = str(auto_hrom_mesh)
    print(f"  FOM/PROM mesh: {mesh_fom_prom}")
    print(f"  HPROM mesh:    {mesh_hprom}")

    parameters = setup_kratos_parameters(mesh_fom_prom)
    parameters_hprom = setup_kratos_parameters(mesh_hprom)
    timings = {}

    fom_eps_file = os.path.join(out_dir, "fom_strain.npy")
    fom_sig_file = os.path.join(out_dir, "fom_stress.npy")
    has_fom_cache = os.path.exists(fom_eps_file) and os.path.exists(fom_sig_file)
    if hprom_only:
        f_eps = None
        f_sig = None
    else:
        if run_fom or (not has_fom_cache and not no_auto_run_missing_baselines):
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
        elif has_fom_cache:
            print("\n[Stage 10] Loading cached FOM.")
            f_eps = np.load(fom_eps_file)
            f_sig = np.load(fom_sig_file)
        else:
            f_eps = None
            f_sig = None

    prom_eps_file = os.path.join(out_dir, "prom_dl_strain.npy")
    prom_sig_file = os.path.join(out_dir, "prom_dl_stress.npy")
    has_prom_cache = os.path.exists(prom_eps_file) and os.path.exists(prom_sig_file)
    if hprom_only:
        p_eps = None
        p_sig = None
    else:
        if run_prom_dl or (not has_prom_cache and not no_auto_run_missing_baselines):
            print("\n[Stage 10] Running PROM-POD-DL...")
            phi_q, free_dofs, _, _, pod_dl_model, device, checkpoint = LoadPromDlModel(
                basis_dir="stage_2_pod_rve",
                pod_dl_data_dir=pod_dl_data_dir,
            )
            t0 = time.perf_counter()
            p_eps, p_sig = RunPromDlBatchSimulation(
                parameters,
                phi_q,
                free_dofs,
                pod_dl_model,
                device,
                strain_path,
                out_dir=out_dir,
                reference_amplitude=emax,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
                linear_solver_mode=str(prom_linear_solver_mode),
                lstsq_rcond=prom_lstsq_rcond,
            )
            timings["PROM-POD-DL"] = time.perf_counter() - t0
            p_eps = np.asarray(p_eps, dtype=float)
            p_sig = np.asarray(p_sig, dtype=float)
            np.save(prom_eps_file, p_eps)
            np.save(prom_sig_file, p_sig)
            print(f"  q_dim={int(checkpoint['q_dim'])} | latent_dim={int(checkpoint['latent_dim'])}")
        elif has_prom_cache:
            print("\n[Stage 10] Loading cached PROM-POD-DL.")
            p_eps = np.load(prom_eps_file)
            p_sig = np.load(prom_sig_file)
        else:
            p_eps = None
            p_sig = None

    hprom_eps_file = os.path.join(out_dir, "hprom_dl_strain.npy")
    hprom_sig_file = os.path.join(out_dir, "hprom_dl_stress.npy")
    has_hprom_cache = os.path.exists(hprom_eps_file) and os.path.exists(hprom_sig_file)
    if run_hprom_dl or not has_hprom_cache:
        print("\n[Stage 10] Running HPROM-POD-DL...")
        (
            phi_q_h,
            free_dofs_h,
            _dir_dofs_h,
            eq_map_h,
            Xc_h,
            Yc_h,
            pod_dl_model_h,
            device_h,
            checkpoint_h,
            ecm_data_h,
        ) = LoadHpromDlModel(
            basis_dir="stage_2_pod_rve",
            pod_dl_data_dir=pod_dl_data_dir,
            hprom_dl_dir=hprom_dl_dir,
        )
        t0 = time.perf_counter()
        h_eps, h_sig = RunHpromDlBatchSimulation(
            parameters_hprom,
            phi_q_h,
            free_dofs_h,
            pod_dl_model_h,
            device_h,
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
        timings["HPROM-POD-DL"] = time.perf_counter() - t0
        h_eps = np.asarray(h_eps, dtype=float)
        h_sig = np.asarray(h_sig, dtype=float)
        np.save(hprom_eps_file, h_eps)
        np.save(hprom_sig_file, h_sig)
        print(f"  q_dim={int(checkpoint_h['q_dim'])} | latent_dim={int(checkpoint_h['latent_dim'])}")
    else:
        print("\n[Stage 10] Loading cached HPROM-POD-DL.")
        h_eps = np.load(hprom_eps_file)
        h_sig = np.load(hprom_sig_file)

    if hprom_only:
        print("\n" + "=" * 60)
        print("  Stage 10 HPROM-POD-DL Only")
        print("=" * 60)
        if "HPROM-POD-DL" in timings:
            print(f"  HPROM-POD-DL time: {timings['HPROM-POD-DL']:.2f}s")
        print("  Baseline comparison skipped by --hprom-only.")
        return

    if f_eps is None or f_sig is None:
        raise RuntimeError(
            "FOM cache not found and auto-run is disabled. "
            "Run with --run-fom or remove --no-auto-run-missing-baselines."
        )
    if p_eps is None or p_sig is None:
        raise RuntimeError(
            "PROM-POD-DL cache not found and auto-run is disabled. "
            "Run with --run-prom-dl or remove --no-auto-run-missing-baselines."
        )

    n = min(len(f_sig), len(p_sig), len(h_sig))
    err_prom = np.linalg.norm(f_sig[:n] - p_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-30)
    err_hprom = np.linalg.norm(f_sig[:n] - h_sig[:n]) / (np.linalg.norm(f_sig[:n]) + 1e-30)

    print("\n" + "=" * 60)
    print("  Stage 10 Summary")
    print("=" * 60)
    print(f"  PROM-POD-DL  vs FOM: Rel. Stress Error = {err_prom:.4e}")
    print(f"  HPROM-POD-DL vs FOM: Rel. Stress Error = {err_hprom:.4e}")
    for method, t in timings.items():
        print(f"  {method} time: {t:.2f}s")

    plot_hprom_dl_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 10: HPROM-POD-DL benchmark")
    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument("--run-prom-dl", action="store_true", help="Force PROM-POD-DL recompute.")
    p.add_argument("--run-hprom-dl", action="store_true", help="Force HPROM-POD-DL recompute.")
    p.add_argument(
        "--hprom-only",
        action="store_true",
        help="Run/load only HPROM-POD-DL and skip FOM/PROM comparisons.",
    )
    p.add_argument(
        "--no-auto-run-missing-baselines",
        action="store_true",
        help="Do not auto-run missing FOM/PROM caches; error out instead.",
    )
    p.add_argument(
        "--prom-linear-solver-mode",
        type=str,
        default="auto",
        choices=["auto", "solve", "lstsq"],
        help="Reduced linear solver mode for PROM-POD-DL when PROM is executed in this stage.",
    )
    p.add_argument(
        "--prom-lstsq-rcond",
        type=float,
        default=None,
        help="Optional rcond for PROM-POD-DL least-squares mode.",
    )
    p.add_argument(
        "--pod-dl-data-dir",
        type=str,
        default="stage_7_pod_dl_data",
        help="Directory with POD-DL model files.",
    )
    p.add_argument(
        "--hprom-dl-dir",
        type=str,
        default="stage_9_hprom_pod_dl_data",
        help="Directory with HPROM-POD-DL ECM file (ecm_weights_all.npz).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="stage_10_hprom_dl_results",
        help="Output directory for Stage 10 POD-DL results.",
    )
    args = p.parse_args()

    run_stage10(
        run_fom=args.run_fom,
        run_prom_dl=args.run_prom_dl,
        run_hprom_dl=args.run_hprom_dl,
        pod_dl_data_dir=args.pod_dl_data_dir,
        hprom_dl_dir=args.hprom_dl_dir,
        out_dir=args.out_dir,
        hprom_only=args.hprom_only,
        no_auto_run_missing_baselines=args.no_auto_run_missing_baselines,
        prom_linear_solver_mode=args.prom_linear_solver_mode,
        prom_lstsq_rcond=args.prom_lstsq_rcond,
    )
