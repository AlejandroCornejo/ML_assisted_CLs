#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 10 (sparse-point variant):
  - Build the same full benchmark strain trajectory as stage10_test_hprom_gpr.py
  - Run HPROM-GPR only at a sparse subset of dynamic points (e.g. 20)
  - Compare against FOM sampled at the same points
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
    exx = eps[:, 0]
    eyy = eps[:, 1]
    gxy = eps[:, 2]
    sxx = sig[:, 0]
    syy = sig[:, 1]
    sxy = sig[:, 2]
    sigma_eq = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
    eps_eq = (2.0 / 3.0) * np.sqrt(np.maximum(exx * exx + eyy * eyy - exx * eyy + 0.75 * gxy * gxy, 0.0))
    return eps_eq, sigma_eq


def _build_step_macro_strain_series(strain_path, seg_steps):
    e_wp = np.asarray(strain_path, dtype=float)
    seg_steps = np.asarray(seg_steps, dtype=int).reshape(-1)
    n_steps = int(np.sum(seg_steps))
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    e_hist = np.zeros((n_steps + 1, 3), dtype=float)
    n_seg = int(len(seg_steps))
    for step in range(1, n_steps + 1):
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        e_hist[step, :] = (1.0 - xi) * e_wp[s, :3] + xi * e_wp[s + 1, :3]
    return e_hist


def _pick_sparse_step_indices(n_steps, n_points):
    n_steps = int(n_steps)
    n_points = int(max(1, n_points))
    if n_steps <= 0:
        return np.zeros(0, dtype=int)
    # select over dynamic increments [1..n_steps], keep endpoints
    idx = np.linspace(1, n_steps, num=min(n_points, n_steps), dtype=int)
    idx = np.unique(idx)
    if idx[-1] != n_steps:
        idx[-1] = n_steps
    return idx


def _plot_sparse_vs_fom(fom_eps_full, fom_sig_full, h_eps_sparse, h_sig_sparse, out_dir):
    n_sparse = min(len(h_sig_sparse), len(h_eps_sparse))
    labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "xy"),
    ]
    for comp, sy, ex, tag in labels:
        plt.figure(figsize=(7, 6))
        # Full FOM trajectory (all points)
        plt.plot(
            fom_eps_full[:, comp],
            fom_sig_full[:, comp],
            "k-",
            lw=1.6,
            label="FOM (all points)",
            zorder=1,
        )
        # Sparse HPROM-GPR evaluation points only
        plt.scatter(
            h_eps_sparse[:n_sparse, comp],
            h_sig_sparse[:n_sparse, comp],
            s=28,
            c="r",
            marker="o",
            label=f"HPROM-GPR ({n_sparse - 1} dynamic-point scatter)",
            zorder=3,
        )
        plt.xlabel(f"{ex} [-]")
        plt.ylabel(f"{sy} [Pa]")
        plt.title(f"Sparse-Point HPROM-GPR vs FOM: {sy}")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sparse_hprom_gpr_vs_fom_{tag}.png"), dpi=180)
        plt.close()

    # Equivalent curve: full FOM line + sparse HPROM scatter
    eps_eq_f, sig_eq_f = _compute_equivalent_stress_strain(fom_eps_full, fom_sig_full)
    eps_eq_h, sig_eq_h = _compute_equivalent_stress_strain(h_eps_sparse[:n_sparse], h_sig_sparse[:n_sparse])
    plt.figure(figsize=(7, 6))
    plt.plot(eps_eq_f, sig_eq_f, "k-", lw=1.6, label="FOM (all points)", zorder=1)
    plt.scatter(
        eps_eq_h,
        sig_eq_h,
        s=30,
        c="r",
        marker="o",
        label=f"HPROM-GPR ({n_sparse - 1} dynamic-point scatter)",
        zorder=3,
    )
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.title(r"Sparse-Point HPROM-GPR vs FOM: $\sigma_{eq}$")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sparse_hprom_gpr_vs_fom_sigma_eq.png"), dpi=180)
    plt.close()


def _plot_timing_comparison(out_dir, fom_time, hprom_time):
    labels = []
    values = []
    colors = []
    if np.isfinite(float(fom_time)):
        labels.append("FOM full")
        values.append(float(fom_time))
        colors.append("gray")
    labels.append("HPROM sparse")
    values.append(float(hprom_time))
    colors.append("red")
    if len(values) == 0:
        return
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=colors, alpha=0.85)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + 0.5 * b.get_width(), v, f"{v:.2f}s", ha="center", va="bottom")
    plt.ylabel("Wall time [s]")
    plt.title("Timing Comparison")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sparse_hprom_gpr_timing_comparison.png"), dpi=180)
    plt.close()


def run_stage10_sparse_points(
    n_points=20,
    run_fom=False,
    run_hprom_gpr=True,
    gpr_data_dir="stage_7_gpr_data_ls",
    hprom_gpr_dir="stage_9_hprom_gpr_data_ls",
    out_dir="stage_10_hprom_gpr_sparse_points",
    qp_init_mode="mu_affine",
):
    os.makedirs(out_dir, exist_ok=True)

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
    plot_path_in_domain(
        control_points,
        waypoints,
        emax,
        rel6,
        domain_type,
        os.path.join(out_dir, "sparse_hprom_gpr_test_path.png"),
    )

    seg_steps_full, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    n_steps_full = int(np.sum(seg_steps_full))
    e_hist_full = _build_step_macro_strain_series(strain_path, seg_steps_full)
    idx_sparse_dyn = _pick_sparse_step_indices(n_steps_full, n_points=n_points)
    idx_sparse_all = np.concatenate(([0], idx_sparse_dyn))
    strain_path_sparse = e_hist_full[idx_sparse_all, :]

    print("=" * 72)
    print("Stage 10 Sparse-Point HPROM-GPR Test")
    print("=" * 72)
    print(f"  Full dynamic steps: {n_steps_full}")
    print(f"  Sparse dynamic points requested: {int(n_points)}")
    print(f"  Sparse dynamic points used: {len(idx_sparse_dyn)}")
    print(f"  q_p init mode: {qp_init_mode}")
    print(f"  GPR data dir: {gpr_data_dir}")
    print(f"  HPROM-GPR dir: {hprom_gpr_dir}")
    print(f"  Output dir: {out_dir}")

    mesh_fom_prom = "rve_geometry"
    mesh_hprom = "rve_geometry"
    auto_hrom_mesh = _load_hrom_mesh_base_from_ecm_dir(hprom_gpr_dir)
    if auto_hrom_mesh:
        mesh_hprom = str(auto_hrom_mesh)
    print(f"  FOM mesh:   {mesh_fom_prom}")
    print(f"  HPROM mesh: {mesh_hprom}")

    parameters_fom = setup_kratos_parameters(mesh_fom_prom)
    parameters_hprom = setup_kratos_parameters(mesh_hprom)

    fom_eps_file = os.path.join(out_dir, "fom_full_strain.npy")
    fom_sig_file = os.path.join(out_dir, "fom_full_stress.npy")
    t_fom = np.nan
    if run_fom or not (os.path.exists(fom_eps_file) and os.path.exists(fom_sig_file)):
        print("\n[Stage10-Sparse] Running full FOM trajectory (reference)...")
        t0 = time.perf_counter()
        f_eps_full, f_sig_full = RunFomBatchSimulation(
            parameters_fom,
            out_dir=out_dir,
            strain_path=strain_path,
            trajectory_index=None,
            save_plot=False,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        )
        t_fom = time.perf_counter() - t0
        print(f"  FOM full time: {t_fom:.2f}s")
        f_eps_full = np.asarray(f_eps_full, dtype=float)
        f_sig_full = np.asarray(f_sig_full, dtype=float)
        np.save(fom_eps_file, f_eps_full)
        np.save(fom_sig_file, f_sig_full)
    else:
        print("\n[Stage10-Sparse] Loading cached full FOM trajectory.")
        f_eps_full = np.load(fom_eps_file)
        f_sig_full = np.load(fom_sig_file)

    f_eps_sparse = f_eps_full[idx_sparse_all, :]
    f_sig_sparse = f_sig_full[idx_sparse_all, :]
    np.save(os.path.join(out_dir, "fom_sparse_indices.npy"), idx_sparse_all)
    np.save(os.path.join(out_dir, "fom_sparse_strain.npy"), f_eps_sparse)
    np.save(os.path.join(out_dir, "fom_sparse_stress.npy"), f_sig_sparse)
    np.save(os.path.join(out_dir, "sparse_macro_strain_path.npy"), strain_path_sparse)

    if not run_hprom_gpr:
        print("\n[Stage10-Sparse] HPROM-GPR run skipped by flag.")
        return

    print("\n[Stage10-Sparse] Running HPROM-GPR on sparse points only...")
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
    h_eps_sparse, h_sig_sparse = RunHpromGprBatchSimulation(
        parameters_hprom,
        phi_p_h,
        phi_s_h,
        free_dofs_h,
        gpr_model_h,
        ecm_data_h,
        out_dir=out_dir,
        strain_path=strain_path_sparse,
        trajectory_index=None,
        # Force one dynamic solve per sparse segment:
        reference_amplitude=1.0e12,
        reference_steps=1,
        eq_map_full=eq_map_h,
        Xc=Xc_h,
        Yc=Yc_h,
        qp_init_mode=qp_init_mode,
    )
    t_h = time.perf_counter() - t0
    h_eps_sparse = np.asarray(h_eps_sparse, dtype=float)
    h_sig_sparse = np.asarray(h_sig_sparse, dtype=float)
    np.save(os.path.join(out_dir, "hprom_sparse_strain.npy"), h_eps_sparse)
    np.save(os.path.join(out_dir, "hprom_sparse_stress.npy"), h_sig_sparse)

    n = min(len(f_sig_sparse), len(h_sig_sparse))
    err_sig = float(np.linalg.norm(f_sig_sparse[:n] - h_sig_sparse[:n]) / max(np.linalg.norm(f_sig_sparse[:n]), 1e-30))
    err_eps = float(np.linalg.norm(f_eps_sparse[:n] - h_eps_sparse[:n]) / max(np.linalg.norm(f_eps_sparse[:n]), 1e-30))
    f_eps_eq_sparse, f_sig_eq_sparse = _compute_equivalent_stress_strain(f_eps_sparse[:n], f_sig_sparse[:n])
    h_eps_eq_sparse, h_sig_eq_sparse = _compute_equivalent_stress_strain(h_eps_sparse[:n], h_sig_sparse[:n])
    err_sig_eq = float(np.linalg.norm(f_sig_eq_sparse - h_sig_eq_sparse) / max(np.linalg.norm(f_sig_eq_sparse), 1e-30))
    err_eps_eq = float(np.linalg.norm(f_eps_eq_sparse - h_eps_eq_sparse) / max(np.linalg.norm(f_eps_eq_sparse), 1e-30))
    hprom_time_per_dynamic = float(t_h / max(len(idx_sparse_dyn), 1))
    fom_time_per_dynamic = float(t_fom / max(n_steps_full, 1)) if np.isfinite(float(t_fom)) else np.nan

    with open(os.path.join(out_dir, "sparse_hprom_gpr_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"full_dynamic_steps={n_steps_full}\n")
        f.write(f"sparse_dynamic_points={len(idx_sparse_dyn)}\n")
        f.write(f"qp_init_mode={qp_init_mode}\n")
        f.write(f"fom_full_runtime_s={t_fom:.16e}\n")
        f.write(f"hprom_sparse_runtime_s={t_h:.16e}\n")
        f.write(f"fom_time_per_dynamic_s={fom_time_per_dynamic:.16e}\n")
        f.write(f"hprom_time_per_dynamic_s={hprom_time_per_dynamic:.16e}\n")
        f.write(f"rel_stress_error_vs_fom_sparse={err_sig:.16e}\n")
        f.write(f"rel_strain_error_vs_fom_sparse={err_eps:.16e}\n")
        f.write(f"rel_sigma_eq_error_vs_fom_sparse={err_sig_eq:.16e}\n")
        f.write(f"rel_eps_eq_error_vs_fom_sparse={err_eps_eq:.16e}\n")
        f.write("sparse_indices_dynamic=" + ",".join([str(int(v)) for v in idx_sparse_dyn]) + "\n")

    print("\n" + "=" * 72)
    print("Sparse-Point Summary")
    print("=" * 72)
    if np.isfinite(float(t_fom)):
        print(f"  FOM full runtime: {t_fom:.2f}s")
        print(f"  FOM time per dynamic step: {fom_time_per_dynamic:.4f}s")
    else:
        print("  FOM full runtime: cache loaded (run with --run-fom for timing).")
    print(f"  HPROM sparse runtime: {t_h:.2f}s")
    print(f"  HPROM time per sparse dynamic point: {hprom_time_per_dynamic:.4f}s")
    print(f"  Rel stress error (sparse vs sampled FOM): {err_sig:.4e}")
    print(f"  Rel strain error (sparse vs sampled FOM): {err_eps:.4e}")
    print(f"  Rel sigma_eq error (sparse vs sampled FOM): {err_sig_eq:.4e}")
    print(f"  Rel eps_eq error (sparse vs sampled FOM): {err_eps_eq:.4e}")

    _plot_sparse_vs_fom(f_eps_full, f_sig_full, h_eps_sparse, h_sig_sparse, out_dir=out_dir)
    _plot_timing_comparison(out_dir, fom_time=t_fom, hprom_time=t_h)
    print(f"  Saved outputs in: {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Stage 10 sparse-point HPROM-GPR test (HPROM solved only on selected macro points)."
    )
    p.add_argument("--n-points", type=int, default=20, help="Number of dynamic points sampled from full 1150-step path.")
    p.add_argument("--run-fom", action="store_true", help="Force full FOM recompute.")
    p.add_argument("--run-hprom-gpr", action="store_true", help="Run sparse HPROM-GPR solve.")
    p.add_argument("--gpr-data-dir", type=str, default="stage_7_gpr_data_ls")
    p.add_argument("--hprom-gpr-dir", type=str, default="stage_9_hprom_gpr_data_ls")
    p.add_argument("--out-dir", type=str, default="stage_10_hprom_gpr_sparse_points")
    p.add_argument(
        "--qp-init-mode",
        type=str,
        default="mu_affine",
        choices=["previous", "zero", "mu_affine"],
        help="Initializer for q_p at each sparse point.",
    )
    args = p.parse_args()

    run_stage10_sparse_points(
        n_points=args.n_points,
        run_fom=args.run_fom,
        run_hprom_gpr=args.run_hprom_gpr,
        gpr_data_dir=args.gpr_data_dir,
        hprom_gpr_dir=args.hprom_gpr_dir,
        out_dir=args.out_dir,
        qp_init_mode=args.qp_init_mode,
    )
