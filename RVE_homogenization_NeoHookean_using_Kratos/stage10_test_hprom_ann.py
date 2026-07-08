#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 10: FOM vs PROM-ANN vs HPROM-ANN benchmark.
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
from prom_ann_solver_rve import LoadPromAnnModel, RunPromAnnBatchSimulation
from hprom_ann_solver_rve import LoadHpromAnnModel, RunHpromAnnBatchSimulation


def _load_hrom_mesh_base_from_ecm_dir(hprom_ann_dir):
    ecm_file = os.path.join(str(hprom_ann_dir), "ecm_weights_all.npz")
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


def _rel_error(pred, ref):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-30))


def _component_rel_errors(pred, ref, labels):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n_comp = min(pred.shape[1], ref.shape[1], len(labels))
    return {
        labels[i]: _rel_error(pred[:, i], ref[:, i])
        for i in range(n_comp)
    }


def _format_component_errors(errors, labels):
    return ", ".join(f"{label}={errors[label]:.4e}" for label in labels if label in errors)


def _load_first_existing_npy(out_dir, candidates):
    for name in candidates:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            return np.load(path), path
    return None, None


def plot_hprom_ann_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings=None):
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
        plt.plot(p_eps[:n, i], p_sig[:n, i], "r--", label="PROM-ANN", linewidth=1.5)
        plt.plot(h_eps[:n, i], h_sig[:n, i], "b:", label="HPROM-ANN", linewidth=1.5)
        plt.title(f"HPROM-ANN Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hprom_ann_comp_{suffix}.png"), dpi=150)
        plt.close()

    f_eps_eq, f_sig_eq = _compute_equivalent_stress_strain(f_eps[:n], f_sig[:n])
    p_eps_eq, p_sig_eq = _compute_equivalent_stress_strain(p_eps[:n], p_sig[:n])
    h_eps_eq, h_sig_eq = _compute_equivalent_stress_strain(h_eps[:n], h_sig[:n])

    plt.figure(figsize=(7, 6))
    plt.plot(f_eps_eq, f_sig_eq, "k-", label="FOM", linewidth=2.0)
    plt.plot(p_eps_eq, p_sig_eq, "r--", label="PROM-ANN", linewidth=1.5)
    plt.plot(h_eps_eq, h_sig_eq, "b:", label="HPROM-ANN", linewidth=1.5)
    plt.title(r"HPROM-ANN Benchmark: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_ann_comp_sigma_eq.png"), dpi=150)
    plt.close()

    fom_norm = np.linalg.norm(f_sig[:n], axis=1) + 1e-30
    err_prom_ann = np.linalg.norm(f_sig[:n] - p_sig[:n], axis=1) / fom_norm
    err_hprom_ann = np.linalg.norm(f_sig[:n] - h_sig[:n], axis=1) / fom_norm

    plt.figure(figsize=(7, 6))
    plt.plot(err_prom_ann, "r-", label="PROM-ANN Error", linewidth=1.5)
    plt.plot(err_hprom_ann, "b-", label="HPROM-ANN Error", linewidth=1.5)
    plt.title("Relative Stress Error Comparison")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_ann_comp_error.png"), dpi=150)
    plt.close()

    if timings:
        plt.figure(figsize=(7, 6))
        methods = list(timings.keys())
        values = [timings[m] for m in methods]
        colors = {"FOM": "gray", "PROM-ANN": "red", "HPROM-ANN": "blue"}
        bars = plt.bar(methods, values, color=[colors.get(m, "green") for m in methods], alpha=0.8)
        for bar, t in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{t:.1f}s", ha="center", va="bottom")
        plt.title("Computational Performance")
        plt.ylabel("Wall-Clock Time [s]")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hprom_ann_performance.png"), dpi=150)
        plt.close()

    print(f"Stage 10 plots saved to: {out_dir}")


def run_stage10(
    run_fom=False,
    run_prom_ann=False,
    run_hprom_ann=False,
    ann_data_dir="stage_7_ann_data",
    hprom_ann_dir="stage_9_hprom_ann_data",
    out_dir="stage_10_hprom_ann_results",
    hprom_homogenization_mode="ecm_fixed",
    hprom_maw_hom_eval_mode="model",
    hprom_corrector_iters=25,
    qp_init_mode="continuation",
):
    os.makedirs(out_dir, exist_ok=True)
    if int(hprom_corrector_iters) == 0 and str(qp_init_mode).strip().lower() != "mu_affine":
        print(
            "  [Stage10] Direct HPROM-ANN mode requested "
            "(--hprom-corrector-iters 0): forcing qp_init_mode='mu_affine'."
        )
        qp_init_mode = "mu_affine"

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
    _plot_file = os.path.join(out_dir, "hprom_ann_test_path.png")
    plot_path_in_domain(control_points, waypoints, emax, rel6, domain_type, _plot_file)

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 60)
    print("  Stage 10: HPROM-ANN Benchmark (FOM vs PROM-ANN vs HPROM-ANN)")
    print("=" * 60)
    print(f"  Waypoints: {len(control_points)}")
    print(f"  Strain path points: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  Segments: {seg_steps}")
    print(f"  ANN data dir: {ann_data_dir}")
    print(f"  HPROM-ANN dir: {hprom_ann_dir}")
    print(f"  Output dir: {out_dir}")
    print(f"  HPROM homogenization mode: {hprom_homogenization_mode}")
    print(f"  HPROM MAW hom eval mode: {hprom_maw_hom_eval_mode}")
    print(f"  HPROM corrector iterations: {hprom_corrector_iters}")
    print(f"  HPROM q_m initializer mode: {qp_init_mode}")

    mesh_fom_prom = "rve_geometry"
    mesh_hprom = "rve_geometry"
    auto_hrom_mesh = _load_hrom_mesh_base_from_ecm_dir(hprom_ann_dir)
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

    prom_eps_file = os.path.join(out_dir, "prom_ann_strain.npy")
    prom_sig_file = os.path.join(out_dir, "prom_ann_stress.npy")
    if run_prom_ann or not (os.path.exists(prom_eps_file) and os.path.exists(prom_sig_file)):
        print("\n[Stage 10] Running PROM-ANN...")
        phi_p, phi_s, free_dofs, _, _, ann_model, device, _include_macro = LoadPromAnnModel(
            basis_dir="stage_2_pod_rve",
            ann_data_dir=ann_data_dir,
        )
        t0 = time.perf_counter()
        p_eps, p_sig = RunPromAnnBatchSimulation(
            parameters,
            phi_p,
            phi_s,
            free_dofs,
            ann_model,
            device,
            strain_path,
            out_dir=out_dir,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        )
        timings["PROM-ANN"] = time.perf_counter() - t0
        p_eps = np.asarray(p_eps, dtype=float)
        p_sig = np.asarray(p_sig, dtype=float)
        np.save(prom_eps_file, p_eps)
        np.save(prom_sig_file, p_sig)
    else:
        print("\n[Stage 10] Loading cached PROM-ANN.")
        p_eps = np.load(prom_eps_file)
        p_sig = np.load(prom_sig_file)

    hprom_eps_file = os.path.join(out_dir, "hprom_ann_strain.npy")
    hprom_sig_file = os.path.join(out_dir, "hprom_ann_stress.npy")
    if run_hprom_ann or not (os.path.exists(hprom_eps_file) and os.path.exists(hprom_sig_file)):
        print("\n[Stage 10] Running HPROM-ANN...")
        (
            phi_p_h,
            phi_s_h,
            free_dofs_h,
            _dir_dofs_h,
            eq_map_h,
            Xc_h,
            Yc_h,
            ann_model_h,
            device_h,
            ecm_data_h,
            _include_macro_h,
        ) = LoadHpromAnnModel(
            basis_dir="stage_2_pod_rve",
            ann_data_dir=ann_data_dir,
            hprom_ann_dir=hprom_ann_dir,
        )
        t0 = time.perf_counter()
        h_eps, h_sig = RunHpromAnnBatchSimulation(
            parameters_hprom,
            phi_p_h,
            phi_s_h,
            free_dofs_h,
            ann_model_h,
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
            homogenization_mode=hprom_homogenization_mode,
            maw_hom_eval_mode=hprom_maw_hom_eval_mode,
            max_its=int(hprom_corrector_iters),
            qp_init_mode=qp_init_mode,
        )
        timings["HPROM-ANN"] = time.perf_counter() - t0
        h_eps = np.asarray(h_eps, dtype=float)
        h_sig = np.asarray(h_sig, dtype=float)
        np.save(hprom_eps_file, h_eps)
        np.save(hprom_sig_file, h_sig)
    else:
        print("\n[Stage 10] Loading cached HPROM-ANN.")
        h_eps = np.load(hprom_eps_file)
        h_sig = np.load(hprom_sig_file)

    n = min(len(f_sig), len(p_sig), len(h_sig), len(f_eps), len(p_eps), len(h_eps))
    err_prom = _rel_error(p_sig[:n], f_sig[:n])
    err_hprom = _rel_error(h_sig[:n], f_sig[:n])
    err_hprom_vs_prom = _rel_error(h_sig[:n], p_sig[:n])

    stress_labels = ["sigma_xx", "sigma_yy", "sigma_xy"]
    strain_labels = ["eps_xx", "eps_yy", "gamma_xy"]
    stress_prom_fom = _component_rel_errors(p_sig[:n], f_sig[:n], stress_labels)
    stress_hprom_fom = _component_rel_errors(h_sig[:n], f_sig[:n], stress_labels)
    stress_hprom_prom = _component_rel_errors(h_sig[:n], p_sig[:n], stress_labels)
    strain_prom_fom = _component_rel_errors(p_eps[:n], f_eps[:n], strain_labels)
    strain_hprom_fom = _component_rel_errors(h_eps[:n], f_eps[:n], strain_labels)
    strain_hprom_prom = _component_rel_errors(h_eps[:n], p_eps[:n], strain_labels)

    q_prom, q_prom_path = _load_first_existing_npy(
        out_dir,
        [
            "prom_ann_run_q_p.npy",
            "prom_ann_q_p.npy",
        ],
    )
    q_hprom, q_hprom_path = _load_first_existing_npy(
        out_dir,
        [
            "hprom_ann_run_q_p.npy",
            "hprom_ann_q_p.npy",
        ],
    )

    q_error = None
    q_comp_errors = None
    q_rmse = None
    q_max_abs = None
    if q_prom is not None and q_hprom is not None:
        q_prom = np.asarray(q_prom, dtype=float)
        q_hprom = np.asarray(q_hprom, dtype=float)
        nq = min(len(q_prom), len(q_hprom))
        ncomp_q = min(q_prom.shape[1], q_hprom.shape[1]) if q_prom.ndim == 2 and q_hprom.ndim == 2 else 0
        if nq > 0 and ncomp_q > 0:
            q_prom_a = q_prom[:nq, :ncomp_q]
            q_hprom_a = q_hprom[:nq, :ncomp_q]
            q_diff = q_hprom_a - q_prom_a
            q_error = _rel_error(q_hprom_a, q_prom_a)
            q_comp_errors = {
                f"q_{i + 1}": _rel_error(q_hprom_a[:, i], q_prom_a[:, i])
                for i in range(ncomp_q)
            }
            q_rmse = float(np.sqrt(np.mean(q_diff * q_diff)))
            q_max_abs = float(np.max(np.abs(q_diff)))

    summary_lines = [
        "",
        "=" * 60,
        "  Stage 10 Summary",
        "=" * 60,
        f"  PROM-ANN  vs FOM: Rel. Stress Error = {err_prom:.4e}",
        f"  HPROM-ANN vs FOM: Rel. Stress Error = {err_hprom:.4e}",
        f"  HPROM-ANN vs PROM-ANN: Rel. Stress Error = {err_hprom_vs_prom:.4e}",
        "  Stress component relative errors [reference in denominator]:",
        f"    PROM/FOM       : {_format_component_errors(stress_prom_fom, stress_labels)}",
        f"    HPROM/FOM      : {_format_component_errors(stress_hprom_fom, stress_labels)}",
        f"    HPROM/PROM     : {_format_component_errors(stress_hprom_prom, stress_labels)}",
        "  Strain component relative errors [reference in denominator]:",
        f"    PROM/FOM       : {_format_component_errors(strain_prom_fom, strain_labels)}",
        f"    HPROM/FOM      : {_format_component_errors(strain_hprom_fom, strain_labels)}",
        f"    HPROM/PROM     : {_format_component_errors(strain_hprom_prom, strain_labels)}",
    ]

    if q_error is not None:
        q_labels = list(q_comp_errors.keys())
        summary_lines.extend(
            [
                f"  HPROM-ANN vs PROM-ANN: Rel. q_m Error = {q_error:.4e}",
                f"    q_m component relative errors: {_format_component_errors(q_comp_errors, q_labels)}",
                f"    q_m RMSE / max |difference|: {q_rmse:.4e} / {q_max_abs:.4e}",
            ]
        )
    else:
        missing = []
        if q_prom is None:
            missing.append("PROM q history")
        if q_hprom is None:
            missing.append("HPROM q history")
        summary_lines.append(f"  q_m PROM/HPROM comparison skipped (missing {', '.join(missing)}).")

    for method, t in timings.items():
        summary_lines.append(f"  {method} time: {t:.2f}s")

    summary_path = os.path.join(out_dir, "stage10_error_summary.txt")
    summary_lines.append(f"  Error summary saved: {summary_path}")
    for line in summary_lines:
        print(line)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines).lstrip() + "\n")

    plot_hprom_ann_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 10: HPROM-ANN benchmark")
    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument("--run-prom-ann", action="store_true", help="Force PROM-ANN recompute.")
    p.add_argument("--run-hprom-ann", action="store_true", help="Force HPROM-ANN recompute.")
    p.add_argument(
        "--ann-data-dir",
        type=str,
        default="stage_7_ann_data",
        help="Directory with ANN data/model files for PROM/HPROM-ANN.",
    )
    p.add_argument(
        "--hprom-ann-dir",
        type=str,
        default="stage_9_hprom_ann_data",
        help="Directory with HPROM-ANN ECM file (ecm_weights_all.npz).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="stage_10_hprom_ann_results",
        help="Output directory for Stage 10 ANN benchmark.",
    )
    p.add_argument(
        "--hprom-homogenization-mode",
        type=str,
        default="ecm_fixed",
        choices=["ecm_fixed", "maw_dynamic", "maw", "maw_separate"],
        help="HPROM-ANN homogenization weights: fixed classical ECM or dynamic MAW-ECM eps/sig.",
    )
    p.add_argument(
        "--hprom-maw-hom-eval-mode",
        type=str,
        default="model",
        choices=["model", "nearest", "oracle"],
        help=(
            "How to evaluate dynamic MAW homogenization weights. 'model' uses the saved "
            "ANN/RBF regressor; 'nearest'/'oracle' uses the exact stored training weights "
            "of the nearest MAW training state. Fixed-classic components are unaffected."
        ),
    )
    p.add_argument(
        "--hprom-corrector-iters",
        type=int,
        default=25,
        help="Maximum HPROM-ANN Newton/corrector iterations. Use 0 to evaluate the direct ANN prediction only.",
    )
    p.add_argument(
        "--qp-init-mode",
        type=str,
        default="continuation",
        choices=["continuation", "previous", "zero", "mu_affine"],
        help=(
            "Initial q_m for HPROM-ANN. For --hprom-corrector-iters 0, use "
            "'mu_affine' to evaluate the direct mu-to-q_m prediction at every step."
        ),
    )
    args = p.parse_args()

    run_stage10(
        run_fom=args.run_fom,
        run_prom_ann=args.run_prom_ann,
        run_hprom_ann=args.run_hprom_ann,
        ann_data_dir=args.ann_data_dir,
        hprom_ann_dir=args.hprom_ann_dir,
        out_dir=args.out_dir,
        hprom_homogenization_mode=args.hprom_homogenization_mode,
        hprom_maw_hom_eval_mode=args.hprom_maw_hom_eval_mode,
        hprom_corrector_iters=args.hprom_corrector_iters,
        qp_init_mode=args.qp_init_mode,
    )
