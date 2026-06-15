import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from stage6_test_hprom import generate_safe_test_path
from stage4_test_rve import plot_path_in_domain
from fom_solver_rve import (
    setup_kratos_parameters,
    RunFomBatchSimulation,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from hprom_solver_rve import RunHpromBatchSimulation
from prom_gpr_solver_rve import LoadPromGprModel, RunPromGprBatchSimulation


def _load_hrom_mesh_base_from_ecm_file(ecm_file):
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


def _save_prom_only_plots(eps_pr, sig_pr, out_dir):
    eps_pr = np.asarray(eps_pr, dtype=float)
    sig_pr = np.asarray(sig_pr, dtype=float)
    n = min(len(eps_pr), len(sig_pr))
    if n == 0:
        return

    comp_labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "xy"),
    ]

    for i, label_sig, label_eps, suffix in comp_labels:
        plt.figure(figsize=(7, 6))
        plt.plot(eps_pr[:n, i], sig_pr[:n, i], "r--", label="PROM-GPR", linewidth=1.8)
        plt.title(f"PROM-GPR Manifold: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"prom_gpr_only_{suffix}.png"), dpi=200)
        plt.close()

    pr_eps_eq, pr_sig_eq = _compute_equivalent_stress_strain(eps_pr[:n], sig_pr[:n])
    plt.figure(figsize=(7, 6))
    plt.plot(pr_eps_eq, pr_sig_eq, "r--", label="PROM-GPR", linewidth=1.8)
    plt.title(r"PROM-GPR Manifold: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prom_gpr_only_equivalent.png"), dpi=200)
    plt.close()




def _load_rom_model(model_dir="stage_2_pod_rve"):
    phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(model_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(model_dir, "eq_map.npy"))
    Xc, Yc = np.load(os.path.join(model_dir, "domain_center.npy"))
    return phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc


def _run_or_load_fom(
    parameters,
    control_points,
    out_dir,
    emax,
    reference_steps,
    force_run=False,
    use_old_stiffness_in_first_iteration=True,
):
    fom_sig_file = os.path.join(out_dir, "fom_stress.npy")
    fom_eps_file = os.path.join(out_dir, "fom_strain.npy")

    if force_run or not (os.path.exists(fom_sig_file) and os.path.exists(fom_eps_file)):
        print("  [Stage 8-GPR] Running local FOM baseline...")
        eps, sig = RunFomBatchSimulation(
            parameters,
            out_dir=out_dir,
            strain_path=control_points,
            trajectory_index=None,
            save_plot=False,
            reference_amplitude=emax,
            reference_steps=reference_steps,
            use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
        )
        eps = np.array(eps) if not isinstance(eps, np.ndarray) else eps
        sig = np.array(sig) if not isinstance(sig, np.ndarray) else sig
        np.save(fom_eps_file, eps)
        np.save(fom_sig_file, sig)
    else:
        print("  [Stage 8-GPR] Loading local FOM baseline cache.")
        eps = np.load(fom_eps_file)
        sig = np.load(fom_sig_file)

    return eps, sig


def _run_or_load_hprom(
    parameters,
    control_points,
    out_dir,
    emax,
    reference_steps,
    ecm_file,
    force_run=False,
    use_old_stiffness_in_first_iteration=True,
):
    hprom_sig_file = os.path.join(out_dir, "hprom_stress.npy")
    hprom_eps_file = os.path.join(out_dir, "hprom_strain.npy")

    if force_run or not (os.path.exists(hprom_sig_file) and os.path.exists(hprom_eps_file)):
        print("  [Stage 8-GPR] Running local HPROM baseline...")
        phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc = _load_rom_model("stage_2_pod_rve")
        ecm = np.load(ecm_file, allow_pickle=True)
        ecm_data = {k: ecm[k] for k in ecm.files}

        eps, sig = RunHpromBatchSimulation(
            parameters,
            phi_f,
            free_dofs,
            dir_dofs,
            eq_map,
            Xc,
            Yc,
            ecm_data=ecm_data,
            out_dir=out_dir,
            strain_path=control_points,
            trajectory_index=None,
            save_plot=False,
            reference_amplitude=emax,
            reference_steps=reference_steps,
            use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
        )
        eps = np.array(eps) if not isinstance(eps, np.ndarray) else eps
        sig = np.array(sig) if not isinstance(sig, np.ndarray) else sig
        np.save(hprom_eps_file, eps)
        np.save(hprom_sig_file, sig)
    else:
        print("  [Stage 8-GPR] Loading local HPROM baseline cache.")
        eps = np.load(hprom_eps_file)
        sig = np.load(hprom_sig_file)

    return eps, sig


def run_stage8_gpr(
    compare_baselines=True,
    run_fom=False,
    run_hprom=False,
    compare_hprom=False,
    plot_only=False,
    use_old_stiffness_in_first_iteration=True,
    gpr_data_dir="stage_7_gpr_data",
    hprom_data_dir="stage_5_hprom_data",
    out_dir="stage_8_prom_gpr_results",
    use_stage6_waypoints=True,
    qp_init_mode="previous",
    max_its=25,
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

    control_points, full_path = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(full_path if use_stage6_waypoints else control_points, dtype=float)

    plot_file = os.path.join(out_dir, "stage8_benchmark_path_multi_view.png")
    plot_path_in_domain(control_points, full_path, emax, rel6, domain_type, plot_file)

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 60)
    print("  Stage 8: PROM-GPR Manifold Benchmark")
    print("=" * 60)
    print(f"  Waypoints: {len(control_points)}")
    print(f"  Strain path mode: {'stage6_waypoints' if use_stage6_waypoints else 'control_points'}")
    print(f"  Strain path points used in solve: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  Reference increment level: {REFERENCE_STEPS_FOR_UNIT_AMPLITUDE}")
    print(f"  Segments: {seg_steps}")
    print(f"  GPR data dir: {gpr_data_dir}")
    include_hprom = bool(compare_hprom or run_hprom)
    if include_hprom:
        print(f"  HPROM data dir: {hprom_data_dir}")

    if plot_only:
        print("[Stage 8-GPR] --plot-only enabled. Skipping solves.")
        return

    ecm_file = os.path.join(hprom_data_dir, "ecm_weights_all.npz")
    mesh_fom_prom = "rve_geometry"
    mesh_hprom = "rve_geometry"
    if include_hprom:
        auto_hrom_mesh = _load_hrom_mesh_base_from_ecm_file(ecm_file)
        if auto_hrom_mesh:
            mesh_hprom = str(auto_hrom_mesh)
    print(f"  FOM/PROM mesh: {mesh_fom_prom}")
    if include_hprom:
        print(f"  HPROM mesh:    {mesh_hprom}")

    phi_p, phi_s, free_dofs, _, _, gpr_model, _include_macro_strain_input = LoadPromGprModel(
        basis_dir="stage_2_pod_rve", gpr_data_dir=gpr_data_dir
    )
    parameters = setup_kratos_parameters(mesh_fom_prom)
    parameters_hprom = setup_kratos_parameters(mesh_hprom) if include_hprom else None

    t0 = time.perf_counter()
    eps_pr, sig_pr = RunPromGprBatchSimulation(
        parameters,
        phi_p,
        phi_s,
        free_dofs,
        gpr_model,
        strain_path,
        out_dir=out_dir,
        reference_amplitude=emax,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
        qp_init_mode=qp_init_mode,
        max_its=max_its,
    )
    t_pr = time.perf_counter() - t0

    np.save(os.path.join(out_dir, "prom_gpr_stress.npy"), sig_pr)
    np.save(os.path.join(out_dir, "prom_gpr_strain.npy"), eps_pr)
    print(f"  PROM-GPR done in {t_pr:.1f}s")

    if not compare_baselines:
        _save_prom_only_plots(eps_pr, sig_pr, out_dir)
        print("\n" + "=" * 40)
        print("  STAGE 8 COMPLETE (PROM-GPR ONLY)")
        print("=" * 40)
        print(f"  PROM-GPR Wall-time: {t_pr:.1f}s")
        print(f"  PROM-only plots saved to: {out_dir}/")
        return

    fom_eps, fom_sig = _run_or_load_fom(
        parameters,
        strain_path,
        out_dir,
        emax,
        REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        force_run=run_fom,
        use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
    )
    if include_hprom:
        hprom_eps, hprom_sig = _run_or_load_hprom(
            parameters_hprom,
            strain_path,
            out_dir,
            emax,
            REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            ecm_file,
            force_run=run_hprom,
            use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
        )
    else:
        hprom_eps = None
        hprom_sig = None

    n = min(len(fom_sig), len(sig_pr), len(fom_eps), len(eps_pr))
    if include_hprom:
        n = min(n, len(hprom_sig), len(hprom_eps))

    plt.rcParams.update(
        {
            "font.size": 12,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    comp_labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "xy"),
    ]

    for i, label_sig, label_eps, suffix in comp_labels:
        plt.figure(figsize=(7, 6))
        plt.plot(fom_eps[:n, i], fom_sig[:n, i], "k-", label="FOM", linewidth=2.0)
        plt.plot(eps_pr[:n, i], sig_pr[:n, i], "r--", label="PROM-GPR", linewidth=1.5)
        if include_hprom:
            plt.plot(hprom_eps[:n, i], hprom_sig[:n, i], "b:", label="HPROM", linewidth=1.5)

        plt.title(f"Manifold Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"gpr_manifold_accuracy_{suffix}.png"), dpi=200)
        plt.close()

    fom_eps_eq, fom_sig_eq = _compute_equivalent_stress_strain(fom_eps[:n], fom_sig[:n])
    pr_eps_eq, pr_sig_eq = _compute_equivalent_stress_strain(eps_pr[:n], sig_pr[:n])

    plt.figure(figsize=(7, 6))
    plt.plot(fom_eps_eq, fom_sig_eq, "k-", label="FOM", linewidth=2.0)
    plt.plot(pr_eps_eq, pr_sig_eq, "r--", label="PROM-GPR", linewidth=1.5)
    if include_hprom:
        hprom_eps_eq, hprom_sig_eq = _compute_equivalent_stress_strain(hprom_eps[:n], hprom_sig[:n])
        plt.plot(hprom_eps_eq, hprom_sig_eq, "b:", label="HPROM", linewidth=1.5)
    plt.title(r"Manifold Benchmark: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gpr_manifold_accuracy_equivalent.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    fom_norm = np.linalg.norm(fom_sig[:n], axis=1)
    diff_pr_norm = np.linalg.norm(fom_sig[:n] - sig_pr[:n], axis=1)
    fom_norm_max = float(np.max(fom_norm)) if fom_norm.size else 0.0
    # Pointwise relative errors are undefined when the reference stress is zero.
    # Use a tiny physical floor only for plotting/history; report the global
    # Frobenius relative error separately as the primary scalar metric.
    rel_floor = max(1e-12 * fom_norm_max, 1e-30)
    err_pr = diff_pr_norm / np.maximum(fom_norm, rel_floor)
    global_rel_pr = float(
        np.linalg.norm(fom_sig[:n] - sig_pr[:n]) / max(np.linalg.norm(fom_sig[:n]), 1e-30)
    )

    plt.plot(err_pr, "r-", label="PROM-GPR Error", linewidth=1.5)
    if include_hprom:
        err_hprom = np.linalg.norm(fom_sig[:n] - hprom_sig[:n], axis=1) / np.maximum(fom_norm, rel_floor)
        plt.plot(err_hprom, "b-", label="HPROM Error", linewidth=1.5)

    plt.title("Relative Stress Error Comparison")
    plt.xlabel("Step")
    plt.ylabel("Pointwise Relative Error [-]")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gpr_manifold_error_history.png"), dpi=200)
    plt.close()

    print("\n" + "=" * 40)
    print("  STAGE 8 COMPLETE")
    print("=" * 40)
    print(f"  PROM-GPR Wall-time: {t_pr:.1f}s")
    print(f"  Global PROM-GPR Stress Rel. Error: {global_rel_pr:.2e}")
    print(f"  Max PROM-GPR Pointwise Error: {np.max(err_pr):.2e}")
    print(f"  Avg PROM-GPR Pointwise Error: {np.mean(err_pr):.2e}")
    print(f"  Pointwise error denominator floor: {rel_floor:.2e}")
    print(f"  Check output directory: {out_dir}/ for plots.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 8: PROM-GPR benchmark")
    p.add_argument("--no-compare", action="store_true", help="Run PROM-GPR only (no FOM/HPROM baselines).")
    p.add_argument("--run-fom", action="store_true", help="Force re-run local Stage 8 FOM baseline.")
    p.add_argument("--run-hprom", action="store_true", help="Force re-run local Stage 8 HPROM baseline.")
    p.add_argument(
        "--compare-hprom",
        action="store_true",
        help="Include HPROM baseline in comparison. Otherwise compare only FOM vs PROM-GPR.",
    )
    p.add_argument("--plot-only", action="store_true", help="Only generate trajectory plot and step report.")
    p.add_argument(
        "--gpr-data-dir",
        type=str,
        default="stage_7_gpr_data",
        help="Directory with sparse_gp_model.npz, phi_p.npy, phi_s.npy.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="stage_8_prom_gpr_results",
        help="Output directory for Stage 8 GPR results.",
    )
    p.add_argument(
        "--hprom-data-dir",
        type=str,
        default="stage_5_hprom_data",
        help="Directory with HPROM ECM file (ecm_weights_all.npz).",
    )
    p.add_argument(
        "--qp-init-mode",
        type=str,
        default="previous",
        choices=["previous", "zero", "mu_affine"],
        help="Initializer for q_p at each increment in PROM-GPR.",
    )
    p.add_argument(
        "--max-its",
        type=int,
        default=25,
        help="Maximum reduced Newton iterations. Use 0 for direct mu-to-q_m prediction.",
    )
    p.add_argument(
        "--no-old-stiffness-first-it",
        action="store_true",
        help="Disable reuse of previous-step reduced stiffness in Newton iteration 0.",
    )
    p.add_argument(
        "--control-points-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--use-stage6-waypoints",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = p.parse_args()

    run_stage8_gpr(
        compare_baselines=not args.no_compare,
        run_fom=args.run_fom,
        run_hprom=args.run_hprom,
        compare_hprom=args.compare_hprom,
        plot_only=args.plot_only,
        use_old_stiffness_in_first_iteration=not args.no_old_stiffness_first_it,
        gpr_data_dir=args.gpr_data_dir,
        hprom_data_dir=args.hprom_data_dir,
        out_dir=args.out_dir,
        use_stage6_waypoints=not args.control_points_only,
        qp_init_mode=args.qp_init_mode,
        max_its=args.max_its,
    )
