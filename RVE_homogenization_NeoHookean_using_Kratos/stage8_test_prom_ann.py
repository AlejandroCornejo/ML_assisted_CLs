import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
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
from prom_ann_solver_rve import LoadPromAnnModel, RunPromAnnBatchSimulation


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




def _load_rom_model(model_dir="stage_2_pod_rve"):
    phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(model_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(model_dir, "eq_map.npy"))
    Xc, Yc = np.load(os.path.join(model_dir, "domain_center.npy"))
    return phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc


def _run_or_load_fom(
    parameters,
    strain_path,
    out_dir,
    emax,
    reference_steps,
    force_run=False,
    use_old_stiffness_in_first_iteration=True,
):
    fom_sig_file = os.path.join(out_dir, "fom_stress.npy")
    fom_eps_file = os.path.join(out_dir, "fom_strain.npy")

    if force_run or not (os.path.exists(fom_sig_file) and os.path.exists(fom_eps_file)):
        print("  [Stage 8] Running local FOM baseline...")
        eps, sig = RunFomBatchSimulation(
            parameters,
            out_dir=out_dir,
            strain_path=strain_path,
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
        print("  [Stage 8] Loading local FOM baseline cache.")
        eps = np.load(fom_eps_file)
        sig = np.load(fom_sig_file)

    return eps, sig


def _run_or_load_hprom(
    parameters,
    strain_path,
    out_dir,
    emax,
    reference_steps,
    force_run=False,
    use_old_stiffness_in_first_iteration=True,
):
    hprom_sig_file = os.path.join(out_dir, "hprom_stress.npy")
    hprom_eps_file = os.path.join(out_dir, "hprom_strain.npy")

    if force_run or not (os.path.exists(hprom_sig_file) and os.path.exists(hprom_eps_file)):
        print("  [Stage 8] Running local HPROM baseline...")
        phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc = _load_rom_model("stage_2_pod_rve")
        ecm = np.load(os.path.join("stage_5_hprom_data", "ecm_weights_all.npz"))
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
            strain_path=strain_path,
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
        print("  [Stage 8] Loading local HPROM baseline cache.")
        eps = np.load(hprom_eps_file)
        sig = np.load(hprom_sig_file)

    return eps, sig


def run_stage8(
    compare_baselines=True,
    run_fom=False,
    run_hprom=False,
    plot_only=False,
    use_old_stiffness_in_first_iteration=True,
    use_stage6_waypoints=True,
):
    out_dir = "stage_8_prom_ann_results"
    os.makedirs(out_dir, exist_ok=True)

    # Stage 4/6-aligned benchmark trajectory
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
    print("  Stage 8: PROM-ANN Manifold Benchmark")
    print("=" * 60)
    print(f"  Waypoints: {len(control_points)}")
    print(f"  Strain path mode: {'stage6_waypoints' if use_stage6_waypoints else 'control_points'}")
    print(f"  Strain path points used in solve: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  Reference increment level: {REFERENCE_STEPS_FOR_UNIT_AMPLITUDE}")
    print(f"  Segments: {seg_steps}")

    if plot_only:
        print("[Stage 8] --plot-only enabled. Skipping solves.")
        return

    # PROM-ANN run
    phi_p, phi_s, free_dofs, _, _, ann_model, device, include_macro_strain_input = LoadPromAnnModel()
    parameters = setup_kratos_parameters("rve_geometry")

    t0 = time.perf_counter()
    eps_pa, sig_pa = RunPromAnnBatchSimulation(
        parameters,
        phi_p,
        phi_s,
        free_dofs,
        ann_model,
        device,
        strain_path,
        reference_amplitude=emax,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        include_macro_strain_input=include_macro_strain_input,
        use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
    )
    t_pa = time.perf_counter() - t0

    np.save(os.path.join(out_dir, "prom_ann_stress.npy"), sig_pa)
    np.save(os.path.join(out_dir, "prom_ann_strain.npy"), eps_pa)
    print(f"  PROM-ANN done in {t_pa:.1f}s")

    if not compare_baselines:
        print("\n" + "=" * 40)
        print("  STAGE 8 COMPLETE (PROM-ANN ONLY)")
        print("=" * 40)
        print(f"  PROM-ANN Wall-time: {t_pa:.1f}s")
        print(f"  Check output directory: {out_dir}/")
        return

    # Local baselines
    fom_eps, fom_sig = _run_or_load_fom(
        parameters,
        strain_path,
        out_dir,
        emax,
        REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        force_run=run_fom,
        use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
    )
    hprom_eps, hprom_sig = _run_or_load_hprom(
        parameters,
        strain_path,
        out_dir,
        emax,
        REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        force_run=run_hprom,
        use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
    )

    # Comparison and plots
    n = min(len(fom_sig), len(sig_pa), len(hprom_sig), len(fom_eps), len(eps_pa), len(hprom_eps))

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
        plt.plot(eps_pa[:n, i], sig_pa[:n, i], "r--", label="PROM-ANN", linewidth=1.5)
        plt.plot(hprom_eps[:n, i], hprom_sig[:n, i], "b:", label="HPROM", linewidth=1.5)

        plt.title(f"Manifold Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"manifold_accuracy_{suffix}.png"), dpi=200)
        plt.close()

    fom_eps_eq, fom_sig_eq = _compute_equivalent_stress_strain(fom_eps[:n], fom_sig[:n])
    pa_eps_eq, pa_sig_eq = _compute_equivalent_stress_strain(eps_pa[:n], sig_pa[:n])
    hprom_eps_eq, hprom_sig_eq = _compute_equivalent_stress_strain(hprom_eps[:n], hprom_sig[:n])

    plt.figure(figsize=(7, 6))
    plt.plot(fom_eps_eq, fom_sig_eq, "k-", label="FOM", linewidth=2.0)
    plt.plot(pa_eps_eq, pa_sig_eq, "r--", label="PROM-ANN", linewidth=1.5)
    plt.plot(hprom_eps_eq, hprom_sig_eq, "b:", label="HPROM", linewidth=1.5)
    plt.title(r"Manifold Benchmark: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "manifold_accuracy_equivalent.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    fom_norm = np.linalg.norm(fom_sig[:n], axis=1) + 1e-30
    err_hprom = np.linalg.norm(fom_sig[:n] - hprom_sig[:n], axis=1) / fom_norm
    err_pa = np.linalg.norm(fom_sig[:n] - sig_pa[:n], axis=1) / fom_norm

    plt.plot(err_pa, "r-", label="PROM-ANN Error", linewidth=1.5)
    plt.plot(err_hprom, "b-", label="HPROM Error", linewidth=1.5)

    plt.title("Relative Stress Error Comparison")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "manifold_error_history.png"), dpi=200)
    plt.close()

    print("\n" + "=" * 40)
    print("  STAGE 8 COMPLETE")
    print("=" * 40)
    print(f"  PROM-ANN Wall-time: {t_pa:.1f}s")
    print(f"  Max PROM-ANN Error: {np.max(err_pa):.2e}")
    print(f"  Avg PROM-ANN Error: {np.mean(err_pa):.2e}")
    print(f"  Check output directory: {out_dir}/ for plots.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 8: PROM-ANN benchmark")
    p.add_argument("--no-compare", action="store_true", help="Run PROM-ANN only (no FOM/HPROM baselines).")
    p.add_argument("--run-fom", action="store_true", help="Force re-run local Stage 8 FOM baseline.")
    p.add_argument("--run-hprom", action="store_true", help="Force re-run local Stage 8 HPROM baseline.")
    p.add_argument("--plot-only", action="store_true", help="Only generate trajectory plot and step report.")
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

    run_stage8(
        compare_baselines=not args.no_compare,
        run_fom=args.run_fom,
        run_hprom=args.run_hprom,
        plot_only=args.plot_only,
        use_old_stiffness_in_first_iteration=not args.no_old_stiffness_first_it,
        use_stage6_waypoints=not args.control_points_only,
    )
