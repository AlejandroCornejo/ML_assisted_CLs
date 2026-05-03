import argparse

import stage8_test_prom_rbf as stage8_base
from prom_rbf_solver_rve_ls import LoadPromRbfModel, RunPromRbfBatchSimulation


# Rebind solver hooks so Stage 8 runs against LS-RBF defaults.
stage8_base.LoadPromRbfModel = LoadPromRbfModel
stage8_base.RunPromRbfBatchSimulation = RunPromRbfBatchSimulation


def run_stage8_rbf_ls(
    compare_baselines=True,
    run_fom=False,
    run_hprom=False,
    plot_only=False,
    use_old_stiffness_in_first_iteration=True,
    rbf_data_dir="stage_7_rbf_data_ls",
    out_dir="stage_8_prom_rbf_ls_results",
    use_stage6_waypoints=True,
):
    return stage8_base.run_stage8_rbf(
        compare_baselines=compare_baselines,
        run_fom=run_fom,
        run_hprom=run_hprom,
        plot_only=plot_only,
        use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
        rbf_data_dir=rbf_data_dir,
        out_dir=out_dir,
        use_stage6_waypoints=use_stage6_waypoints,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8: PROM-RBF LS benchmark")
    parser.add_argument("--no-compare", action="store_true", help="Run PROM-RBF only (no FOM/HPROM baselines).")
    parser.add_argument("--run-fom", action="store_true", help="Force re-run local Stage 8 FOM baseline.")
    parser.add_argument("--run-hprom", action="store_true", help="Force re-run local Stage 8 HPROM baseline.")
    parser.add_argument("--plot-only", action="store_true", help="Only generate trajectory plot and step report.")
    parser.add_argument(
        "--rbf-data-dir",
        type=str,
        default="stage_7_rbf_data_ls",
        help="Directory with rbf_model.npz, phi_p.npy, phi_s.npy.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="stage_8_prom_rbf_ls_results",
        help="Output directory for Stage 8 RBF-LS results.",
    )
    parser.add_argument(
        "--no-old-stiffness-first-it",
        action="store_true",
        help="Disable reuse of previous-step reduced stiffness in Newton iteration 0.",
    )
    parser.add_argument(
        "--control-points-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use-stage6-waypoints",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    run_stage8_rbf_ls(
        compare_baselines=not args.no_compare,
        run_fom=args.run_fom,
        run_hprom=args.run_hprom,
        plot_only=args.plot_only,
        use_old_stiffness_in_first_iteration=not args.no_old_stiffness_first_it,
        rbf_data_dir=args.rbf_data_dir,
        out_dir=args.out_dir,
        use_stage6_waypoints=not args.control_points_only,
    )
