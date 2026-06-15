#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import stage8_test_prom_ann as stage8_base


def run_stage8_ann_ls(
    compare_baselines=True,
    run_fom=False,
    run_hprom=False,
    compare_hprom=False,
    plot_only=False,
    use_old_stiffness_in_first_iteration=True,
    ann_data_dir="stage_7_ann_data_ls",
    hprom_data_dir="stage_9_hprom_ann_data_ls",
    out_dir="stage_8_prom_ann_ls_results",
    use_stage6_waypoints=True,
    qp_init_mode="previous",
    max_its=25,
    normalized_dq_cutoff=1.0e-4,
    fail_on_nonconvergence=True,
    baseline_cache_dir="",
):
    return stage8_base.run_stage8(
        compare_baselines=compare_baselines,
        run_fom=run_fom,
        run_hprom=run_hprom,
        compare_hprom=compare_hprom,
        plot_only=plot_only,
        use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
        ann_data_dir=ann_data_dir,
        hprom_data_dir=hprom_data_dir,
        out_dir=out_dir,
        use_stage6_waypoints=use_stage6_waypoints,
        qp_init_mode=qp_init_mode,
        max_its=max_its,
        normalized_dq_cutoff=normalized_dq_cutoff,
        fail_on_nonconvergence=fail_on_nonconvergence,
        baseline_cache_dir=baseline_cache_dir,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 8: PROM-ANN-LS benchmark")
    p.add_argument("--no-compare", action="store_true", help="Run PROM-ANN-LS only (no FOM/HPROM baselines).")
    p.add_argument("--run-fom", action="store_true", help="Force re-run local Stage 8 FOM baseline.")
    p.add_argument("--run-hprom", action="store_true", help="Force re-run local Stage 8 HPROM baseline.")
    p.add_argument("--compare-hprom", action="store_true", help="Include the traditional HPROM baseline.")
    p.add_argument("--plot-only", action="store_true", help="Only generate trajectory plot and step report.")
    p.add_argument(
        "--no-old-stiffness-first-it",
        action="store_true",
        help="Disable reuse of previous-step reduced stiffness in Newton iteration 0.",
    )
    p.add_argument("--ann-data-dir", type=str, default="stage_7_ann_data_ls", help="ANN-LS model/data directory.")
    p.add_argument(
        "--hprom-data-dir",
        type=str,
        default="stage_9_hprom_ann_data_ls",
        help="HPROM-ANN-LS ECM directory (ecm_weights_all.npz).",
    )
    p.add_argument("--out-dir", type=str, default="stage_8_prom_ann_ls_results", help="Output directory.")
    p.add_argument("--baseline-cache-dir", type=str, default="", help="Optional FOM cache directory.")
    p.add_argument(
        "--qp-init-mode",
        type=str,
        default="previous",
        choices=["previous", "zero", "mu_affine"],
    )
    p.add_argument("--max-its", type=int, default=25)
    p.add_argument(
        "--normalized-dq-cutoff",
        type=float,
        default=1.0e-4,
        help=(
            "Converge when ||D_q^-1 delta_q_m|| falls below this value, where "
            "D_q contains the ANN training standard deviations."
        ),
    )
    p.add_argument(
        "--fail-on-nonconvergence",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "1 raises when --max-its is exhausted; 0 keeps the best finite "
            "iterate and continues to the next increment."
        ),
    )
    p.add_argument("--control-points-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--use-stage6-waypoints", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()

    run_stage8_ann_ls(
        compare_baselines=not a.no_compare,
        run_fom=a.run_fom,
        run_hprom=a.run_hprom,
        compare_hprom=a.compare_hprom,
        plot_only=a.plot_only,
        use_old_stiffness_in_first_iteration=not a.no_old_stiffness_first_it,
        ann_data_dir=a.ann_data_dir,
        hprom_data_dir=a.hprom_data_dir,
        out_dir=a.out_dir,
        use_stage6_waypoints=not a.control_points_only,
        qp_init_mode=a.qp_init_mode,
        max_its=a.max_its,
        normalized_dq_cutoff=a.normalized_dq_cutoff,
        fail_on_nonconvergence=bool(a.fail_on_nonconvergence),
        baseline_cache_dir=a.baseline_cache_dir,
    )
