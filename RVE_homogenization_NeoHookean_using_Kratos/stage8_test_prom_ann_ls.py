#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import stage8_test_prom_ann as stage8_base


def run_stage8_ann_ls(
    compare_baselines=True,
    run_fom=False,
    run_hprom=False,
    plot_only=False,
    use_old_stiffness_in_first_iteration=True,
    ann_data_dir="stage_7_ann_data_ls",
    hprom_data_dir="stage_9_hprom_ann_data_ls",
    out_dir="stage_8_prom_ann_ls_results",
    use_stage6_waypoints=True,
):
    return stage8_base.run_stage8(
        compare_baselines=compare_baselines,
        run_fom=run_fom,
        run_hprom=run_hprom,
        plot_only=plot_only,
        use_old_stiffness_in_first_iteration=use_old_stiffness_in_first_iteration,
        ann_data_dir=ann_data_dir,
        hprom_data_dir=hprom_data_dir,
        out_dir=out_dir,
        use_stage6_waypoints=use_stage6_waypoints,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 8: PROM-ANN-LS benchmark")
    p.add_argument("--no-compare", action="store_true", help="Run PROM-ANN-LS only (no FOM/HPROM baselines).")
    p.add_argument("--run-fom", action="store_true", help="Force re-run local Stage 8 FOM baseline.")
    p.add_argument("--run-hprom", action="store_true", help="Force re-run local Stage 8 HPROM baseline.")
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
    p.add_argument("--control-points-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--use-stage6-waypoints", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()

    run_stage8_ann_ls(
        compare_baselines=not a.no_compare,
        run_fom=a.run_fom,
        run_hprom=a.run_hprom,
        plot_only=a.plot_only,
        use_old_stiffness_in_first_iteration=not a.no_old_stiffness_first_it,
        ann_data_dir=a.ann_data_dir,
        hprom_data_dir=a.hprom_data_dir,
        out_dir=a.out_dir,
        use_stage6_waypoints=not a.control_points_only,
    )

