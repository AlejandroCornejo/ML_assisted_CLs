#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import stage10_test_hprom_ann as stage10_base


def run_stage10_ann_ls(
    run_fom=False,
    run_prom_ann=False,
    run_hprom_ann=False,
    ann_data_dir="stage_7_ann_model_ls_newton",
    hprom_ann_dir="stage_9_hprom_ann_data_ls_independent_sum990",
    out_dir="stage_10_hprom_ann_ls_results_independent_sum990",
    hprom_homogenization_mode="ecm_fixed",
):
    return stage10_base.run_stage10(
        run_fom=run_fom,
        run_prom_ann=run_prom_ann,
        run_hprom_ann=run_hprom_ann,
        ann_data_dir=ann_data_dir,
        hprom_ann_dir=hprom_ann_dir,
        out_dir=out_dir,
        hprom_homogenization_mode=hprom_homogenization_mode,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 10: HPROM-ANN-LS benchmark")
    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument("--run-prom-ann", action="store_true", help="Force PROM-ANN recompute.")
    p.add_argument("--run-hprom-ann", action="store_true", help="Force HPROM-ANN recompute.")
    p.add_argument("--ann-data-dir", type=str, default="stage_7_ann_model_ls_newton", help="ANN-LS model/data directory.")
    p.add_argument(
        "--hprom-ann-dir",
        type=str,
        default="stage_9_hprom_ann_data_ls_independent_sum990",
        help="Directory with HPROM-ANN-LS ECM file (ecm_weights_all.npz).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="stage_10_hprom_ann_ls_results_independent_sum990",
        help="Output directory for Stage 10 ANN-LS benchmark.",
    )
    p.add_argument(
        "--hprom-homogenization-mode",
        type=str,
        default="ecm_fixed",
        choices=["ecm_fixed", "maw_dynamic", "maw", "maw_separate"],
        help="HPROM-ANN-LS homogenization weights: fixed classical ECM or dynamic MAW-ECM eps/sig.",
    )
    a = p.parse_args()

    run_stage10_ann_ls(
        run_fom=a.run_fom,
        run_prom_ann=a.run_prom_ann,
        run_hprom_ann=a.run_hprom_ann,
        ann_data_dir=a.ann_data_dir,
        hprom_ann_dir=a.hprom_ann_dir,
        out_dir=a.out_dir,
        hprom_homogenization_mode=a.hprom_homogenization_mode,
    )
