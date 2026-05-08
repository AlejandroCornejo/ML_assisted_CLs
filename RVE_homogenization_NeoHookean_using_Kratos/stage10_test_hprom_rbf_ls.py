#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import stage10_test_hprom_rbf as stage10_base


def run_stage10_rbf_ls(
    run_fom=False,
    run_prom_rbf=False,
    run_hprom_rbf=False,
    rbf_data_dir="stage_7_rbf_data_ls",
    hprom_rbf_dir="stage_9_hprom_rbf_data_ls",
    out_dir="stage_10_hprom_rbf_ls_results",
):
    return stage10_base.run_stage10(
        run_fom=run_fom,
        run_prom_rbf=run_prom_rbf,
        run_hprom_rbf=run_hprom_rbf,
        rbf_data_dir=rbf_data_dir,
        hprom_rbf_dir=hprom_rbf_dir,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 10: HPROM-RBF-LS benchmark")
    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument("--run-prom-rbf", action="store_true", help="Force PROM-RBF recompute.")
    p.add_argument("--run-hprom-rbf", action="store_true", help="Force HPROM-RBF recompute.")
    p.add_argument(
        "--rbf-data-dir",
        type=str,
        default="stage_7_rbf_data_ls",
        help="Directory with LS-RBF model files.",
    )
    p.add_argument(
        "--hprom-rbf-dir",
        type=str,
        default="stage_9_hprom_rbf_data_ls",
        help="Directory with LS HPROM-RBF ECM file.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="stage_10_hprom_rbf_ls_results",
        help="Output directory for Stage 10 LS results.",
    )
    a = p.parse_args()

    run_stage10_rbf_ls(
        run_fom=a.run_fom,
        run_prom_rbf=a.run_prom_rbf,
        run_hprom_rbf=a.run_hprom_rbf,
        rbf_data_dir=a.rbf_data_dir,
        hprom_rbf_dir=a.hprom_rbf_dir,
        out_dir=a.out_dir,
    )
