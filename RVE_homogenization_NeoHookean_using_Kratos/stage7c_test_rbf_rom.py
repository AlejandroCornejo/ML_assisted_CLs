#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 7c (RBF): pure reconstruction check on one training trajectory.
"""

from stage7c_reconstruction_check import run_stage7c_reconstruction_check


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 7c (RBF): pure reconstruction check")
    p.add_argument("--trajectory-index", type=int, default=1, help="Training trajectory index for reconstruction check.")
    p.add_argument("--fom-dir", type=str, default="stage_1_training_set_fom", help="Stage 1 directory.")
    p.add_argument("--basis-dir", type=str, default="stage_2_pod_rve", help="Stage 2 POD directory.")
    p.add_argument("--ann-data-dir", type=str, default="stage_7_ann_data", help="ANN data/model directory.")
    p.add_argument("--rbf-data-dir", type=str, default="stage_7_rbf_data", help="RBF data/model directory.")
    p.add_argument("--out-dir", type=str, default="stage_7c_reconstruction_results", help="Output directory.")
    args = p.parse_args()

    run_stage7c_reconstruction_check(
        model_type="rbf",
        trajectory_index=args.trajectory_index,
        fom_dir=args.fom_dir,
        basis_dir=args.basis_dir,
        ann_data_dir=args.ann_data_dir,
        rbf_data_dir=args.rbf_data_dir,
        out_dir=args.out_dir,
    )
