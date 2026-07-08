#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import stage10_test_hprom_ann as stage10_base


def run_stage10_ann_ls(
    run_fom=False,
    run_prom_ann=False,
    run_hprom_ann=False,
    run_hprom_ann_direct=False,
    ann_data_dir="stage_7_ann_model_ls_newton",
    hprom_ann_dir="stage_9_hprom_ann_data_ls_independent_sum990",
    out_dir="stage_10_hprom_ann_ls_results_independent_sum990",
    hprom_homogenization_mode="ecm_fixed",
    hprom_maw_hom_eval_mode="model",
    hprom_corrector_iters=25,
    hprom_include_manifold_curvature=1,
    qp_init_mode="continuation",
):
    if int(hprom_corrector_iters) == 0 and str(qp_init_mode).strip().lower() != "mu_affine":
        print(
            "  [Stage10-LS] Direct HPROM-ANN mode requested "
            "(--hprom-corrector-iters 0): forcing qp_init_mode='mu_affine'."
        )
        qp_init_mode = "mu_affine"
    return stage10_base.run_stage10(
        run_fom=run_fom,
        run_prom_ann=run_prom_ann,
        run_hprom_ann=run_hprom_ann,
        run_hprom_ann_direct=run_hprom_ann_direct,
        ann_data_dir=ann_data_dir,
        hprom_ann_dir=hprom_ann_dir,
        out_dir=out_dir,
        hprom_homogenization_mode=hprom_homogenization_mode,
        hprom_maw_hom_eval_mode=hprom_maw_hom_eval_mode,
        hprom_corrector_iters=hprom_corrector_iters,
        hprom_include_manifold_curvature=hprom_include_manifold_curvature,
        qp_init_mode=qp_init_mode,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 10: HPROM-ANN-LS benchmark")
    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument("--run-prom-ann", action="store_true", help="Force PROM-ANN recompute.")
    p.add_argument("--run-hprom-ann", action="store_true", help="Force HPROM-ANN recompute.")
    p.add_argument(
        "--run-hprom-ann-direct",
        action="store_true",
        help="Also run D-HPROM-ANN-LS: direct ANN prediction with no Newton/corrector iterations.",
    )
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
        help="Maximum HPROM-ANN-LS Newton/corrector iterations. Use 0 to evaluate the direct ANN prediction only.",
    )
    p.add_argument(
        "--hprom-include-manifold-curvature",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "Include the exact nonlinear-manifold Hessian curvature term in the HPROM-ANN-LS Newton tangent. "
            "Use 0 for a faster Gauss-Newton/quasi-Newton tangent."
        ),
    )
    p.add_argument(
        "--qp-init-mode",
        type=str,
        default="continuation",
        choices=["continuation", "previous", "zero", "mu_affine"],
        help=(
            "Initial q_m for HPROM-ANN-LS. For --hprom-corrector-iters 0, use "
            "'mu_affine' to evaluate the direct mu-to-q_m prediction at every step."
        ),
    )
    a = p.parse_args()

    run_stage10_ann_ls(
        run_fom=a.run_fom,
        run_prom_ann=a.run_prom_ann,
        run_hprom_ann=a.run_hprom_ann,
        run_hprom_ann_direct=a.run_hprom_ann_direct,
        ann_data_dir=a.ann_data_dir,
        hprom_ann_dir=a.hprom_ann_dir,
        out_dir=a.out_dir,
        hprom_homogenization_mode=a.hprom_homogenization_mode,
        hprom_maw_hom_eval_mode=a.hprom_maw_hom_eval_mode,
        hprom_corrector_iters=a.hprom_corrector_iters,
        hprom_include_manifold_curvature=a.hprom_include_manifold_curvature,
        qp_init_mode=a.qp_init_mode,
    )
