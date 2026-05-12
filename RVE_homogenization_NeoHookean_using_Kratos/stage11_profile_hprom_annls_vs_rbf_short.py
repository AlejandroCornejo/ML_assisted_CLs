#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import sys
import numpy as np

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from fom_solver_rve import setup_kratos_parameters
from hprom_ann_solver_rve import LoadHpromAnnModel, RunHpromAnnBatchSimulation
from hprom_rbf_solver_rve import LoadHpromRbfModel, RunHpromRbfBatchSimulation


def _require_hrom_mesh_from_ecm(ecm_data, label):
    if "hrom_mesh_base" not in ecm_data:
        raise RuntimeError(f"[{label}] Missing 'hrom_mesh_base' in ECM data.")
    mesh = str(np.ravel(ecm_data["hrom_mesh_base"])[0])
    if not os.path.exists(mesh + ".mdpa"):
        raise RuntimeError(f"[{label}] HROM mesh file not found: {mesh}.mdpa")
    return mesh


def _safe_div(a, b):
    b = float(b)
    if abs(b) <= 1e-30:
        return np.nan
    return float(a) / b


def main():
    p = argparse.ArgumentParser(description="Short timing profile: HPROM-ANN-LS vs HPROM-RBF")
    p.add_argument("--basis-dir", default="stage_2_pod_rve")
    p.add_argument("--ann-data-dir", default="stage_7_ann_data_ls")
    p.add_argument("--hprom-ann-dir", default="stage_9_hprom_ann_data_ls")
    p.add_argument("--rbf-data-dir", default="stage_7_rbf_data")
    p.add_argument("--hprom-rbf-dir", default="stage_9_hprom_rbf_data")
    p.add_argument("--out-dir", default="stage_11_profile_annls_vs_rbf_short")
    p.add_argument("--target-exx", type=float, default=0.20)
    p.add_argument("--target-eyy", type=float, default=0.00)
    p.add_argument("--target-gxy", type=float, default=0.00)
    p.add_argument("--reference-amplitude", type=float, default=0.20)
    p.add_argument("--reference-steps", type=int, default=80)
    p.add_argument("--max-its", type=int, default=25)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    strain_path = np.array(
        [
            [0.0, 0.0, 0.0],
            [float(args.target_exx), float(args.target_eyy), float(args.target_gxy)],
        ],
        dtype=float,
    )

    print("=" * 78)
    print("Stage11 short profile: HPROM-ANN-LS vs HPROM-RBF")
    print("=" * 78)
    print(f"strain_path: {strain_path.tolist()}")
    print(f"reference_amplitude={args.reference_amplitude:.6e}, reference_steps={args.reference_steps}")

    (
        phi_p_ann,
        phi_s_ann,
        free_ann,
        _dir_ann,
        eq_map_ann,
        Xc_ann,
        Yc_ann,
        ann_model,
        ann_device,
        ecm_ann,
        include_macro_ann,
    ) = LoadHpromAnnModel(
        basis_dir=args.basis_dir,
        ann_data_dir=args.ann_data_dir,
        hprom_ann_dir=args.hprom_ann_dir,
    )
    mesh_ann = _require_hrom_mesh_from_ecm(ecm_ann, "HPROM-ANN-LS")
    params_ann = setup_kratos_parameters(mesh_ann)

    (
        phi_p_rbf,
        phi_s_rbf,
        free_rbf,
        _dir_rbf,
        eq_map_rbf,
        Xc_rbf,
        Yc_rbf,
        rbf_model,
        ecm_rbf,
        include_macro_rbf,
    ) = LoadHpromRbfModel(
        basis_dir=args.basis_dir,
        rbf_data_dir=args.rbf_data_dir,
        hprom_rbf_dir=args.hprom_rbf_dir,
    )
    mesh_rbf = _require_hrom_mesh_from_ecm(ecm_rbf, "HPROM-RBF")
    params_rbf = setup_kratos_parameters(mesh_rbf)

    ann_out = os.path.join(args.out_dir, "ann_ls")
    rbf_out = os.path.join(args.out_dir, "rbf")
    os.makedirs(ann_out, exist_ok=True)
    os.makedirs(rbf_out, exist_ok=True)

    print("\n[Run] HPROM-ANN-LS")
    _eps_ann, _sig_ann, stats_ann = RunHpromAnnBatchSimulation(
        params_ann,
        phi_p_ann,
        phi_s_ann,
        free_ann,
        ann_model,
        ann_device,
        ecm_ann,
        out_dir=ann_out,
        strain_path=strain_path,
        reference_amplitude=float(args.reference_amplitude),
        reference_steps=int(args.reference_steps),
        eq_map_full=eq_map_ann,
        Xc=Xc_ann,
        Yc=Yc_ann,
        max_its=int(args.max_its),
        return_stats=True,
    )

    print("\n[Run] HPROM-RBF")
    _eps_rbf, _sig_rbf, stats_rbf = RunHpromRbfBatchSimulation(
        params_rbf,
        phi_p_rbf,
        phi_s_rbf,
        free_rbf,
        rbf_model,
        ecm_rbf,
        out_dir=rbf_out,
        strain_path=strain_path,
        reference_amplitude=float(args.reference_amplitude),
        reference_steps=int(args.reference_steps),
        eq_map_full=eq_map_rbf,
        Xc=Xc_rbf,
        Yc=Yc_rbf,
        max_its=int(args.max_its),
        return_stats=True,
    )

    summary = {
        "settings": {
            "strain_path": strain_path.tolist(),
            "reference_amplitude": float(args.reference_amplitude),
            "reference_steps": int(args.reference_steps),
            "max_its": int(args.max_its),
            "mesh_ann": mesh_ann,
            "mesh_rbf": mesh_rbf,
        },
        "ann_ls": stats_ann,
        "rbf": stats_rbf,
        "ratios": {
            "total_time_ann_over_rbf": _safe_div(stats_ann["total"], stats_rbf["total"]),
            "iters_ann_over_rbf": _safe_div(stats_ann["newton_iters_total"], stats_rbf["newton_iters_total"]),
            "mean_iter_time_ann_over_rbf": _safe_div(
                stats_ann["mean_time_per_newton_iter"], stats_rbf["mean_time_per_newton_iter"]
            ),
            "mean_map_iter_time_ann_over_rbf": _safe_div(
                stats_ann["mean_map_time_per_newton_iter"], stats_rbf["mean_map_time_per_newton_iter"]
            ),
        },
    }

    out_json = os.path.join(args.out_dir, "timing_summary.json")
    out_txt = os.path.join(args.out_dir, "timing_summary.txt")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("HPROM short timing summary: ANN-LS vs RBF\n")
        for k, v in summary["settings"].items():
            f.write(f"{k}={v}\n")
        f.write("\n[ANN-LS]\n")
        for k, v in stats_ann.items():
            f.write(f"{k}={v}\n")
        f.write("\n[RBF]\n")
        for k, v in stats_rbf.items():
            f.write(f"{k}={v}\n")
        f.write("\n[Ratios]\n")
        for k, v in summary["ratios"].items():
            f.write(f"{k}={v}\n")

    print("\n" + "=" * 78)
    print("Short timing summary")
    print("=" * 78)
    print(f"ANN total: {stats_ann['total']:.3f}s | iters={stats_ann['newton_iters_total']} | mean/iter={stats_ann['mean_time_per_newton_iter']:.6f}s")
    print(f"RBF total: {stats_rbf['total']:.3f}s | iters={stats_rbf['newton_iters_total']} | mean/iter={stats_rbf['mean_time_per_newton_iter']:.6f}s")
    print(
        "Ratios: total={:.3f}, iters={:.3f}, mean_iter_time={:.3f}, mean_map_iter_time={:.3f}".format(
            summary["ratios"]["total_time_ann_over_rbf"],
            summary["ratios"]["iters_ann_over_rbf"],
            summary["ratios"]["mean_iter_time_ann_over_rbf"],
            summary["ratios"]["mean_map_iter_time_ann_over_rbf"],
        )
    )
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
