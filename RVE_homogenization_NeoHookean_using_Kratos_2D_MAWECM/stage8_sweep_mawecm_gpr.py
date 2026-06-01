#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run Stage8 MAW-ECM-GPR sweep over multiple minimum support sizes.

Pipeline per support value:
1) stage8b_build_mawecm_res_model_rbf.py
2) stage6c_create_hrom_mdpa.py (inplace update of MAW npz with HROM metadata)
3) stage8_test_hprom_mawecm_gpr_online.py

By default this script stops at first failure (break-on-fail behavior).

Notes:
- This sweep is aligned with the current Stage8b interface
  (local or graph-enabled residual MAW stage).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _parse_supports(txt: str) -> List[int]:
    vals: List[int] = []
    for token in txt.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("supports list is empty")
    return vals


def _support_tag(n: int) -> str:
    return "auto" if int(n) <= 0 else str(int(n))


def _cmd_to_str(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _run_and_log(
    cmd: Sequence[str],
    cwd: str,
    log_file: str,
    env: Dict[str, str],
) -> Tuple[int, str]:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    header = f"$ {_cmd_to_str(cmd)}\n\n"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        out = proc.stdout or ""
        f.write(out)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return int(proc.returncode), out


def _check_hrom_keys(npz_path: str) -> None:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    needed = [
        "hrom_mesh_base",
        "hrom_full_mesh_base",
        "hrom_element_full_indices",
        "hrom_n_elem",
        "w_eps_hrom",
        "w_sig_hrom",
    ]
    missing = [k for k in needed if k not in data.files]
    if missing:
        raise RuntimeError(f"HROM keys missing in {npz_path}: {missing}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage8 MAW-ECM-GPR sweep runner")
    p.add_argument("--supports", type=str, default="100,80,60,40,0")
    p.add_argument("--root-dir", type=str, default=".")

    p.add_argument("--dataset-dir", type=str, default="stage_8a_mawecm_res_dataset")
    p.add_argument("--maw-mode", type=str, default="res_only", choices=["res_only", "res_eps", "res_eps_sig"])
    p.add_argument("--hom-source", type=str, default="full_mesh", choices=["full_mesh", "fixed_ecm"])
    p.add_argument("--fixed-ecm-file", type=str, default="stage_6b_hprom_ecm/ecm_weights_all.npz")
    p.add_argument("--res-bootstrap-ecm-file", type=str, default="")
    p.add_argument("--max-number-zeros-active-set-loop-maw-ecm", type=int, default=1)
    p.add_argument("--use-global-graph-2ndstage", type=int, default=0, choices=[0, 1])
    p.add_argument("--smooth-laplacian-all-iterations", type=int, default=0, choices=[0, 1])
    p.add_argument("--alpha-smooth", type=float, default=1.0e4)
    p.add_argument("--graph-mode", type=str, default="structured_grid", choices=["structured_grid", "knn"])
    p.add_argument("--graph-knn", type=int, default=8)
    p.add_argument("--graph-kernel", type=str, default="gaussian", choices=["gaussian", "binary"])
    p.add_argument("--graph-sigma", type=float, default=-1.0)
    p.add_argument("--graph-cell-weight-mode", type=str, default="binary", choices=["binary"])
    p.add_argument("--maw-min-support-size-res", type=int, default=-1)
    p.add_argument("--maw-min-support-size-eps", type=int, default=0)
    p.add_argument("--maw-min-support-size-sig", type=int, default=0)
    p.add_argument("--maw-hom-conservative", type=int, default=1, choices=[0, 1])
    p.add_argument("--maw-res-enforce-nonnegativity", type=int, default=1, choices=[0, 1])
    p.add_argument("--maw-hom-enforce-nonnegativity", type=int, default=0, choices=[0, 1])
    p.add_argument("--save-weight-field-plots", type=int, default=1, choices=[0, 1])
    p.add_argument("--show-weight-field-plots", type=int, default=0, choices=[0, 1])
    p.add_argument("--max-weight-field-plots", type=int, default=0)
    p.add_argument("--weight-plot-format", type=str, default="png")
    p.add_argument("--enforce-sum-weights", type=int, default=0, choices=[0, 1])
    p.add_argument("--res-bootstrap-constrain-sum-weights", type=int, default=1, choices=[0, 1])
    p.add_argument("--sum-weights-target", type=float, default=-1.0)
    p.add_argument("--rbf-clip-nonnegative", type=int, default=1, choices=[0, 1])
    p.add_argument("--rbf-renorm", type=int, default=1, choices=[0, 1])

    p.add_argument("--stage8b-prefix", type=str, default="stage_8b_hprom_mawecm_res_rbf_")

    p.add_argument("--base-mesh", type=str, default="rve_geometry")
    p.add_argument("--selection-key", type=str, default="Z_union", choices=["Z_res", "Z_union", "Z_eps", "Z_sig"])
    p.add_argument("--condition-mode", type=str, default="all", choices=["all", "selected_nodes", "none"])
    p.add_argument("--hrom-mesh-prefix", type=str, default="rve_geometry_stage8b_maw_")
    p.add_argument("--save-extra-selection-images", type=int, default=1, choices=[0, 1])
    p.add_argument("--extra-selection-keys", type=str, default="Z_res,Z_eps,Z_sig,Z_union")
    p.add_argument(
        "--hrom-min-dirichlet-nodes",
        type=int,
        default=0,
        help="If >0, stage6c augments HROM selection to cover at least this many dirichlet nodes.",
    )
    p.add_argument(
        "--hrom-dirichlet-submodelpart",
        type=str,
        default="dirichlet",
        help="Dirichlet submodelpart name used by stage6c coverage augmentation.",
    )

    p.add_argument("--stage8-online-prefix", type=str, default="stage_8_online_hprom_mawecm_gpr_")
    p.add_argument("--run-fom", type=int, default=1, choices=[0, 1])
    p.add_argument("--run-prom-gpr", type=int, default=1, choices=[0, 1])
    p.add_argument("--run-hprom-mawecm-gpr", type=int, default=1, choices=[0, 1])
    p.add_argument("--hprom-homogenization-mode", type=str, default="ecm_separate", choices=["full_fom", "ecm_separate"])
    p.add_argument("--hprom-use-hrom-mdpa", type=int, default=1, choices=[0, 1])
    p.add_argument("--hprom-hrom-strict", type=int, default=1, choices=[0, 1])
    p.add_argument("--hprom-update-maw-each-iter", type=int, default=1, choices=[0, 1])
    p.add_argument("--hprom-include-weight-tangent", type=int, default=1, choices=[0, 1])
    p.add_argument("--hprom-clip-nonnegative", type=int, default=1, choices=[0, 1])
    p.add_argument("--hprom-renorm-weights", type=int, default=0, choices=[0, 1])
    p.add_argument("--hprom-sync-modelpart-each-newton-iter", type=int, default=0, choices=[0, 1])
    p.add_argument("--hprom-call-entity-hooks-each-newton-iter", type=int, default=0, choices=[0, 1])
    p.add_argument("--hprom-use-analysis-stage-solution-step-hooks", type=int, default=0, choices=[0, 1])
    p.add_argument("--hprom-fail-on-nonconvergence", type=int, default=1, choices=[0, 1])
    p.add_argument("--save-plots", type=int, default=1, choices=[0, 1])

    p.add_argument("--stop-on-fail", type=int, default=1, choices=[0, 1])
    p.add_argument("--logs-root", type=str, default="stage_8_sweep_logs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.use_global_graph_2ndstage) == 1 and int(args.smooth_laplacian_all_iterations) == 1:
        raise SystemExit(
            "Invalid sweep setup: --use-global-graph-2ndstage 1 requires "
            "--smooth-laplacian-all-iterations 0 (phase-1 first, then graph stage-2)."
        )
    root_dir = os.path.abspath(args.root_dir)
    supports = _parse_supports(args.supports)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_root = os.path.join(root_dir, args.logs_root, stamp)
    os.makedirs(logs_root, exist_ok=True)

    env = os.environ.copy()
    # Avoid Matplotlib cache warnings and speed up repeated imports.
    env.setdefault("MPLCONFIGDIR", os.path.join(root_dir, ".mplconfig"))
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)

    py = sys.executable
    stop_on_fail = bool(int(args.stop_on_fail))
    summary = []

    print("=" * 78)
    print("Stage8 MAW-ECM-GPR sweep")
    print("=" * 78)
    print(f"root_dir    : {root_dir}")
    print(f"supports    : {supports} (0 => auto)")
    print(f"logs_root   : {logs_root}")
    print(f"maw_mode    : {args.maw_mode}")
    print(f"hom_mode    : {args.hprom_homogenization_mode}")
    print(f"hom_source8b: {args.hom_source}")
    print(f"graph_stage : {int(args.use_global_graph_2ndstage)}")
    print(f"sum(w) local: {int(args.enforce_sum_weights)}")
    print(f"sum(w) boot : {int(args.res_bootstrap_constrain_sum_weights)}")
    print(f"stop_on_fail: {int(stop_on_fail)}")
    if int(args.hprom_use_hrom_mdpa) != 1 or int(args.hprom_hrom_strict) != 1:
        print(
            "[INFO] Legacy online options (--hprom-use-hrom-mdpa, --hprom-hrom-strict) are ignored "
            "by current strict Stage8 online script."
        )

    for s in supports:
        tag = _support_tag(s)
        print("\n" + "-" * 78)
        print(f"[CASE {tag}] starting")
        print("-" * 78)

        case_logs = os.path.join(logs_root, f"case_{tag}")
        os.makedirs(case_logs, exist_ok=True)

        stage8b_out = os.path.join(root_dir, f"{args.stage8b_prefix}{tag}")
        mawecm_npz = os.path.join(stage8b_out, "ecm_weights_all.npz")

        stage8b_cmd = [
            py,
            "stage8b_build_mawecm_res_model_rbf.py",
            "--dataset-dir",
            args.dataset_dir,
            "--maw-mode",
            str(args.maw_mode),
            "--hom-source",
            str(args.hom_source),
            "--max-number-zeros-active-set-loop-maw-ecm",
            str(int(args.max_number_zeros_active_set_loop_maw_ecm)),
            "--use-global-graph-2ndstage",
            str(int(args.use_global_graph_2ndstage)),
            "--smooth-laplacian-all-iterations",
            str(int(args.smooth_laplacian_all_iterations)),
            "--alpha-smooth",
            str(float(args.alpha_smooth)),
            "--graph-mode",
            str(args.graph_mode),
            "--graph-knn",
            str(int(args.graph_knn)),
            "--graph-kernel",
            str(args.graph_kernel),
            "--graph-sigma",
            str(float(args.graph_sigma)),
            "--graph-cell-weight-mode",
            str(args.graph_cell_weight_mode),
            "--maw-min-support-size",
            str(int(s)),
            "--maw-min-support-size-res",
            str(int(args.maw_min_support_size_res)),
            "--maw-min-support-size-eps",
            str(int(args.maw_min_support_size_eps)),
            "--maw-min-support-size-sig",
            str(int(args.maw_min_support_size_sig)),
            "--maw-hom-conservative",
            str(int(args.maw_hom_conservative)),
            "--maw-res-enforce-nonnegativity",
            str(int(args.maw_res_enforce_nonnegativity)),
            "--maw-hom-enforce-nonnegativity",
            str(int(args.maw_hom_enforce_nonnegativity)),
            "--save-weight-field-plots",
            str(int(args.save_weight_field_plots)),
            "--show-weight-field-plots",
            str(int(args.show_weight_field_plots)),
            "--max-weight-field-plots",
            str(int(args.max_weight_field_plots)),
            "--weight-plot-format",
            args.weight_plot_format,
            "--enforce-sum-weights",
            str(int(args.enforce_sum_weights)),
            "--res-bootstrap-constrain-sum-weights",
            str(int(args.res_bootstrap_constrain_sum_weights)),
            "--sum-weights-target",
            str(float(args.sum_weights_target)),
            "--rbf-clip-nonnegative",
            str(int(args.rbf_clip_nonnegative)),
            "--rbf-renorm",
            str(int(args.rbf_renorm)),
            "--out-dir",
            f"{args.stage8b_prefix}{tag}",
        ]
        if str(args.hom_source).strip().lower() == "fixed_ecm":
            stage8b_cmd.extend(["--fixed-ecm-file", str(args.fixed_ecm_file)])
        if str(args.res_bootstrap_ecm_file).strip():
            stage8b_cmd.extend(["--res-bootstrap-ecm-file", str(args.res_bootstrap_ecm_file).strip()])

        rc, _ = _run_and_log(stage8b_cmd, root_dir, os.path.join(case_logs, "stage8b.log"), env)
        if rc != 0 or (not os.path.exists(mawecm_npz)):
            summary.append({"case": tag, "status": "FAILED", "failed_step": "stage8b", "return_code": rc})
            print(f"[CASE {tag}] FAILED at stage8b (rc={rc}).")
            if stop_on_fail:
                break
            continue

        stage6c_cmd = [
            py,
            "stage6c_create_hrom_mdpa.py",
            "--base-mesh",
            args.base_mesh,
            "--ecm-file",
            os.path.relpath(mawecm_npz, root_dir),
            "--selection-key",
            args.selection_key,
            "--condition-mode",
            args.condition_mode,
            "--output-mesh",
            f"{args.hrom_mesh_prefix}{tag}_hrom",
            "--inplace-ecm",
            "--save-selection-image",
            f"{args.stage8b_prefix}{tag}/Z_union_selected_elements.png",
            "--save-extra-selection-images",
            str(int(args.save_extra_selection_images)),
            "--extra-selection-keys",
            str(args.extra_selection_keys),
            "--model-label",
            f"HPROM-MAWECM-GPR-{tag}",
            "--min-dirichlet-nodes",
            str(int(args.hrom_min_dirichlet_nodes)),
            "--dirichlet-submodelpart",
            str(args.hrom_dirichlet_submodelpart),
        ]
        rc, _ = _run_and_log(stage6c_cmd, root_dir, os.path.join(case_logs, "stage6c.log"), env)
        if rc != 0:
            summary.append({"case": tag, "status": "FAILED", "failed_step": "stage6c", "return_code": rc})
            print(f"[CASE {tag}] FAILED at stage6c (rc={rc}).")
            if stop_on_fail:
                break
            continue
        try:
            _check_hrom_keys(mawecm_npz)
        except Exception as exc:
            summary.append({"case": tag, "status": "FAILED", "failed_step": "stage6c_postcheck", "error": str(exc)})
            print(f"[CASE {tag}] FAILED post-check stage6c: {exc}")
            if stop_on_fail:
                break
            continue

        stage8online_out = os.path.join(root_dir, f"{args.stage8_online_prefix}{tag}")
        stage8online_cmd = [
            py,
            "stage8_test_hprom_mawecm_gpr_online.py",
            "--mawecm-file",
            os.path.relpath(mawecm_npz, root_dir),
            "--hprom-update-maw-each-iter",
            str(int(args.hprom_update_maw_each_iter)),
            "--hprom-include-weight-tangent",
            str(int(args.hprom_include_weight_tangent)),
            "--hprom-homogenization-mode",
            str(args.hprom_homogenization_mode),
            "--hprom-clip-nonnegative",
            str(int(args.hprom_clip_nonnegative)),
            "--hprom-renorm-weights",
            str(int(args.hprom_renorm_weights)),
            "--hprom-sync-modelpart-each-newton-iter",
            str(int(args.hprom_sync_modelpart_each_newton_iter)),
            "--hprom-call-entity-hooks-each-newton-iter",
            str(int(args.hprom_call_entity_hooks_each_newton_iter)),
            "--hprom-use-analysis-stage-solution-step-hooks",
            str(int(args.hprom_use_analysis_stage_solution_step_hooks)),
            "--hprom-fail-on-nonconvergence",
            str(int(args.hprom_fail_on_nonconvergence)),
            "--save-plots",
            str(int(args.save_plots)),
            "--out-dir",
            f"{args.stage8_online_prefix}{tag}",
        ]
        if int(args.run_fom):
            stage8online_cmd.append("--run-fom")
        if int(args.run_prom_gpr):
            stage8online_cmd.append("--run-prom-gpr")
        if int(args.run_hprom_mawecm_gpr):
            stage8online_cmd.append("--run-hprom-mawecm-gpr")

        rc, _ = _run_and_log(stage8online_cmd, root_dir, os.path.join(case_logs, "stage8_online.log"), env)
        summary_file = os.path.join(stage8online_out, "stage8_online_summary.json")
        if rc != 0 or (not os.path.exists(summary_file)):
            summary.append({"case": tag, "status": "FAILED", "failed_step": "stage8_online", "return_code": rc})
            print(f"[CASE {tag}] FAILED at stage8 online (rc={rc}).")
            if stop_on_fail:
                break
            continue

        summary.append(
            {
                "case": tag,
                "status": "OK",
                "stage8b_out": os.path.relpath(stage8b_out, root_dir),
                "stage8_online_out": os.path.relpath(stage8online_out, root_dir),
                "summary_file": os.path.relpath(summary_file, root_dir),
            }
        )
        print(f"[CASE {tag}] OK")

    print("\n" + "=" * 78)
    print("Sweep summary")
    print("=" * 78)
    for item in summary:
        if item.get("status") == "OK":
            print(f"[OK]   case={item['case']} summary={item['summary_file']}")
        else:
            print(f"[FAIL] case={item.get('case')} step={item.get('failed_step')} detail={item.get('error', item.get('return_code'))}")

    out_json = os.path.join(logs_root, "sweep_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved sweep summary: {out_json}")

    if stop_on_fail and any(x.get("status") != "OK" for x in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
