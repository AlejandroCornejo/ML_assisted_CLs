#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Focused profiler: FOM vs HPROM on a short trajectory.

Outputs:
- raw solver logs
- cProfile summaries
- timing/iteration summary (txt + npz)

This is intended to diagnose why HPROM can be slower than FOM.
"""

import argparse
import cProfile
import io
import os
import re
import time
import pstats
import numpy as np
from contextlib import redirect_stdout

from stage6_test_hprom import generate_safe_test_path

from fom_solver_rve import (
    setup_kratos_parameters,
    RunFomBatchSimulation,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from hprom_solver_rve import (
    RunHpromBatchSimulation,
)


def _count_step_and_iterations(log_text, step_tag):
    n_steps = len(re.findall(rf"^\[{re.escape(step_tag)}\]\s+Step", log_text, flags=re.MULTILINE))
    n_iters = len(re.findall(r"^\s*> It\s+\d+", log_text, flags=re.MULTILINE))
    return n_steps, n_iters


def _parse_hprom_internal_timing(log_text):
    m = re.search(
        r"\[HPROM\]\s+Timing:\s+assembly=([0-9eE+\-.]+)s,\s+solve=([0-9eE+\-.]+)s",
        log_text,
    )
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def _profile_run(label, run_callable, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    prof = cProfile.Profile()
    log_buf = io.StringIO()

    t0 = time.perf_counter()
    with redirect_stdout(log_buf):
        prof.enable()
        eps, sig = run_callable()
        prof.disable()
    elapsed = time.perf_counter() - t0

    log_text = log_buf.getvalue()
    with open(os.path.join(out_dir, f"{label.lower()}_log.txt"), "w", encoding="utf-8") as f:
        f.write(log_text)

    stats_stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stats_stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(120)
    with open(os.path.join(out_dir, f"{label.lower()}_cprofile.txt"), "w", encoding="utf-8") as f:
        f.write(stats_stream.getvalue())

    if label == "FOM":
        n_steps, n_iters = _count_step_and_iterations(log_text, "FOM")
        asm_t, solve_t = np.nan, np.nan
    else:
        n_steps, n_iters = _count_step_and_iterations(log_text, "HPROM")
        asm_t, solve_t = _parse_hprom_internal_timing(log_text)

    return {
        "elapsed": float(elapsed),
        "n_steps": int(n_steps),
        "n_iterations": int(n_iters),
        "avg_iter_per_step": float(n_iters / max(n_steps, 1)),
        "hprom_assembly_time": float(asm_t),
        "hprom_solve_time": float(solve_t),
        "eps": np.asarray(eps, dtype=float),
        "sig": np.asarray(sig, dtype=float),
    }


def _load_hprom_model_and_data(
    model_dir="stage_2_pod_rve",
    hprom_dir="stage_5_hprom_data",
):
    phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(model_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(model_dir, "eq_map.npy"))
    Xc, Yc = np.load(os.path.join(model_dir, "domain_center.npy"))
    ecm = np.load(os.path.join(hprom_dir, "ecm_weights_all.npz"))
    ecm_data = {k: ecm[k] for k in ecm.files}
    return phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc, ecm_data


def main():
    p = argparse.ArgumentParser(description="Profile FOM vs HPROM on a short path.")
    p.add_argument("--out-dir", default="profile_fom_vs_hprom_results")
    p.add_argument(
        "--ref-steps",
        type=int,
        default=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        help="Dynamic-step reference (default: same as Stage 11/FOM setting).",
    )
    p.add_argument(
        "--segment-index",
        type=int,
        default=0,
        help="Segment index from the Stage 11 path when profiling a single segment.",
    )
    p.add_argument(
        "--full-path",
        action="store_true",
        help="Profile full Stage 11 path instead of a single segment.",
    )
    p.add_argument("--max-newton-it", type=int, default=20)
    p.add_argument("--use-bundle-ref-amp", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"
    ref_amp_bundle = emax
    bundle_path = "stage_0_trajectory/stage_0_trajectories.npz"
    if os.path.exists(bundle_path):
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        if "emax" in data:
            emax = float(np.ravel(data["emax"])[0])
        else:
            emax = float(np.ravel(data["reference_amplitude"])[0])
        if "reference_amplitude" in data:
            ref_amp_bundle = float(np.ravel(data["reference_amplitude"])[0])
        else:
            ref_amp_bundle = float(emax)
        if "domain_type" in data:
            domain_type = str(np.ravel(data["domain_type"])[0])

    reference_amplitude = float(ref_amp_bundle if args.use_bundle_ref_amp else emax)

    control_pts, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    waypoints = np.asarray(waypoints, dtype=float)
    if args.full_path:
        strain_path = waypoints
        path_mode = "full-path"
        seg_idx = -1
    else:
        n_seg = int(len(waypoints) - 1)
        if n_seg <= 0:
            raise RuntimeError("Invalid Stage 11 path: less than 2 waypoints.")
        seg_idx = int(max(0, min(args.segment_index, n_seg - 1)))
        strain_path = waypoints[seg_idx : seg_idx + 2, :].copy()
        path_mode = "single-segment"

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=int(args.ref_steps),
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    total_steps = int(np.sum(seg_steps))

    parameters = setup_kratos_parameters("rve_geometry")

    phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc, ecm_data = _load_hprom_model_and_data(
        model_dir="stage_2_pod_rve",
        hprom_dir="stage_5_hprom_data",
    )

    def _run_fom():
        return RunFomBatchSimulation(
            parameters,
            out_dir=out_dir,
            save_plot=False,
            strain_path=strain_path,
            trajectory_index=None,
            reference_amplitude=reference_amplitude,
            reference_steps=int(args.ref_steps),
            max_newton_it=int(args.max_newton_it),
        )

    def _run_hprom():
        return RunHpromBatchSimulation(
            parameters,
            phi_f,
            free_dofs,
            dir_dofs,
            eq_map,
            Xc,
            Yc,
            ecm_data,
            out_dir=out_dir,
            save_plot=False,
            strain_path=strain_path,
            trajectory_index=None,
            reference_amplitude=reference_amplitude,
            reference_steps=int(args.ref_steps),
            max_newton_it=int(args.max_newton_it),
        )

    print("[Profiler] Running FOM...")
    res_fom = _profile_run("FOM", _run_fom, out_dir)
    print("[Profiler] Running HPROM...")
    res_hpr = _profile_run("HPROM", _run_hprom, out_dir)

    # Relative stress mismatch on common length
    n = min(len(res_fom["sig"]), len(res_hpr["sig"]))
    rel_err = np.linalg.norm(res_hpr["sig"][:n] - res_fom["sig"][:n]) / (np.linalg.norm(res_fom["sig"][:n]) + 1e-30)

    summary_lines = [
        "=" * 72,
        "FOM vs HPROM (short-path profile)",
        "=" * 72,
        f"Path mode: {path_mode}",
        f"Segment index: {seg_idx}",
        f"Path points used: {len(strain_path)}",
        f"Dynamic steps on profiled path: {total_steps}",
        f"Reference amplitude used: {reference_amplitude:.6e}",
        f"Ref steps: {int(args.ref_steps)}",
        f"Max Newton iterations: {int(args.max_newton_it)}",
        "",
        f"FOM wall time [s]: {res_fom['elapsed']:.6f}",
        f"FOM steps: {res_fom['n_steps']}",
        f"FOM total Newton iters: {res_fom['n_iterations']}",
        f"FOM avg iters/step: {res_fom['avg_iter_per_step']:.6f}",
        "",
        f"HPROM wall time [s]: {res_hpr['elapsed']:.6f}",
        f"HPROM steps: {res_hpr['n_steps']}",
        f"HPROM total Newton iters: {res_hpr['n_iterations']}",
        f"HPROM avg iters/step: {res_hpr['avg_iter_per_step']:.6f}",
        f"HPROM reported assembly [s]: {res_hpr['hprom_assembly_time']:.6f}",
        f"HPROM reported reduced solve [s]: {res_hpr['hprom_solve_time']:.6f}",
        "",
        f"HPROM/FOMVEC wall ratio: {res_hpr['elapsed'] / max(res_fom['elapsed'], 1e-30):.6f}",
        f"Rel. stress error (HPROM vs FOMVEC): {rel_err:.6e} (n={n})",
        "",
        f"Saved logs: {os.path.join(out_dir, 'fom_log.txt')}, {os.path.join(out_dir, 'hprom_log.txt')}",
        f"Saved profiles: {os.path.join(out_dir, 'fom_cprofile.txt')}, {os.path.join(out_dir, 'hprom_cprofile.txt')}",
    ]

    summary_txt = "\n".join(summary_lines)
    print(summary_txt)
    with open(os.path.join(out_dir, "profile_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_txt + "\n")

    np.savez(
        os.path.join(out_dir, "profile_summary.npz"),
        fom_wall=np.array([res_fom["elapsed"]], dtype=float),
        hprom_wall=np.array([res_hpr["elapsed"]], dtype=float),
        fom_steps=np.array([res_fom["n_steps"]], dtype=int),
        hprom_steps=np.array([res_hpr["n_steps"]], dtype=int),
        fom_iters=np.array([res_fom["n_iterations"]], dtype=int),
        hprom_iters=np.array([res_hpr["n_iterations"]], dtype=int),
        hprom_assembly=np.array([res_hpr["hprom_assembly_time"]], dtype=float),
        hprom_solve=np.array([res_hpr["hprom_solve_time"]], dtype=float),
        rel_err=np.array([rel_err], dtype=float),
    )


if __name__ == "__main__":
    main()
