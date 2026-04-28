#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 1 (Kratos): run the stage-0 trajectory bundle with kratos_solver_rve.py."""

import argparse
import os
import re
import subprocess
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 Kratos training launcher")
    p.add_argument(
        "--stage0-file",
        type=str,
        default="stage_0_trajectory/stage_0_trajectories.npz",
        help="Path to stage-0 trajectory bundle (.npz).",
    )
    p.add_argument(
        "--which",
        type=str,
        default="all",
        help='Which trajectory to run ("all", "both", or a specific index like "1").',
    )
    p.add_argument("--mesh", type=str, default="rve_geometry", help="Mesh base name.")
    p.add_argument("--out-dir", type=str, default="stage_1_training_set_kratos", help="Output root directory.")

    p.add_argument("--sanitize-mdpa", action="store_true", help="Pass --sanitize-mdpa to kratos solver.")
    p.add_argument("--ref-steps", type=int, default=None, help="Override reference steps.")
    p.add_argument("--reference-amplitude", type=float, default=None, help="Override reference amplitude.")
    p.add_argument("--young-mpa", type=float, default=None, help="Override Young modulus in MPa.")
    p.add_argument("--poisson", type=float, default=None, help="Override Poisson ratio.")
    return p.parse_args()


def discover_indices_from_npz(path):
    data = np.load(path)
    idx = []
    pat = re.compile(r"^trajectory_(\d+)$")
    for k in data.keys():
        m = pat.match(str(k))
        if m:
            idx.append(int(m.group(1)))
    idx = sorted(set(idx))
    if len(idx) == 0:
        raise ValueError(f"No trajectory_<i> keys found in: {path}")
    return idx


def build_indices(which, stage0_file):
    if which == "all":
        return discover_indices_from_npz(stage0_file)
    if which == "both":
        return [1, 2]
    return [int(which)]


def run_single(args, idx, workdir):
    out_dir = os.path.join(args.out_dir, f"trajectory_{idx}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python3",
        "kratos_solver_rve.py",
        "--mesh",
        str(args.mesh),
        "--strain-waypoints-file",
        str(args.stage0_file),
        "--trajectory-index",
        str(idx),
        "--out-dir",
        out_dir,
    ]

    if args.sanitize_mdpa:
        cmd.append("--sanitize-mdpa")
    if args.ref_steps is not None:
        cmd += ["--ref-steps", str(int(args.ref_steps))]
    if args.reference_amplitude is not None:
        cmd += ["--reference-amplitude", str(float(args.reference_amplitude))]
    if args.young_mpa is not None:
        cmd += ["--young-mpa", str(float(args.young_mpa))]
    if args.poisson is not None:
        cmd += ["--poisson", str(float(args.poisson))]

    print("[STAGE1-KRATOS] Running:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=workdir)
    if result.returncode != 0:
        raise RuntimeError(
            f"Trajectory {idx} FAILED (exit code {result.returncode}). "
            f"Aborting Stage 1. Try increasing --ref-steps for smaller strain increments."
        )


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    stage0_file_abs = args.stage0_file
    if not os.path.isabs(stage0_file_abs):
        stage0_file_abs = os.path.join(script_dir, stage0_file_abs)
    if not os.path.exists(stage0_file_abs):
        raise FileNotFoundError(f"stage0 trajectory file not found: {stage0_file_abs}")

    args.stage0_file = stage0_file_abs
    indices = build_indices(args.which, args.stage0_file)
    indices = indices[::-1]  # hardest (highest Gxy) first → fail fast

    print(f"[STAGE1-KRATOS] Starting batch of {len(indices)} trajectories...")
    for i, idx in enumerate(indices, 1):
        print(f"\n{'='*60}")
        print(f"[STAGE1-KRATOS] Trajectory {idx}  ({i}/{len(indices)})")
        print(f"{'='*60}")
        try:
            run_single(args, idx, script_dir)
        except RuntimeError as e:
            print(f"\n[FATAL] {e}")
            print(f"[FATAL] Completed {i-1}/{len(indices)} trajectories before failure.")
            raise SystemExit(1)

    print(f"\n[STAGE1-KRATOS] All {len(indices)} trajectories completed successfully.")


if __name__ == "__main__":
    main()
