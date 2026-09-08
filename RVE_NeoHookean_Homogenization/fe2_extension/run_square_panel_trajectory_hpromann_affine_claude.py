#!/usr/bin/env python3
"""Run the purpose-built square-panel trajectory on HPROM-ANN's "mu_affine"
(never-continued/memoryless) variant, contrasting against the continuation
wrapper's own crash on this exact trajectory: isolates whether the failure
comes from bad warm-starting or persists even with zero call-to-call
memory."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.chdir(str(HERE))

N_WORKERS = 16
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

import hprom_ann_law_parallel_affine_claude as hprom_affine_module  # noqa: E402
hprom_affine_module.ensure_persistent_executor(n_workers=N_WORKERS)

from run_square_panel_trajectory_claude import run_trajectory  # noqa: E402
import run_cruciform_fe2_claude as rc  # noqa: E402

which = "hprom_ann_parallel_affine"
rc.MATERIAL_FUNCS[which] = hprom_affine_module.HpromAnnParallelAffineWrapper(
    hprom_ann_dir=str(HPROMANN_DIR), n_workers=N_WORKERS,
)


def main():
    print(f"=== {which}, square panel, purpose-built trajectory ===", flush=True)
    t0 = time.time()
    try:
        res = run_trajectory(which, verbose=True)
        print(f"[{which}] fully_converged={res['fully_converged']}  ever_diverged={res['ever_diverged']}  "
              f"wall={res['wall_time']:.1f}s  ({time.time() - t0:.1f}s elapsed)", flush=True)
        n_stalled = sum(1 for s in res["step_log"] if s["status"] == "stalled")
        n_diverged = sum(1 for s in res["step_log"] if s["status"] == "diverged")
        print(f"[{which}] {n_stalled} stalled, {n_diverged} diverged (of {len(res['step_log'])})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{which}] CRASHED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)

    print("HPROMANN_AFFINE_TRAJECTORY_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
