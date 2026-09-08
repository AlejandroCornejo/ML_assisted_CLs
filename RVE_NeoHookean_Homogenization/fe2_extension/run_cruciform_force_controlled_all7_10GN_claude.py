#!/usr/bin/env python3
"""All 7 non-FOM Cruciform tiers (4 PANN + Linear-HPROM + HPROM-ANN +
D-HPROM-ANN), force-controlled, at a single clean, paper-presentable
load: F_total_final = 1.0e10 N (10 GN) at each tip. Chosen as a round
number close to 8x the reference force (1.2551e9 N, ICNN's own reaction
force at the standard delta=1.2 displacement protocol) -- the level
where Regression first started diverging in the earlier sweep. Every
call wrapped so a hard crash in one model (already seen once, for ICNN
at 10x reference force in an earlier, cruder sweep) does not block
seeing the others' results. Saves each result under cruciform_force_
results_<key>_claude.npz (the force-controlled driver's own default
naming), distinct from every existing displacement-controlled file."""
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

N_STEPS = 20
N_WORKERS = 16
TOTAL_FORCE_FINAL = 1.0e10  # 10 GN, clean round number, ~8x the earlier reference

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

import dhprom_ann_law_parallel_claude as dhprom_par_module  # noqa: E402
dhprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)
import hprom_ann_law_parallel_claude as hprom_par_module  # noqa: E402
hprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)
import linear_hprom_law_parallel_claude as linear_hprom_par_module  # noqa: E402
linear_hprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)

import run_cruciform_fe2_claude as rc  # noqa: E402
from run_cruciform_fe2_force_controlled_claude import run_newton_fe2_cruciform_force  # noqa: E402

which_linear = "linear_hprom_parallel_continuation"
which_hprom_ann = "hprom_ann_parallel_continuation"
which_dhprom = "dhprom_ann_parallel"
rc.MATERIAL_FUNCS[which_linear] = rc.make_linear_hprom_parallel_continuation_material_func(n_workers=N_WORKERS)
rc.MATERIAL_FUNCS[which_hprom_ann] = rc.make_hprom_ann_parallel_continuation_material_func(str(HPROMANN_DIR), n_workers=N_WORKERS)
rc.MATERIAL_FUNCS[which_dhprom] = rc.make_dhprom_ann_parallel_material_func(str(DHPROMANN_DIR), n_workers=N_WORKERS)

MODELS = ("pann_certified", "pann_ickan", "pann_regression", "pann_free",
          which_linear, which_hprom_ann, which_dhprom)


def main():
    results = {}
    for which in MODELS:
        print(f"\n=== {which}, force-controlled, F={TOTAL_FORCE_FINAL:.3e} N (10 GN) ===", flush=True)
        t0 = time.time()
        try:
            res = run_newton_fe2_cruciform_force(
                which, n_body=6, n_arm_len=4, n_steps=N_STEPS, total_force_final=TOTAL_FORCE_FINAL,
                verbose=False, use_line_search=True, save_npz=True,
            )
            last = res["step_log"][-1]
            print(f"[{which}] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
                  f"iters={[s['iters'] for s in res['step_log']]}, "
                  f"status={[s['status'][0] for s in res['step_log']]}, "
                  f"final tip_ux={last['tip_ux_px_mean']:.4f}, tip_uy={last['tip_uy_py_mean']:.4f}, "
                  f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)
            results[which] = res
        except Exception as exc:  # noqa: BLE001
            print(f"[{which}] CRASHED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
            results[which] = None

    print("\n=== SUMMARY: all 7 tiers at F=1.0e10 N (10 GN) ===", flush=True)
    for which in MODELS:
        r = results[which]
        if r is None:
            print(f"  {which:32s}: CRASHED", flush=True)
        else:
            print(f"  {which:32s}: fully_converged={r['fully_converged']}, ever_diverged={r['ever_diverged']}, "
                  f"wall={r['wall_time']:.1f}s", flush=True)
    print("ALL7_10GN_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
