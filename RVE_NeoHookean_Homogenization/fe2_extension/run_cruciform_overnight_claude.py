#!/usr/bin/env python3
"""Batch: the cruciform biaxial benchmark at the validated smoke-test
resolution (n_body=6, n_arm_len=4 -- confirmed conforming: n_body must be
divisible by 6 under arm_width_fraction=2/3, see build_cruciform_mesh_
claude.py's own alignment check), safer/validated loading
(arm_width_fraction=2/3, delta=1.2, giving median|gamma12|~0.007, only
~7% of Gauss points above 0.1 -- comfortably inside/near the RVE's own
training envelope), run through the 4 cheap PANN tiers first (fast,
sequential, sanity-check-by-consistency) then D-HPROM-ANN (the 10-element
reaction-force rule). Explicit user choice this round: n_body=6, not 12
(the earlier production resolution) or above, and stop after
D-HPROM-ANN -- HPROM-ANN (iterative) and the true FOM-nested reference
are both deliberately held out of this run.

Everything runs sequentially in this one process -- never concurrently --
per this project's own standing serial-only-benchmarking rule.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

N_BODY, N_ARM_LEN = 6, 4
DELTA = 1.2
N_STEPS = 20
DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"


def main():
    import dhprom_ann_direct_law_float64_claude as dhprom_f64_module
    from run_cruciform_fe2_claude import run_newton_fe2_cruciform

    results = {}

    # HPROM-ANN's own validation check (n_body=6 scale) already ran and
    # diverged on step 20/20 only (the most extreme load step; 19/20 steps
    # converged to ~1e-9 residuals) -- its memoryless, cold-started-every-
    # call Newton correction couldn't resolve the hardest corner states at
    # the full target deformation. D-HPROM-ANN has no such risk (no
    # iterative correction at all, pure closed-form decode) and already
    # passed cleanly twice (n_body=6 alone, and alongside all 4 PANN tiers)
    # -- proceeding directly to the production batch with D-HPROM-ANN,
    # per the user's own explicit choice for this run ("ir a la segura").

    print(f"=== PANN tiers (n_body={N_BODY}, n_arm_len={N_ARM_LEN}, delta={DELTA}, n_steps={N_STEPS}) ===", flush=True)
    for which in ("pann_regression", "pann_free", "pann_certified", "pann_ickan"):
        t0 = time.time()
        res = run_newton_fe2_cruciform(
            which, n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
            delta_x_final=DELTA, delta_y_final=DELTA, verbose=False, use_line_search=True, save_npz=True,
        )
        results[which] = res
        d = np.load(HERE / f"cruciform_results_{which}_claude.npz")
        g12 = np.abs(d["e_gp"][:, 2])
        print(f"[{which}] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
              f"E11={res['e11_range']}, E22={res['e22_range']}, g12={res['g12_range']}, "
              f"median|g12|={np.median(g12):.4f}, frac>0.1={np.mean(g12 > 0.1):.3f}, "
              f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)

    print(f"\n=== D-HPROM-ANN-FE2 (10-elem reaction-force rule), same config ===", flush=True)
    assert dhprom_f64_module._DEFAULT_LAW_F64 is None
    dhprom_f64_module.get_law_float64(hprom_ann_dir=str(DHPROMANN_DIR))
    t0 = time.time()
    res = run_newton_fe2_cruciform(
        "dhprom_f64_consistent", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
        delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
    )
    results["dhprom_f64_consistent"] = res
    d = np.load(HERE / "cruciform_results_dhprom_f64_consistent_claude.npz")
    g12 = np.abs(d["e_gp"][:, 2])
    print(f"[dhprom_f64_consistent] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"E11={res['e11_range']}, E22={res['e22_range']}, g12={res['g12_range']}, "
          f"median|g12|={np.median(g12):.4f}, frac>0.1={np.mean(g12 > 0.1):.3f}, "
          f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)

    print("\n=== OVERNIGHT BATCH SUMMARY ===", flush=True)
    for which, res in results.items():
        print(f"  {which:20s}: fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
              f"wall={res['wall_time']:.1f}s", flush=True)
    print("OVERNIGHT_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
