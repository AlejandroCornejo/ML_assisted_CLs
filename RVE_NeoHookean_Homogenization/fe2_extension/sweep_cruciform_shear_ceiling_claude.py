#!/usr/bin/env python3
"""Find the equibiaxial delta at which the Cruciform's reentrant-corner peak
shear (gamma12, Voigt engineering shear at the 8 symmetric arm-body-junction
Gauss points) first exceeds the RVE's own trained shear envelope (+-0.1,
from trajectories/stage_0_trajectory/stage_0_trajectories.npz's real
relative_boundary). Uses ICNN (pann_certified) as the macro law: fast, and
robust at every delta tested all session, so it isolates the geometry effect
without any material-law robustness confound. save_npz=False throughout --
this must never touch cruciform_results_pann_certified_claude.npz, the file
backing the paper's own Table 8."""
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

import run_cruciform_fe2_claude as rc  # noqa: E402

WHICH = "pann_certified"
DELTAS = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 3.0, 3.6, 4.2, 4.8]
SHEAR_TRAINED_LIMIT = 0.1


def main():
    print(f"=== Cruciform shear-ceiling sweep, material={WHICH}, n_body=6 ===", flush=True)
    rows = []
    for delta in DELTAS:
        t0 = time.time()
        try:
            res = rc.run_newton_fe2_cruciform(
                WHICH, n_body=6, n_arm_len=4, n_steps=20,
                delta_x_final=delta, delta_y_final=delta,
                verbose=False, use_line_search=True, save_npz=False,
            )
            g12_lo, g12_hi = res["g12_range"]
            e11_lo, e11_hi = res["e11_range"]
            e22_lo, e22_hi = res["e22_range"]
            peak_abs_g12 = max(abs(g12_lo), abs(g12_hi))
            print(f"  delta={delta:5.2f}  E11=[{e11_lo:+.4f},{e11_hi:+.4f}]  "
                  f"E22=[{e22_lo:+.4f},{e22_hi:+.4f}]  g12=[{g12_lo:+.4f},{g12_hi:+.4f}]  "
                  f"peak|g12|={peak_abs_g12:.4f}  fully_converged={res['fully_converged']}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
            rows.append((delta, e11_hi, e22_hi, peak_abs_g12, res["fully_converged"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  delta={delta:5.2f}  CRASHED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}",
                  flush=True)
            rows.append((delta, None, None, None, False))

    print("\n=== Shear-ceiling crossing ===", flush=True)
    prev = None
    for delta, e11_hi, e22_hi, peak_g12, conv in rows:
        if peak_g12 is None:
            continue
        if peak_g12 > SHEAR_TRAINED_LIMIT and (prev is None or prev[3] <= SHEAR_TRAINED_LIMIT):
            print(f"  crosses {SHEAR_TRAINED_LIMIT} between delta={prev[0] if prev else '?'} "
                  f"(peak|g12|={prev[3] if prev else float('nan'):.4f}) and delta={delta} "
                  f"(peak|g12|={peak_g12:.4f})", flush=True)
        prev = (delta, e11_hi, e22_hi, peak_g12, conv)

    print("SHEAR_CEILING_SWEEP_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
