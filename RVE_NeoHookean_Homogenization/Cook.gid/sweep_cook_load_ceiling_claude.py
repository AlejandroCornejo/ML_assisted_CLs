#!/usr/bin/env python3
"""How far into the RVE's own trained strain box (E11,E22 in [-0.1,2.0],
gamma12 in [-0.1,0.1]) does Cook's membrane's OWN naturally-emerging macro
strain field reach, as its established tip load is scaled up? Unlike the
cruciform (reentrant corners -> a genuine local singularity, confirmed
this session), Cook's membrane has no sharp/reentrant features anywhere
-- a plain convex trapezoid -- so there is no known concentration risk
here; whatever E11/E22/gamma12 emerges at higher load is a genuine,
physically-coupled structural response (Cook already combines shear and
bending by construction), not something forced via a prescribed boundary
condition the way the square-panel tests were.

Monkey-patches run_cook_pann_claude.TOTAL_FORCE_FINAL (a module-level
constant used directly inside its own Newton loop) rather than modifying
the file -- save_npz=False throughout, so none of the paper's own
established cook_results_*_claude.npz files are ever touched."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/Cook.gid")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.chdir(str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.insert(0, KRATOS_PATH)

import run_cook_pann_claude as cook  # noqa: E402

WHICH = "certified"
MULTIPLES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
E_BOUNDS = {"E11": (-0.1, 2.0), "E22": (-0.1, 2.0), "gamma12": (-0.1, 0.1)}

BASE_FORCE = cook.TOTAL_FORCE_FINAL


def in_box(e11_range, e22_range, g12_range):
    (e11_lo, e11_hi), (e22_lo, e22_hi), (g12_lo, g12_hi) = e11_range, e22_range, g12_range
    ok = (e11_lo >= E_BOUNDS["E11"][0] and e11_hi <= E_BOUNDS["E11"][1]
          and e22_lo >= E_BOUNDS["E22"][0] and e22_hi <= E_BOUNDS["E22"][1]
          and g12_lo >= E_BOUNDS["gamma12"][0] and g12_hi <= E_BOUNDS["gamma12"][1])
    return ok


def main():
    print(f"=== Cook's membrane load-ceiling sweep, material={WHICH}, base force={BASE_FORCE:.4e} N ===", flush=True)
    for mult in MULTIPLES:
        cook.TOTAL_FORCE_FINAL = BASE_FORCE * mult
        t0 = time.time()
        try:
            res = cook.run_newton(WHICH, nx=16, ny=16, verbose=False, diagnose=False,
                                   use_line_search=True, save_npz=False)
            e11_r, e22_r, g12_r = res["e11_range"], res["e22_range"], res["gamma12_range"]
            ok = in_box(e11_r, e22_r, g12_r)
            print(f"  mult={mult:6.1f}x  E11=[{e11_r[0]:+.4f},{e11_r[1]:+.4f}]  "
                  f"E22=[{e22_r[0]:+.4f},{e22_r[1]:+.4f}]  g12=[{g12_r[0]:+.4f},{g12_r[1]:+.4f}]  "
                  f"WITHIN_BOX={ok}  fully_converged={res['fully_converged']}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  mult={mult:6.1f}x  CRASHED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}",
                  flush=True)
        finally:
            cook.TOTAL_FORCE_FINAL = BASE_FORCE

    print("COOK_LOAD_CEILING_SWEEP_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
