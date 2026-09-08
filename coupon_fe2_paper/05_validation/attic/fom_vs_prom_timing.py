#!/usr/bin/env python3
"""FOM against PROM on the SAME states, SAME ramp density, SAME process.

The speedup previously quoted compared a measured PROM time (1.894 s) against
an ESTIMATED FOM time (~7 s) extrapolated from a different mesh. That is not a
speedup measurement. Both are timed here under identical conditions:

  * same 40 test states
  * same SUBSTEPS_PER_UNIT_STRAIN = 200, which is the FOM's own density and
    which the PROM was measured to need too (at 60 it produced 11 catastrophic
    outliers out of 200 states, all fixed at 200 and bit-identical at 600)
  * same mesh, same process, same thread count

Accuracy is reported alongside, since a speedup at a different accuracy is not
a speedup either.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(ROOT / "00_rve"), str(ROOT / "04_training"),
          str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.append("/home/kratos/Kratos_Eigen_Check/bin/Release")

N = 40
b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
d = np.load(ROOT / "03_data" / "data.npz")

from periodic_fom import PeriodicRVE  # noqa: E402
import periodic_fom as pf  # noqa: E402
from linear_prom import LinearPROM  # noqa: E402
import linear_prom as lp  # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402

assert pf.SUBSTEPS_PER_UNIT_STRAIN == lp.SUBSTEPS_PER_UNIT_STRAIN, (
    f"ramp mismatch: FOM {pf.SUBSTEPS_PER_UNIT_STRAIN} vs "
    f"PROM {lp.SUBSTEPS_PER_UNIT_STRAIN}")
print(f"ramp density (both): {pf.SUBSTEPS_PER_UNIT_STRAIN:g} substeps/unit strain")

E_s, S_s = d["E_test"], d["S_test"]
ok = np.isfinite(S_s).all(axis=1)
E_s, S_s = E_s[ok], S_s[ok]
idx = np.random.default_rng(3).choice(E_s.shape[0], N, replace=False)

with true_neo_hookean_active():
    rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"), cell_area=float(d["cell_area"]))
    prom = LinearPROM(rve, b["Phi_ROM"])
    res = {}
    for tag, fn in (("FOM", lambda E: rve.solve(E)[0]),
                    ("PROM", lambda E: prom.solve(E)[0])):
        t, errs = [], []
        for i in idx:
            t0 = time.perf_counter()
            S = fn(E_s[i])
            t.append(time.perf_counter() - t0)
            errs.append(np.linalg.norm(S - S_s[i]) / np.linalg.norm(S_s[i]))
        res[tag] = (np.array(t), np.array(errs))
        print(f"  {tag:5s} n={N}  {np.mean(t):.3f} s/solve (median "
              f"{np.median(t):.3f})   error median {np.median(errs):.4e}",
              flush=True)
tf, tp = res["FOM"][0], res["PROM"][0]
print(f"\nspeedup on mean time: {np.mean(tf) / np.mean(tp):.2f}x   "
      f"on median: {np.median(tf) / np.median(tp):.2f}x")
print("FOM error is the reference, so its 'error' column is roundoff by "
      "construction and only confirms the comparison is against the same data.")
print("TIMING_DONE")
