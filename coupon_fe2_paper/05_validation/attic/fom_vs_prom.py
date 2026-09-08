#!/usr/bin/env python3
"""FOM against PROM under IDENTICAL conditions. One table, all measured.

Everything that could differ is forced equal and asserted at run time:

  * same states
  * same ramp density (200 substeps per unit strain, the FOM's own; the PROM
    was measured to need it too -- at 60 it produced 11 catastrophic outliers
    out of 200 states, two diverging, all fixed at 200 and bit-identical at
    600, so 200 is converged rather than merely better)
  * same Newton tolerance, both overridden to one value
  * same mesh, same process, same thread count

The FOM solved IN THIS RUN is the reference, not the stored dataset, so both
columns come from one tolerance in one process.

Caveat stated rather than hidden: the two Newton criteria act on different
vectors -- the FOM's on the displacement increment over 6320 dofs, the PROM's
on the modal increment over 39. Identical criterion form and identical number
is the closest apples-to-apples available without changing the FOM; it is not
a claim that both stop at literally the same residual level. Making that exact
would mean switching both to a residual-based criterion, which is a change to
the FOM and belongs to its own decision.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(ROOT / "00_rve"), str(ROOT / "04_training"),
          str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

N = 40
TOL = 1.0e-10

import periodic_fom as pf      # noqa: E402
import linear_prom as lp       # noqa: E402

pf.NEWTON_TOL = TOL
lp.NEWTON_TOL = TOL
lp.SUBSTEPS_PER_UNIT_STRAIN = pf.SUBSTEPS_PER_UNIT_STRAIN

from periodic_fom import PeriodicRVE                             # noqa: E402
from linear_prom import LinearPROM                               # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active   # noqa: E402


def main():
    assert pf.NEWTON_TOL == lp.NEWTON_TOL, "tolerance mismatch"
    assert pf.SUBSTEPS_PER_UNIT_STRAIN == lp.SUBSTEPS_PER_UNIT_STRAIN, "ramp mismatch"
    print(f"tol {TOL:g} for both, ramp {pf.SUBSTEPS_PER_UNIT_STRAIN:g}/unit for both",
          flush=True)

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    E_all, S_all = d["E_test"], d["S_test"]
    ok = np.isfinite(S_all).all(axis=1)
    E_all = E_all[ok]
    idx = np.random.default_rng(3).choice(E_all.shape[0], N, replace=False)

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        Phi = b["Phi_ROM"]
        prom = LinearPROM(rve, Phi)

        t_f, t_p, nfail = [], [], 0
        U_f, U_p, Sf, Sp = [], [], [], []
        for j, i in enumerate(idx, 1):
            E = E_all[i]
            t0 = time.perf_counter()
            S_f, u_f = rve.solve(E)
            tf = time.perf_counter() - t0
            try:
                t0 = time.perf_counter()
                S_p, q = prom.solve(E)
                tp = time.perf_counter() - t0
            except RuntimeError:
                nfail += 1
                continue
            t_f.append(tf)
            t_p.append(tp)
            u_p = Phi @ q
            U_f.append(u_f); U_p.append(u_p)
            Sf.append(S_f); Sp.append(S_p)
            if j % 10 == 0:
                print(f"  {j}/{N} done", flush=True)

    t_f, t_p = np.array(t_f), np.array(t_p)
    U_f, U_p = np.array(U_f), np.array(U_p)
    Sf, Sp = np.array(Sf), np.array(Sp)

    # TABLE 1: one relative FROBENIUS norm per quantity, over all states at
    # once. This is the reporting number.
    e_u = np.linalg.norm(U_p - U_f) / np.linalg.norm(U_f)
    e_s = np.linalg.norm(Sp - Sf) / np.linalg.norm(Sf)

    print(f"\n=== TABLE 1 (clean) === {len(t_f)} states, {nfail} PROM failures\n")
    print(f"{'quantity':<34} {'value':>14}")
    print("-" * 50)
    print(f"{'displacement error (rel. Frobenius)':<34} {e_u:14.4e}")
    print(f"{'stress error (rel. Frobenius)':<34} {e_s:14.4e}")
    print(f"{'FOM wall clock (s)':<34} {t_f.sum():14.2f}")
    print(f"{'PROM wall clock (s)':<34} {t_p.sum():14.2f}")
    print(f"{'speedup':<34} {t_f.sum() / t_p.sum():13.2f}x")

    # TABLE 2: per-state distribution. Kept because a global Frobenius norm
    # AVERAGES OVER outliers, and outliers are what caught the ramp mistake:
    # the same comparison at 60 substeps/unit had median 3.9e-07 with max
    # 2.5e+01, i.e. 11 of 200 states broken. A single norm would have reported
    # something plausible and hidden it.
    eu = np.linalg.norm(U_p - U_f, axis=1) / np.linalg.norm(U_f, axis=1)
    es = np.linalg.norm(Sp - Sf, axis=1) / np.linalg.norm(Sf, axis=1)
    print("\n=== TABLE 2 (diagnostic, per state) ===\n")
    print(f"{'quantity':<24} {'median':>12} {'p90':>12} {'max':>12}")
    print("-" * 64)
    print(f"{'displacement error':<24} {np.median(eu):12.4e} "
          f"{np.percentile(eu, 90):12.4e} {eu.max():12.4e}")
    print(f"{'stress error':<24} {np.median(es):12.4e} "
          f"{np.percentile(es, 90):12.4e} {es.max():12.4e}")
    print(f"{'FOM time (s)':<24} {np.median(t_f):12.4f} "
          f"{np.percentile(t_f, 90):12.4f} {t_f.max():12.4f}")
    print(f"{'PROM time (s)':<24} {np.median(t_p):12.4f} "
          f"{np.percentile(t_p, 90):12.4f} {t_p.max():12.4f}")
    np.savez_compressed(HERE / "fom_vs_prom.npz", t_fom=t_f, t_prom=t_p,
                        err_u_frob=e_u, err_s_frob=e_s,
                        err_u_per_state=eu, err_s_per_state=es, tol=TOL)
    print("FOM_VS_PROM_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
