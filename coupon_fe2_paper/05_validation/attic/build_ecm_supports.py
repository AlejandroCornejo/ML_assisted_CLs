#!/usr/bin/env python3
"""Build and save both ECM supports, with the weights explicitly tied to the
element indices they belong to.

The reason this exists as a script rather than a throwaway: an earlier version
saved the stress support SORTED and its weights in the ECM's OWN OUTPUT ORDER.
Every downstream consumer that wrote a reduced mesh in sorted order then gave
each element another element's weight, and the HPROM's stress error came out
at 1.27e-01 instead of 1.53e-04 -- a factor of 830, with the solve itself
perfectly convergent and the numbers entirely plausible. That is the previous
project's failure mode exactly: weights that do not match what they multiply.

So both orders are saved for both rules, and the weights are keyed to element
indices, so alignment is not something a consumer has to get right by
convention.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
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

RES_SVD_TOL = 1.0e-3      # 135 elements
SIG_SVD_TOL = 1.0e-5      # 73 elements, where the stress rule saturates


def main():
    import periodic_fom as pf
    import linear_prom as lp
    lp.SUBSTEPS_PER_UNIT_STRAIN = pf.SUBSTEPS_PER_UNIT_STRAIN

    from periodic_fom import PeriodicRVE
    from linear_prom import LinearPROM
    from _material_law_guard_claude import true_neo_hookean_active
    from ecm_residual import build_integrand_basis
    from hprom_full import build_stress_integrand, rank_for, run_ecm

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    rule = np.load(HERE / "ecm_residual_rule.npz")
    n_snap = int(rule["n_snap"])

    U, E = d["U_train"], d["E_train"]
    ok = np.isfinite(U).all(axis=1)
    U, E = U[ok], E[ok]

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        prom = LinearPROM(rve, b["Phi_ROM"])

        Ures, sv_res = build_integrand_basis(rve, prom.TPhi, U, E, n_snap,
                                             1e-12, verbose=False)
        r_res = rank_for(sv_res, RES_SVD_TOL)
        z_res, w_res = run_ecm(np.ascontiguousarray(Ures[:, :r_res]))

        Usig, sv_sig = build_stress_integrand(rve, U, E, n_snap, verbose=False)
        r_sig = rank_for(sv_sig, SIG_SVD_TOL)
        z_sig, w_sig = run_ecm(np.ascontiguousarray(Usig[:, :r_sig]))

    for nm, z, w, r in (("residual", z_res, w_res, r_res),
                        ("stress", z_sig, w_sig, r_sig)):
        print(f"{nm:9s}: rank {r:4d} -> {z.size:4d} elements "
              f"({100 * z.size / 1546:4.1f}%), weights "
              f"[{w.min():.3e}, {w.max():.3e}], sum {w.sum():.4f}, "
              f"all positive {bool(np.all(w > 0))}")

    inter = np.intersect1d(z_res, z_sig)
    union = np.union1d(z_res, z_sig)
    print(f"\noverlap {inter.size} elements, union {union.size} "
          f"({100 * union.size / 1546:.1f}% of the mesh)")

    # Weights keyed to element index, and both orders kept, so no consumer has
    # to reproduce an ordering convention.
    np.savez_compressed(
        HERE / "ecm_supports.npz",
        z_res_ecm_order=z_res, w_res_ecm_order=w_res,
        z_sig_ecm_order=z_sig, w_sig_ecm_order=w_sig,
        z_res=np.sort(z_res), w_res=w_res[np.argsort(z_res)],
        z_sig=np.sort(z_sig), w_sig=w_sig[np.argsort(z_sig)],
        rank_res=r_res, rank_sig=r_sig,
        res_svd_tol=RES_SVD_TOL, sig_svd_tol=SIG_SVD_TOL)

    # Self-check that the sorted pairing is right.
    s = np.load(HERE / "ecm_supports.npz")
    m1 = dict(zip(s["z_res_ecm_order"].tolist(), s["w_res_ecm_order"].tolist()))
    m2 = dict(zip(s["z_res"].tolist(), s["w_res"].tolist()))
    m3 = dict(zip(s["z_sig_ecm_order"].tolist(), s["w_sig_ecm_order"].tolist()))
    m4 = dict(zip(s["z_sig"].tolist(), s["w_sig"].tolist()))
    ok = (m1 == m2) and (m3 == m4)
    print(f"sorted-vs-ECM-order weight pairing identical: {'OK' if ok else 'FAIL'}")
    print("ECM_SUPPORTS_DONE" if ok else "ECM_SUPPORTS_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
