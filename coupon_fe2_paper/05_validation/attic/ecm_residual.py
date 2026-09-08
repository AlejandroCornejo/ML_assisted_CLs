#!/usr/bin/env python3
"""Step 2 of the hierarchy: the ECM rule for the PROJECTED RESIDUAL, and the
HPROM it enables.

TWO SEPARATE RULES, not one shared. The projected residual and the homogenized
stress are different integrals over the same mesh, and a rule fitted for one
does not serve the other -- that is exactly the failure the previous project
hit, where weights fitted for the volume average were used against the
reaction-force integrand and produced S == 0. This file builds only the
residual rule; the stress rule comes after, on an HPROM that already works.

Separated rather than shared for consistency with the MAW-ECM route, which
ended up separated. Having HPROM shared and MAW-ECM separated would make two
rows of the comparison differ in something other than what is being compared.
And separation costs little: a shared support must serve both integrands, so
it is built on a combined basis and ends up about the size of the union of two
independent supports anyway.

CONSTRUCTION

    c_e^(j) = (T Phi)[dofs_e, :]^T f_int,e^(j)        in R^39

the element-wise contribution to the projected residual at training state j.
Stacked over elements and states, its leading left singular vectors are the
integrand basis the ECM samples. Evaluated at the FOM snapshots: the GLOBAL
residual vanishes there, but the element-wise contributions do not -- they
cancel, and reproducing that cancellation is precisely what the cubature must
do.

The hyperreduced assembler needs no new code: VectorizedAssembler already
takes `elements` and `element_scales`, and applies the latter as
`w_detJ *= element_scales`, which is the ECM weight semantics exactly.

ACCEPTANCE. The HPROM must reproduce the PROM, not the FOM: the PROM is what
it approximates, so the difference between them is the CUBATURE error alone,
with the projection error already accounted for. PROM reference, measured
under identical conditions: displacement 1.90e-06, stress 4.77e-07 relative
Frobenius, 0 failures, 4.99x over the FOM.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
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

N_SNAP_ECM = 495          # every 10th training state
SVD_TOL = 1.0e-6          # truncation of the integrand basis
ECM_TOL = 1.0e-6          # ECM's own approximation tolerance


def build_integrand_basis(rve, TPhi, U_train, E_train, n_snap, svd_tol,
                          verbose=True):
    a = rve.assembler
    ne, ndl = a.n_elems, a.n_local_dof
    dofs = np.asarray(a.rows_R, dtype=np.int64).reshape(ne, ndl)
    TPhi_e = TPhi[dofs]                       # (ne, ndl, r)

    step = max(1, U_train.shape[0] // n_snap)
    sel = np.arange(0, U_train.shape[0], step)[:n_snap]
    C = np.empty((ne, sel.size * TPhi.shape[1]))
    t0 = time.perf_counter()
    for k, j in enumerate(sel):
        a.Assemble(rve.T @ U_train[j] + rve._g(E_train[j]))
        f_int = a._f_int.reshape(ne, ndl)
        C[:, k * TPhi.shape[1]:(k + 1) * TPhi.shape[1]] = np.einsum(
            "eld,el->ed", TPhi_e, f_int)
    if verbose:
        print(f"  integrand matrix {C.shape} in {time.perf_counter() - t0:.1f}s",
              flush=True)

    # n_el is the smaller dimension, so the Gram route over elements is exact
    # and cheap here.
    G = C @ C.T
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = np.clip(w[order], 0.0, None), V[:, order]
    sv = np.sqrt(w)
    tail = np.cumsum(sv[::-1] ** 2)[::-1]
    total = np.sum(sv ** 2)
    r = next((i for i in range(1, sv.size + 1)
              if np.sqrt((tail[i] if i < sv.size else 0.0) / total) <= svd_tol),
             sv.size)
    if verbose:
        print(f"  integrand basis rank {r} at tol {svd_tol:g} "
              f"(sv1/svr = {sv[0] / max(sv[r - 1], 1e-300):.2e})", flush=True)
    return np.ascontiguousarray(V[:, :r]), sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-snap", type=int, default=N_SNAP_ECM)
    ap.add_argument("--svd-tol", type=float, default=SVD_TOL)
    ap.add_argument("--ecm-tol", type=float, default=ECM_TOL)
    a_ = ap.parse_args()

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    Phi = b["Phi_ROM"]
    U_train, E_train = d["U_train"], d["E_train"]
    ok = np.isfinite(U_train).all(axis=1)
    U_train, E_train = U_train[ok], E_train[ok]

    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    from linear_prom import LinearPROM
    from empirical_cubature_method import EmpiricalCubatureMethod

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        prom = LinearPROM(rve, Phi)
        print(f"mesh {rve.assembler.n_elems} elements, basis {Phi.shape[1]} modes")

        print("\nbuilding the residual integrand basis:")
        Ubasis, sv = build_integrand_basis(
            rve, prom.TPhi, U_train, E_train, a_.n_snap, a_.svd_tol)

        print("\nrunning ECM:")
        ecm = EmpiricalCubatureMethod(ECM_tolerance=a_.ecm_tol,
                                      Filter_tolerance=0.0, Plotting=False)
        ecm.SetUp(Ubasis, constrain_sum_of_weights=True)
        t0 = time.perf_counter()
        ecm.Run()
        z = np.asarray(sorted(ecm.z), dtype=np.int64)
        w_map = dict(zip(np.asarray(ecm.z, dtype=np.int64).tolist(),
                         np.asarray(ecm.w, dtype=float).ravel().tolist()))
        w = np.array([w_map[int(e)] for e in z])
        print(f"  {z.size} elements of {rve.assembler.n_elems} "
              f"({100 * z.size / rve.assembler.n_elems:.1f}%), "
              f"{time.perf_counter() - t0:.1f}s")
        print(f"  weights: min {w.min():.4e}  max {w.max():.4e}  "
              f"sum {w.sum():.4f} (target {rve.assembler.n_elems})")
        print(f"  all positive: {bool(np.all(w > 0))}")

        np.savez_compressed(HERE / "ecm_residual_rule.npz", z=z, w=w,
                            sv=sv, n_basis=Ubasis.shape[1],
                            svd_tol=a_.svd_tol, ecm_tol=a_.ecm_tol,
                            n_snap=a_.n_snap)
    print("\nECM_RESIDUAL_RULE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
