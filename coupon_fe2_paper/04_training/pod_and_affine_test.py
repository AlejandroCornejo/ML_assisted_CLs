#!/usr/bin/env python3
"""Stage 04, first step: the POD basis, and the decisive test on the affine
mu -> q map that D-HPROM-ANN depends on.

WHAT IS BEING TESTED, and why it comes before anything else. In the code the
object is `qp_init_mu_affine`, and `q_p = [mu, 1] @ b_aff`. The same object
plays two different roles, and that IS the difference between the variants:

    HPROM-ANN   (iterative)  the affine map is only the Newton INITIALIZER
    D-HPROM-ANN (direct)     the affine map IS the answer; no equilibrium solve

So D-HPROM-ANN's accuracy is bounded by how affine q really is in mu. That is
a structural assumption of the method, not a data-quantity issue: if q is not
near-affine, no amount of data removes the error floor. Measuring it now,
before any tier is trained, is cheap and it decides scope.

The test needs NO choice of T_m. Since q_M = T_m q_rom is a LINEAR map of the
modal coefficients, q_M is affine in mu for some T_m only if the leading
entries of q_rom are themselves affine in mu. So the residual is measured on
q_rom directly, which avoids inventing an identification strategy whose
details sit in sections of Hernandez's paper not yet read.

TWO CONVENTIONS are compared, because the answer plausibly depends on which:

  A  POD on the FLUCTUATION, u_ind - (E.X restricted to independent dofs).
     This project's `hprom_solver_rve` does this: `u_aff_free + phi_f q`.
  B  POD on the TOTAL u_ind. Hernandez does this: his d_BOUND is nonzero only
     on slave and corner nodes, so d_red carries the affine part, and q_rom is
     then dominated by a term EXACTLY linear in mu.

Convention B should give the better affine fit by construction. If it does,
that is an argument for switching; if A is comparable, the existing pipeline
stands unchanged. Either way it is measured, not assumed.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(ROOT / "00_rve"), str(PROJ / "fe2_extension"),
          str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

TOLS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
R_D = 3          # latent dimension, matching the parameter dimension
N_MODES_REPORT = 8


def deformation_gradient(E):
    """F from Green-Lagrange E in Voigt [E11, E22, gamma12], gamma engineering."""
    Et = np.array([[E[0], 0.5 * E[2]], [0.5 * E[2], E[1]]])
    C = 2.0 * Et + np.eye(2)
    w, V = np.linalg.eigh(C)
    if np.min(w) <= 0.0:
        raise RuntimeError("C not positive definite")
    return V @ np.diag(np.sqrt(w)) @ V.T


def affine_at_independent(E, ind_xy, ind_comp):
    H = deformation_gradient(E) - np.eye(2)
    return np.einsum("ij,kj->ki", H, ind_xy)[np.arange(ind_xy.shape[0]), ind_comp]


def pod_method_of_snapshots(D):
    """Left singular vectors and values of D (n_dof x n_snap) via the Gram
    matrix. Exact, and cheaper here than a dense SVD of D since n_snap is the
    smaller dimension."""
    G = D.T @ D
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    w = np.clip(w, 0.0, None)
    sv = np.sqrt(w)
    keep = sv > sv[0] * 1e-14
    sv, V = sv[keep], V[:, keep]
    Phi = (D @ V) / sv
    return Phi, sv


def rank_for_tol(sv, tol):
    """Smallest r with ||D - Phi_r Phi_r^T D||_F / ||D||_F <= tol."""
    e = np.cumsum(sv[::-1] ** 2)[::-1]
    total = np.sum(sv ** 2)
    for r in range(1, sv.size + 1):
        tail = e[r] if r < sv.size else 0.0
        if np.sqrt(tail / total) <= tol:
            return r
    return sv.size


def affine_fit(mu, Q):
    """Least squares Q ~ [mu, 1] B. Returns per-column relative residual."""
    A = np.column_stack([mu, np.ones(mu.shape[0])])
    B, *_ = np.linalg.lstsq(A, Q, rcond=None)
    R = Q - A @ B
    denom = np.linalg.norm(Q, axis=0)
    denom[denom == 0.0] = 1.0
    return np.linalg.norm(R, axis=0) / denom, B, float(np.linalg.cond(A))


def main():
    d = np.load(ROOT / "03_data" / "data.npz")
    E = d["E_train"]
    U = d["U_train"]
    ok = np.all(np.isfinite(U), axis=1)
    if not np.all(ok):
        print(f"dropping {int(np.sum(~ok))} states with non-finite snapshots")
    E, U = E[ok], U[ok]
    print(f"snapshots: {U.shape[0]} states x {U.shape[1]} independent dofs")

    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"), cell_area=float(d["cell_area"]))
    ind_xy, ind_comp = rve.ind_xy, rve.ind_comp
    if ind_xy.shape[0] != U.shape[1]:
        raise RuntimeError(f"independent-dof count mismatch: mesh {ind_xy.shape[0]} "
                           f"vs snapshots {U.shape[1]}")

    aff = np.array([affine_at_independent(E[i], ind_xy, ind_comp)
                    for i in range(E.shape[0])])
    fluct = U - aff
    print(f"affine part carries {np.linalg.norm(aff) / np.linalg.norm(U) * 100:.2f}% "
          f"of the snapshot norm; fluctuation "
          f"{np.linalg.norm(fluct) / np.linalg.norm(U) * 100:.2f}%")

    out = {}
    for name, D in (("A_fluctuation", fluct.T), ("B_total", U.T)):
        Phi, sv = pod_method_of_snapshots(np.ascontiguousarray(D))
        ranks = {t: rank_for_tol(sv, t) for t in TOLS}
        Q = (Phi.T @ D).T                      # q_rom per snapshot
        res, B, cond = affine_fit(E, Q[:, :max(8, R_D)])
        print(f"\n--- convention {name} ---")
        print("  singular value decay (normalized): "
              + "  ".join(f"{s / sv[0]:.2e}" for s in sv[:N_MODES_REPORT]))
        print("  rank for tol: " + "  ".join(f"{t:g}->{r}" for t, r in ranks.items()))
        print(f"  affine mu-fit design matrix cond {cond:.2f}")
        print("  affine fit residual per mode:")
        for k in range(min(N_MODES_REPORT, res.size)):
            flag = "  <-- latent" if k < R_D else ""
            print(f"    mode {k + 1}: {res[k]:.4e}{flag}")
        lead = float(np.max(res[:R_D]))
        print(f"  WORST over the {R_D} latent modes: {lead:.4e}")
        out[name] = dict(sv=sv, ranks=ranks, res=res, lead=lead)

    a, b = out["A_fluctuation"]["lead"], out["B_total"]["lead"]
    print(f"\nleading-mode affine residual:  A (fluctuation) {a:.4e}   "
          f"B (total) {b:.4e}")
    print(f"convention B is {'BETTER' if b < a else 'NOT better'} "
          f"by a factor of {max(a, b) / max(min(a, b), 1e-300):.2f}")
    print("\nReading: this residual is the floor on D-HPROM-ANN, since it uses "
          "the affine map as its answer. HPROM-ANN only uses it to start "
          "Newton, so a large residual costs it iterations, not accuracy.")

    np.savez_compressed(HERE / "pod_affine_test.npz",
                        sv_A=out["A_fluctuation"]["sv"], sv_B=out["B_total"]["sv"],
                        res_A=out["A_fluctuation"]["res"], res_B=out["B_total"]["res"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
