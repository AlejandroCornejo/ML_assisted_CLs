#!/usr/bin/env python3
"""Stage 04: the factorized decoder's master/slave bases, and the
(q_M, q_S) training pairs the ANN closure is learned from.

All of this is deterministic linear algebra with no free choices, taken
straight from Hernandez's construction rather than invented:

  1. POD:            Phi_ROM from the snapshot matrix, truncated at tol
  2. modal coeffs:   Q_rom = Phi_ROM^T D
  3. identification: T_m = M_train Q_rom^+     (input-informed: makes the
                     latent coordinates BE the macro strain, q_M ~ mu)
  4. master basis:   Phi_ROM T_m^T = Phi_M Sig_M V_M^T   (compact SVD)
  5. scaling:        A_M = Sig_M^-1 V_M^T
  6. slave basis:    Phi_S spans the orthogonal complement of Col(Phi_M)
                     WITHIN Col(Phi_ROM)
  7. pairs:          q_M = T_m Q_rom,   q_S = Phi_S^T D

Step 6 is the one worth stating carefully: Phi_M is NOT the first three POD
modes. It comes from the SVD of Phi_ROM T_m^T, so the master/slave split is a
ROTATION of the POD basis aligned with the identification. Taking "first three
versus the rest" would be wrong.

A_M is what makes the decomposition consistent, and the derivation pins it
down rather than leaving it as a convention: premultiplying
d_red = Phi_M A_M q_M + Phi_S N(q_M) by Phi_M^T and using the orthogonality
Phi_M^T Phi_S = 0 gives Phi_M^T d_red = A_M q_M; substituting
q_M = T_m Phi_ROM^T d_red and the SVD yields Sig_M V_M^T A_M^T = I, hence
A_M = Sig_M^-1 V_M^T.

ACCEPTANCE. The decisive test is that with q_S taken from the DATA rather
than from a network,

    d_red = Phi_M A_M q_M + Phi_S q_S

must hold to the POD truncation error and no worse. If the master/slave split
or A_M were wrong, this identity would fail while every orthogonality check
still passed.

Convention B (snapshots retain the affine part) is used, chosen in
pod_and_affine_test.py by measurement: it beat the fluctuation convention by
about 2.5x on the identification residual at all four truncations.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(ROOT / "00_rve"),
          str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from pod_and_affine_test import pod_method_of_snapshots, rank_for_tol  # noqa: E402

R_D = 3
TOLS = (1e-6,)


def pod_direct_svd(D):
    """Left singular vectors and values of D by a DIRECT SVD.

    Not the method of snapshots. That route eigendecomposes D^T D, which
    SQUARES the condition number, so a retained mode's relative accuracy
    degrades like eps*(sigma_1/sigma_k)^2. Measured consequence at r_ROM = 39,
    where sigma_39/sigma_1 ~ 1e-6: Phi_S came out orthonormal only to 1.08e-04
    and Col([Phi_M, Phi_S]) missed Col(Phi_ROM) by 4.9e-06, while at
    r_ROM = 16, whose modes are all large, every check passed at 1e-11 or
    better. A direct SVD costs one dense factorization here and removes the
    problem at source.
    """
    U, sv, _Vt = np.linalg.svd(D, full_matrices=False)
    return U, sv


def build(D, M, tol):
    """D (n_ind, n_snap) snapshots, M (3, n_snap) macro strains."""
    Phi_full, sv = pod_direct_svd(D)
    r = rank_for_tol(sv, tol)
    Phi = np.ascontiguousarray(Phi_full[:, :r])

    Q_rom = Phi.T @ D                                   # (r, n_snap)
    T_m = M @ np.linalg.pinv(Q_rom)                     # (3, r)

    # master basis and scaling from the compact SVD of Phi_ROM T_m^T
    Phi_M, sig_M, VMt = np.linalg.svd(Phi @ T_m.T, full_matrices=False)
    A_M = np.diag(1.0 / sig_M) @ VMt                    # Sig^-1 V^T, (3, 3)

    # slave basis: complement of Col(Phi_M) inside Col(Phi_ROM). Done in the
    # r-dimensional coordinate space of Phi_ROM so Col(Phi_S) is guaranteed to
    # stay inside Col(Phi_ROM), which a complement taken in the full space
    # would not be.
    B = Phi.T @ Phi_M                                   # (r, 3), orthonormal cols
    Qf, _ = np.linalg.qr(B, mode="complete")            # (r, r)
    C = Qf[:, R_D:]                                     # (r, r-3)

    # ENERGY-ORDER the slave basis. The QR returns an ARBITRARY orthonormal
    # basis of the complement, so its columns carry no ordering: "the first k
    # columns of Phi_S" would be k arbitrary directions, not the k most
    # important slave modes. Measured consequence of not doing this -- the
    # median stress error of a truncated reconstruction came out at order 1 to
    # 25 (i.e. 100% to 2500%) for every k < 36 and then fell to 1.2e-06 at the
    # full set, a cliff rather than a convergence, because truncation was
    # discarding large-energy content.
    #
    # Hernandez's definition only requires Phi_S to be SOME orthonormal basis
    # of the complement, leaving the choice free; energy ordering is the useful
    # one, and not only for truncation studies -- it makes the ANN's outputs
    # ordered by importance, which is what a per-mode loss weighting needs.
    qS_raw = C.T @ Q_rom                                # (r-3, n_snap)
    U_s, sv_S, _Vs = np.linalg.svd(qS_raw, full_matrices=False)
    Phi_S = Phi @ (C @ U_s)                             # (n_ind, r-3), ordered

    q_M = T_m @ Q_rom                                   # (3, n_snap)
    q_S = Phi_S.T @ D                                   # (r-3, n_snap)

    # POD truncation error, the floor the reconstruction test is measured against
    trunc = float(np.linalg.norm(D - Phi @ (Phi.T @ D)) / np.linalg.norm(D))
    return dict(Phi=Phi, sv=sv, sv_S=sv_S, r=r, T_m=T_m, Phi_M=Phi_M, A_M=A_M,
                Phi_S=Phi_S, q_M=q_M, q_S=q_S, trunc=trunc, D=D, M=M)


def report(res, label):
    Phi, Phi_M, Phi_S = res["Phi"], res["Phi_M"], res["Phi_S"]
    A_M, q_M, q_S, D, M = res["A_M"], res["q_M"], res["q_S"], res["D"], res["M"]
    nD = np.linalg.norm(D)

    o_MM = float(np.max(np.abs(Phi_M.T @ Phi_M - np.eye(R_D))))
    o_SS = float(np.max(np.abs(Phi_S.T @ Phi_S - np.eye(Phi_S.shape[1]))))
    o_MS = float(np.max(np.abs(Phi_M.T @ Phi_S)))

    # [Phi_M, Phi_S] must span exactly Col(Phi_ROM): compare the projectors,
    # via their action on the snapshots rather than forming n_ind x n_ind.
    PMS = Phi_M @ (Phi_M.T @ D) + Phi_S @ (Phi_S.T @ D)
    PR = Phi @ (Phi.T @ D)
    span = float(np.linalg.norm(PMS - PR) / nD)

    # A_M consistency: Phi_M^T d_red must equal A_M q_M
    amq = float(np.linalg.norm(A_M @ q_M - Phi_M.T @ D)
                / max(np.linalg.norm(Phi_M.T @ D), 1e-300))

    # THE test: reconstruction with q_S from the data
    rec = Phi_M @ (A_M @ q_M) + Phi_S @ q_S
    rec_err = float(np.linalg.norm(D - rec) / nD)

    ident = float(np.linalg.norm(q_M - M) / np.linalg.norm(M))

    print(f"\n--- {label}: r_ROM = {res['r']}, r_D = {R_D}, "
          f"slaves = {Phi_S.shape[1]} ---")
    print(f"  POD truncation error          {res['trunc']:.3e}")
    print(f"  reconstruction with data q_S  {rec_err:.3e}   <-- must match the line above")
    print(f"  A_M consistency               {amq:.3e}")
    print(f"  identification q_M vs mu      {ident:.3e}")
    print(f"  orthogonality  MM {o_MM:.2e}  SS {o_SS:.2e}  MS {o_MS:.2e}")
    print(f"  span([Phi_M,Phi_S]) vs Col(Phi_ROM)  {span:.3e}")
    svS = res["sv_S"]
    print(f"  slave energy spectrum (normalized): "
          + "  ".join(f"{v / svS[0]:.2e}" for v in svS[:6])
          + f"  ...  {svS[-1] / svS[0]:.2e}")

    checks = [
        ("reconstruction equals POD truncation",
         rec_err < max(1.5 * res["trunc"], 1e-12),
         f"{rec_err:.3e} vs {res['trunc']:.3e}"),
        ("A_M consistent", amq < 1e-9, f"{amq:.3e}"),
        ("Phi_M orthonormal", o_MM < 1e-10, f"{o_MM:.2e}"),
        ("Phi_S orthonormal", o_SS < 1e-10, f"{o_SS:.2e}"),
        ("Phi_M perpendicular to Phi_S", o_MS < 1e-10, f"{o_MS:.2e}"),
        ("master+slave span equals Phi_ROM's", span < 1e-10, f"{span:.3e}"),
    ]
    return checks, ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convention", choices=("A", "B"), default="B")
    a = ap.parse_args()

    d = np.load(ROOT / "03_data" / "data.npz")
    E, U = d["E_train"], d["U_train"]
    ok = np.isfinite(U).all(axis=1)
    E, U = E[ok], U[ok]
    print(f"snapshots {U.shape[0]} x {U.shape[1]}, convention {a.convention}")

    if a.convention == "A":
        from pod_and_affine_test import affine_at_independent
        from periodic_fom import PeriodicRVE
        from _material_law_guard_claude import true_neo_hookean_active
        with true_neo_hookean_active():
            rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                              cell_area=float(d["cell_area"]))
        aff = np.array([affine_at_independent(E[i], rve.ind_xy, rve.ind_comp)
                        for i in range(E.shape[0])])
        D = np.ascontiguousarray((U - aff).T)
    else:
        D = np.ascontiguousarray(U.T)
    M = np.ascontiguousarray(E.T)

    all_ok = True
    for tol in TOLS:
        res = build(D, M, tol)
        checks, ident = report(res, f"tol {tol:g}")
        print("  acceptance:")
        for name, good, detail in checks:
            print(f"    [{'OK  ' if good else 'FAIL'}] {name}  ({detail})")
        all_ok = all_ok and all(g for _, g, _ in checks)
        np.savez_compressed(
            HERE / f"decoder_basis_{a.convention}_r{res['r']}.npz",
            Phi_ROM=res["Phi"], Phi_M=res["Phi_M"], Phi_S=res["Phi_S"],
            A_M=res["A_M"], T_m=res["T_m"], q_M=res["q_M"], q_S=res["q_S"],
            E_train=E, sv=res["sv"], sv_S=res["sv_S"], r_ROM=res["r"], r_D=R_D,
            trunc=res["trunc"], ident=ident)

    print("\nDECODER_BASIS_PASS" if all_ok else "\nDECODER_BASIS_FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
