#!/usr/bin/env python3
"""Deploy the MAW-ECM stress rule inside the HPROM-ANN and measure the error
that actually matters: homogenized stress against the FOM, in and out of the
training envelope.

Constraint reproduction was a proxy. This is the quantity the paper reports.

WHAT IS COMPARED, all on the SAME converged q from the SAME Newton solve, so
the only thing that varies is the cubature used for the homogenized stress:

    classic ECM, 73 points, fixed w    the accurate reference rule
    classic ECM, 10 points, fixed w    the equal-cost baseline
    MAW-ECM,     10 points, w(q)       the adaptive rule

The residual inside Newton keeps the classic fixed-weight rule. That is not a
concession, it is what the cancellation measurement says to do: the projected
residual is a 98% cancellation of its element contributions, so a non-negative
10-point rule for it can only be near-degenerate, and it is independently how
the previous project deployed MAW-ECM -- on the homogenization target.

NO TIMING IS REPORTED HERE. Wall-clock numbers are not measured while other
jobs share the machine; accuracy is unaffected by CPU contention, speed-up is
not. Timing belongs in a separate serial run.

ADAPTIVE WEIGHTS ARE APPLIED BY RESCALING w_detJ. The assembler folds
`element_scales` into `w_detJ` once, in its constructor, so weights cannot be
changed afterwards through the public interface. But `w_detJ` is read only
inside Assemble and the stress/volume routines -- nothing else caches it or
anything derived from it (checked) -- so building with UNIT scales and setting
`w_detJ = base * w(q)` before each assemble is exactly equivalent to a
per-state cubature weight, with no modification to the shared solver.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
from pathlib import Path

import numpy as np

import maw_lab as L

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# maw_lab puts the shared solver directories on the path but not the stage
# directories this script needs (periodic_fom, the decoder basis loader).
for _p in (str(ROOT / "00_rve"), str(ROOT / "04_training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_EVAL = 40
TOL = 1.0e-10
MAX_IT = 60
SUBSTEPS = 200.0
# Both classic rules are now selected by SPECTRUM TOLERANCE rather than by a
# hand-picked point count. The earlier run fixed the residual at 40 points
# because that is what the manifold rule happened to use at tol 1e-3; that made
# the residual's own cubature error (2.1e-02) part of the floor, which is
# exactly what a tighter tolerance separates out.
RES_TOL = 1.0e-4      # classic fixed-weight residual rule inside Newton
SIG_REF_TOL = 1.0e-4  # the accurate classic stress rule
SIG_PTS = 10          # the adaptive stress rule, and its equal-cost baseline


def rank_for(sv, tol):
    tail = np.cumsum(sv[::-1] ** 2)[::-1]
    total = np.sum(sv ** 2)
    for i in range(1, sv.size + 1):
        if np.sqrt((tail[i] if i < sv.size else 0.0) / total) <= tol:
            return i
    return sv.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--fields", default="maw_fields.npz",
                    help="npz holding the trained field(s)")
    ap.add_argument("--sig-pts", type=int, default=SIG_PTS,
                    help="which MAW stress rule to deploy, by point count")
    a_ = ap.parse_args()
    sig_pts = int(a_.sig_pts)

    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    from reduced_mesh import ReducedAssembly
    from numpy_decoder import NumpyDecoder

    d = np.load(ROOT / "03_data" / "data.npz")
    fl = np.load(HERE / a_.fields)
    lab = L.load()
    q_tr, ne = lab["q"], lab["ne"]
    fi, vi = L.split(q_tr.shape[0])

    key = f"sig_{sig_pts}"
    z_maw = fl[f"{key}_z"]
    model = dict(state={k[len(key) + 5:]: fl[k] for k in fl.files
                        if k.startswith(f"{key}_net_")},
                 mu=fl[f"{key}_mu"], sd=fl[f"{key}_sd"],
                 act=str(fl[f"{key}_act"]), target_sum=float(ne))
    W_field_tr = L.field_weights(model, q_tr)

    C_sig = lab["C_sig"]
    b_sig = L.targets(C_sig)
    A_maw = L.blocks(C_sig, z_maw)
    print(f"MAW stress rule: {z_maw.size} points")
    print(f"  constraint error, held-out states: "
          f"{np.median(L.const_err(A_maw[vi], b_sig[vi], W_field_tr[:, vi])):.4e}")
    print(f"  w >= 0: {bool(np.all(W_field_tr >= 0))}   "
          f"sum {W_field_tr.sum(0).min():.4f}..{W_field_tr.sum(0).max():.4f} "
          f"(n_elements {ne})", flush=True)

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        dec = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                           ROOT / "04_training" / "nslave.npz")
        full = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"

        U_res, sv_res = L.integrand_svd(lab["C_res"], fi)
        z_res, w_res, _ = L.classic_ecm(lab["C_res"], rank_for(sv_res, RES_TOL),
                                        U=U_res)
        U_sg, sv_sg = L.integrand_svd(C_sig, fi)
        z_ref, w_ref, _ = L.classic_ecm(C_sig, rank_for(sv_sg, SIG_REF_TOL),
                                        U=U_sg)
        z_b, w_b, _ = L.classic_ecm(C_sig, z_maw.size - 1, U=U_sg)
        print(f"residual rule: tol {RES_TOL:.0e} -> {z_res.size} elements")
        print(f"stress reference: tol {SIG_REF_TOL:.0e} -> {z_ref.size} elements")
        print(f"stress baseline / MAW: {z_b.size} / {z_maw.size} elements",
              flush=True)

        Tcsr = rve.T.tocsr()
        counts = np.diff(Tcsr.indptr)
        Trows = np.full(rve.n_dof, -1, dtype=np.int64)
        nz = counts > 0
        if int(counts.max()) != 1 or int(counts[nz].min()) != 1:
            raise RuntimeError("T is not one-nonzero-per-row")
        Trows[nz] = Tcsr.indices[Tcsr.indptr[:-1][nz]]

        def restricted(sel):
            rows = Trows[sel]
            mask = (rows >= 0).astype(float)
            return dec.restrict(np.maximum(rows, 0)), mask

        red_res = ReducedAssembly(full, HERE / "dep_res", z_res, w_res, rve,
                                  np.zeros((rve.n_dof, 1)))
        red_ref = ReducedAssembly(full, HERE / "dep_ref", z_ref, w_ref, rve,
                                  np.zeros((rve.n_dof, 1)))
        red_b = ReducedAssembly(full, HERE / "dep_base", z_b, w_b, rve,
                                np.zeros((rve.n_dof, 1)))
        # unit scales, so w_detJ IS the base and the adaptive weights multiply
        # it directly with no division by the initial weights
        red_m = ReducedAssembly(full, HERE / "dep_maw", z_maw,
                                np.ones(z_maw.size), rve,
                                np.zeros((rve.n_dof, 1)))
        base_m = np.array(red_m.asm.w_detJ, dtype=float, copy=True)

        dr, mr = restricted(red_res.sel)
        packs = {}
        for nmk, red in (("ref", red_ref), ("base", red_b), ("maw", red_m)):
            dd, mm = restricted(red.sel)
            packs[nmk] = (red, dd, mm)

        # ALIGNMENT CHECK. Reduced-mesh element order must match the sorted
        # support the weights were computed for. This exact class of bug --
        # weights paired with the wrong support ordering -- has already cost
        # this project real time, and it is invisible in a global norm, so it
        # is checked directly: with the ORACLE per-state weights the reduced
        # rule must reproduce the full-mesh stress to the constraint error.
        Worc = fl[f"{key}_W_oracle"]
        j0 = int(fi[0])
        red_m.asm.w_detJ = base_m * Worc[:, j0][:, None]
        ddm, mmm = packs["maw"][1], packs["maw"][2]
        dd0, _ = ddm.value_and_jac(q_tr[j0])
        E0 = np.load(ROOT / "03_data" / "data.npz")["E_train"][
            np.load(HERE / "full_integrand.npz")["state_index"][j0]]
        red_m.asm.Assemble(dd0 * mmm + rve._g(E0)[packs["maw"][0].sel])
        S_or = rve.homogenized_stress(E0, assembler=red_m.asm)
        Pb = b_sig[j0, :4].reshape(2, 2)
        Fb = rve._fom.DeformationGradientFromGreenLagrange2D(E0)
        Sb = np.linalg.solve(Fb, Pb)
        S_ex = np.array([Sb[0, 0], Sb[1, 1], 0.5 * (Sb[0, 1] + Sb[1, 0])])
        al = np.linalg.norm(S_or - S_ex) / np.linalg.norm(S_ex)
        print(f"alignment check (oracle weights vs full-mesh stress): {al:.3e}",
              flush=True)
        if al > 1e-6:
            print("  WARNING: support/weight alignment or ordering is wrong; "
                  "the numbers below are not trustworthy")

        E_all = {}
        for nm in ("test", "probe"):
            S_s = d[f"S_{nm}"]
            ok = np.isfinite(S_s).all(axis=1)
            Es, Ss = d[f"E_{nm}"][ok], S_s[ok]
            idx = np.random.default_rng(3).choice(
                Es.shape[0], min(a_.n_eval, Es.shape[0]), replace=False)
            E_all[nm] = (Es[idx], Ss[idx])

        res = {}
        wmin, ssum, q_dep = np.inf, [], {}
        for nm in ("test", "probe"):
            Es, Ss = E_all[nm]
            out = {k: [] for k in ("ref", "base", "maw")}
            keep = []
            q_dep[nm] = []
            for j in range(Es.shape[0]):
                E = Es[j]
                try:
                    n_sub = max(1, int(np.ceil(SUBSTEPS * np.linalg.norm(E))))
                    q = np.zeros(3)
                    for k in range(1, n_sub + 1):
                        g = rve._g(E * (k / n_sub))
                        for _ in range(MAX_IT):
                            dd, PD = dr.value_and_jac(q)
                            K, R = red_res.asm.Assemble(
                                dd * mr + g[red_res.sel])
                            PD = PD * mr[:, None]
                            dq = np.linalg.solve(PD.T @ (K @ PD), PD.T @ R)
                            q = q + dq
                            if (np.linalg.norm(dq)
                                    / max(np.linalg.norm(q), 1e-30) < TOL):
                                break
                        else:
                            raise RuntimeError("Newton failed")
                    gE = rve._g(E)
                    for nmk in ("ref", "base", "maw"):
                        red, ddc, mmc = packs[nmk]
                        if nmk == "maw":
                            wq = L.field_weights(model, q[None, :])[:, 0]
                            wmin = min(wmin, float(wq.min()))
                            ssum.append(float(wq.sum()))
                            red.asm.w_detJ = base_m * wq[:, None]
                        dd, _ = ddc.value_and_jac(q)
                        red.asm.Assemble(dd * mmc + gE[red.sel])
                        out[nmk].append(
                            rve.homogenized_stress(E, assembler=red.asm))
                    q_dep[nm].append(q.copy())
                    keep.append(j)
                except (RuntimeError, np.linalg.LinAlgError):
                    pass
            B = Ss[keep]
            res[nm] = {k: (np.linalg.norm(np.array(v) - B) / np.linalg.norm(B),
                           np.linalg.norm(np.array(v) - B, axis=1)
                           / np.linalg.norm(B, axis=1))
                       for k, v in out.items()}
            res[nm]["n"] = len(keep)
            print(f"{nm}: {len(keep)}/{Es.shape[0]} solved", flush=True)
        n_res, n_ref, n_base, n_maw = (z_res.size, z_ref.size, z_b.size,
                                       z_maw.size)

    print(f"\n=== homogenized stress error vs FOM, relative Frobenius ==="
          f"\n    residual inside Newton: classic ECM, {n_res} elements "
          f"(tol {RES_TOL:.0e}), identical in every row\n")
    print(f"{'stress rule':<28} {'pts':>4} {'in-envelope':>13} "
          f"{'out-of-envelope':>16} {'degradation':>12}")
    print("-" * 78)
    for k, lb, np_ in (("ref", "classic ECM, fixed w", n_ref),
                       ("base", "classic ECM, fixed w", n_base),
                       ("maw", "MAW-ECM, field w(q)", n_maw)):
        a, p = res["test"][k][0], res["probe"][k][0]
        print(f"{lb:<28} {np_:>4} {a:>13.4e} {p:>16.4e} {p / a:>11.1f}x")

    print("\n=== per state (median / p90 / max) ===\n")
    for k, lb, np_ in (("ref", f"classic {n_ref}", n_ref),
                       ("base", f"classic {n_base}", n_base),
                       ("maw", f"MAW {n_maw}", n_maw)):
        for nm in ("test", "probe"):
            e = res[nm][k][1]
            print(f"{lb:<12} {nm:>6} {np.median(e):11.4e} "
                  f"{np.percentile(e, 90):11.4e} {e.max():11.4e}")

    # WHY THE CONSTRAINT-ERROR GAIN DOES NOT CARRY OVER. The field is fit at q
    # taken from FOM training snapshots, but deployed at the q the HPROM-ANN's
    # own Newton solve converges to. If those differ, the field is queried off
    # the manifold it was fit on, and a trained component loses accuracy there
    # exactly as any other trained closure does. This reports how far outside
    # the training q box the deployed queries actually land.
    lo, hi = q_tr.min(axis=0), q_tr.max(axis=0)
    span = np.maximum(hi - lo, 1e-300)
    print(f"\ndeployed q vs the training q box, in units of its span:")
    Qs = {}
    for nm in ("test", "probe"):
        Q = np.asarray(q_dep[nm])
        Qs[nm] = Q
        ex = np.maximum(np.maximum((lo - Q) / span, (Q - hi) / span),
                        0.0).max(axis=1)
        print(f"  {nm:>6}: outside {int(np.sum(ex > 0))}/{Q.shape[0]}   "
              f"excursion median {np.median(ex):.4f}  max {ex.max():.4f}")

    print(f"\nstructural guarantees at DEPLOYED states, including "
          f"out-of-envelope:")
    print(f"  min weight {wmin:.3e} (>= 0 required)")
    print(f"  weight sum {min(ssum):.6f}..{max(ssum):.6f} "
          f"(n_elements {ne} required)")
    print(f"  equal-cost gain: "
          f"{res['test']['base'][0] / res['test']['maw'][0]:.2f}x in-envelope, "
          f"{res['probe']['base'][0] / res['probe']['maw'][0]:.2f}x out")
    np.savez_compressed(HERE / "deploy_maw.npz",
                        q_test=Qs["test"], q_probe=Qs["probe"],
                        **{f"{nm}_{k}": res[nm][k][1] for nm in ("test", "probe")
                           for k in ("ref", "base", "maw")})
    print("\nDEPLOY_MAW_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
