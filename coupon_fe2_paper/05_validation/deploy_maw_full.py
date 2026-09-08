#!/usr/bin/env python3
"""The full MAW-HPROM-ANN: adaptive weights on BOTH rules, residual inside the
Newton loop.

WHY THIS EXISTS, AND WHY IT SHOULD HAVE EXISTED FIRST. Every residual number so
far was judged against the classic 88-point rule's constraint error of
1.5805e-03, and MAW never came close, so I concluded the residual could not be
hyperreduced adaptively. That yardstick was wrong. What deployment needs is
already measured:

    residual rule            constraint error      deployed stress error
    classic ECM, 40 points        2.08e-02               9.2952e-04
    classic ECM, 88 points        1.58e-03               9.3205e-04

A 13x refinement of the residual cubature moved the deployed error by 0.3%.
The floor is set by the manifold closure, not by the quadrature, so a residual
rule at ~2e-02 is already sufficient -- and the 10-point MAW rule delivers
1.97e-02, indistinguishable from the classic 40-point rule. Judging it against
1.58e-03 was measuring an intermediate again, the same mistake this project has
now made three times.

TANGENT. With w = w(q) the residual depends on q through the weights too, so a
consistent tangent needs dw/dq. This first pass FREEZES the weights within each
Newton iteration (evaluating them at the current q, but not differentiating
them), which is a modified Newton: the fixed point is unchanged, only the
convergence rate. Iteration counts are reported so that the choice can be
judged rather than assumed -- if they blow up, the analytic dw/dq already
exists in the previous project's `mawecm_ann_jacobian_claude.py`.

Both rules keep their structural guarantees at every query, in and out of the
training envelope: w = n_elements * softmax(logits(q)) gives w >= 0 and
sum(w) = n_elements exactly.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
from pathlib import Path

import numpy as np

import maw_lab as L

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT / "00_rve"), str(ROOT / "04_training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_EVAL = 40
TOL = 1.0e-10
MAX_IT = 200
SUBSTEPS = 200.0
RES_TOL = 1.0e-4       # classic residual rule, for the reference row
FLOOR = 9.3205e-04     # deployed stress error with the classic 88-point rule


def load_field(fl, key, ne):
    return dict(state={k[len(key) + 5:]: fl[k] for k in fl.files
                       if k.startswith(f"{key}_net_")},
                mu=fl[f"{key}_mu"], sd=fl[f"{key}_sd"],
                act=str(fl[f"{key}_act"]), target_sum=float(ne))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--res-pts", type=int, default=10)
    ap.add_argument("--sig-pts", type=int, default=10)
    ap.add_argument("--res-fields", default="maw_phase2_res.npz",
                    help="npz holding the residual weight field; the 100k-epoch "
                         "annealed field lives in maw_res_long10.npz, the "
                         "20k-epoch sweep field in maw_phase2_res.npz")
    a_ = ap.parse_args()

    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    from reduced_mesh import ReducedAssembly
    from numpy_decoder import NumpyDecoder

    d = np.load(ROOT / "03_data" / "data.npz")
    lab = L.load()
    ne = lab["ne"]
    fi, _ = L.split(lab["q"].shape[0])

    fr = np.load(HERE / a_.res_fields)
    fs = np.load(HERE / "maw_phase2_sig.npz")
    z_res_maw = fr[f"res_{a_.res_pts}_z"]
    z_sig_maw = fs[f"sig_{a_.sig_pts}_z"]
    m_res = load_field(fr, f"res_{a_.res_pts}", ne)
    m_sig = load_field(fs, f"sig_{a_.sig_pts}", ne)

    A_r = L.blocks(lab["C_res"], z_res_maw)
    b_r = L.targets(lab["C_res"])
    W_r = L.field_weights(m_res, lab["q"])
    print(f"MAW residual rule: {z_res_maw.size} points, "
          f"constraint error (all states) "
          f"{np.median(L.const_err(A_r, b_r, W_r)):.4e}")
    print(f"MAW stress rule:   {z_sig_maw.size} points")
    print(f"deployed floor (classic 88-pt residual, 75-pt stress): "
          f"{FLOOR:.4e}\n", flush=True)

    def rank_for(sv, tol):
        tail = np.cumsum(sv[::-1] ** 2)[::-1]
        tot = np.sum(sv ** 2)
        for i in range(1, sv.size + 1):
            if np.sqrt((tail[i] if i < sv.size else 0.0) / tot) <= tol:
                return i
        return sv.size

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        dec = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                           ROOT / "04_training" / "nslave.npz")
        full = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"

        Ur, svr = L.integrand_svd(lab["C_res"], fi)
        z_res_cl, w_res_cl, _ = L.classic_ecm(lab["C_res"],
                                              rank_for(svr, RES_TOL), U=Ur)
        Us, svs = L.integrand_svd(lab["C_sig"], fi)
        z_sig_cl, w_sig_cl, _ = L.classic_ecm(lab["C_sig"],
                                              rank_for(svs, RES_TOL), U=Us)

        Tcsr = rve.T.tocsr()
        cnt = np.diff(Tcsr.indptr)
        Trows = np.full(rve.n_dof, -1, dtype=np.int64)
        nzm = cnt > 0
        if int(cnt.max()) != 1 or int(cnt[nzm].min()) != 1:
            raise RuntimeError("T is not one-nonzero-per-row")
        Trows[nzm] = Tcsr.indices[Tcsr.indptr[:-1][nzm]]

        def restricted(sel):
            rows = Trows[sel]
            return dec.restrict(np.maximum(rows, 0)), (rows >= 0).astype(float)

        # residual assemblers
        red_rc = ReducedAssembly(full, HERE / "fd_rc", z_res_cl, w_res_cl,
                                 rve, np.zeros((rve.n_dof, 1)))
        red_rm = ReducedAssembly(full, HERE / "fd_rm", z_res_maw,
                                 np.ones(z_res_maw.size), rve,
                                 np.zeros((rve.n_dof, 1)))
        base_rm = np.array(red_rm.asm.w_detJ, dtype=float, copy=True)
        # stress assemblers
        red_sc = ReducedAssembly(full, HERE / "fd_sc", z_sig_cl, w_sig_cl,
                                 rve, np.zeros((rve.n_dof, 1)))
        red_sm = ReducedAssembly(full, HERE / "fd_sm", z_sig_maw,
                                 np.ones(z_sig_maw.size), rve,
                                 np.zeros((rve.n_dof, 1)))
        base_sm = np.array(red_sm.asm.w_detJ, dtype=float, copy=True)

        dr_c, mk_rc = restricted(red_rc.sel)
        dr_m, mk_rm = restricted(red_rm.sel)
        ds_c, mk_sc = restricted(red_sc.sel)
        ds_m, mk_sm = restricted(red_sm.sel)

        def solve(E, adaptive):
            """Newton on the reduced equilibrium. Weights frozen per iteration."""
            red, dcd, msk = ((red_rm, dr_m, mk_rm) if adaptive
                             else (red_rc, dr_c, mk_rc))
            n_sub = max(1, int(np.ceil(SUBSTEPS * np.linalg.norm(E))))
            q = np.zeros(3)
            nit = 0
            for kk in range(1, n_sub + 1):
                g = rve._g(E * (kk / n_sub))
                for _ in range(MAX_IT):
                    if adaptive:
                        wq = L.field_weights(m_res, q[None, :])[:, 0]
                        red.asm.w_detJ = base_rm * wq[:, None]
                    dd, PD = dcd.value_and_jac(q)
                    K, R = red.asm.Assemble(dd * msk + g[red.sel])
                    PD = PD * msk[:, None]
                    dq = np.linalg.solve(PD.T @ (K @ PD), PD.T @ R)
                    q = q + dq
                    nit += 1
                    if np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30) < TOL:
                        break
                else:
                    raise RuntimeError("Newton failed")
            return q, nit

        def stress(E, q, adaptive):
            red, dcd, msk = ((red_sm, ds_m, mk_sm) if adaptive
                             else (red_sc, ds_c, mk_sc))
            if adaptive:
                wq = L.field_weights(m_sig, q[None, :])[:, 0]
                red.asm.w_detJ = base_sm * wq[:, None]
            dd, _ = dcd.value_and_jac(q)
            red.asm.Assemble(dd * msk + rve._g(E)[red.sel])
            return rve.homogenized_stress(E, assembler=red.asm)

        CASES = (("classic res + classic sig", False, False),
                 ("classic res + MAW sig", False, True),
                 ("MAW res + classic sig", True, False),
                 ("MAW res + MAW sig", True, True))

        res = {}
        for nm in ("test", "probe"):
            S_s = d[f"S_{nm}"]
            ok = np.isfinite(S_s).all(axis=1)
            Es, Ss = d[f"E_{nm}"][ok], S_s[ok]
            idx = np.random.default_rng(3).choice(
                Es.shape[0], min(a_.n_eval, Es.shape[0]), replace=False)
            Es, Ss = Es[idx], Ss[idx]
            for lb, ar, asg in CASES:
                out, keep, its, nf = [], [], [], 0
                for j in range(Es.shape[0]):
                    try:
                        q, nit = solve(Es[j], ar)
                        out.append(stress(Es[j], q, asg))
                        its.append(nit)
                        keep.append(j)
                    except (RuntimeError, np.linalg.LinAlgError):
                        nf += 1
                if not keep:
                    res[(nm, lb)] = (np.nan, np.nan, 0, nf, np.nan)
                    continue
                Aa, Bb = np.array(out), Ss[keep]
                fr_ = np.linalg.norm(Aa - Bb) / np.linalg.norm(Bb)
                pe = (np.linalg.norm(Aa - Bb, axis=1)
                      / np.linalg.norm(Bb, axis=1))
                res[(nm, lb)] = (fr_, np.median(pe), len(keep), nf,
                                 float(np.mean(its)))
                print(f"  {nm:>6} | {lb:<26} frob {fr_:.4e}  "
                      f"median {np.median(pe):.4e}  n={len(keep)} fail={nf}  "
                      f"its/solve {np.mean(its):.0f}", flush=True)

    print("\n=== homogenized stress error vs FOM, relative Frobenius ===\n")
    print(f"{'residual rule':<14} {'stress rule':<14} {'in-envelope':>13} "
          f"{'out-of-env':>13} {'degrad':>8} {'its':>6}")
    print("-" * 74)
    for lb, ar, asg in CASES:
        a = res[("test", lb)]
        p = res[("probe", lb)]
        rl = "MAW 10" if ar else "classic 88"
        sl = "MAW 10" if asg else "classic 75"
        print(f"{rl:<14} {sl:<14} {a[0]:>13.4e} {p[0]:>13.4e} "
              f"{p[0] / a[0]:>7.1f}x {a[4]:>6.0f}")
    print(f"\nfloor (classic + classic, measured earlier): {FLOOR:.4e}")
    print("\nDEPLOY_FULL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
