#!/usr/bin/env python3
"""Wall-clock timing of the full hierarchy, measured serially on an idle machine.

THE NUMBER THIS SUPPLIES. Every accuracy result so far was reported with no
timing at all, deliberately, because something was always sharing the CPU and a
speed-up measured under contention is worthless. Accuracy is unaffected by
contention; speed-up is not. This is the run that must be alone.

It also settles whether MAW-ECM pays. The adaptive rule uses 20 reduced
elements against 163 for the classic cubature -- 8x fewer -- but takes 23% more
Newton iterations (weights frozen within each iteration, so the tangent is
inconsistent) and evaluates a weight field and rescales w_detJ on every
iteration. Whether the net is 6x or 1.5x is not something the element counts
answer.

Every model solves the SAME states with the SAME tolerance and the SAME ramp
density, asserted at run time rather than assumed -- an earlier comparison in
this project was invalidated by a PROM using a coarser ramp than the FOM.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
import time
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
CUB_TOL = 1.0e-4          # classic cubature tolerance for the manifold rules
REPEATS = 1


def rank_for(sv, tol):
    tail = np.cumsum(sv[::-1] ** 2)[::-1]
    tot = np.sum(sv ** 2)
    for i in range(1, sv.size + 1):
        if np.sqrt((tail[i] if i < sv.size else 0.0) / tot) <= tol:
            return i
    return sv.size


def load_field(fl, key, ne):
    return dict(state={k[len(key) + 5:]: fl[k] for k in fl.files
                       if k.startswith(f"{key}_net_")},
                mu=fl[f"{key}_mu"], sd=fl[f"{key}_sd"],
                act=str(fl[f"{key}_act"]), target_sum=float(ne))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    a_ = ap.parse_args()

    import torch

    import linear_prom as lp
    import periodic_fom as pf
    pf.NEWTON_TOL = TOL
    lp.NEWTON_TOL = TOL
    lp.SUBSTEPS_PER_UNIT_STRAIN = pf.SUBSTEPS_PER_UNIT_STRAIN
    SUB = float(pf.SUBSTEPS_PER_UNIT_STRAIN)
    assert lp.SUBSTEPS_PER_UNIT_STRAIN == pf.SUBSTEPS_PER_UNIT_STRAIN
    assert lp.NEWTON_TOL == pf.NEWTON_TOL == TOL

    from periodic_fom import PeriodicRVE
    from linear_prom import LinearPROM
    from _material_law_guard_claude import true_neo_hookean_active
    from reduced_mesh import ReducedAssembly
    from numpy_decoder import NumpyDecoder
    from train_nslave import build_net

    bas = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    sup = np.load(HERE / "ecm_supports.npz")
    nsl = np.load(ROOT / "04_training" / "nslave.npz")
    lab = L.load()
    ne = lab["ne"]
    fi, _ = L.split(lab["q"].shape[0])

    Phi = bas["Phi_ROM"]
    Phi_M, Phi_S, A_M = bas["Phi_M"], bas["Phi_S"], bas["A_M"]
    mu_m, mu_s = nsl["mu_mean"], nsl["mu_std"]

    fr = np.load(HERE / "maw_res_long10.npz")
    fs = np.load(HERE / "maw_phase2_sig.npz")
    z_rm, z_sm = fr["res_10_z"], fs["sig_10_z"]
    m_res = load_field(fr, "res_10", ne)
    m_sig = load_field(fs, "sig_10", ne)

    net = build_net(3, Phi_S.shape[1], width=int(nsl["width"]),
                    depth=int(nsl["depth"]))
    net.load_state_dict({k: torch.from_numpy(nsl[k]) for k in net.state_dict()})
    net.eval()

    print(f"tol {TOL:.0e}, ramp {SUB:.0f}/unit strain, "
          f"{a_.n_eval} in-envelope states, {REPEATS} repeat(s)\n", flush=True)

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        prom = LinearPROM(rve, Phi)
        dec = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                           ROOT / "04_training" / "nslave.npz")
        full = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"

        Ur, svr = L.integrand_svd(lab["C_res"], fi)
        z_rc, w_rc, _ = L.classic_ecm(lab["C_res"], rank_for(svr, CUB_TOL), U=Ur)
        Us, svs = L.integrand_svd(lab["C_sig"], fi)
        z_sc, w_sc, _ = L.classic_ecm(lab["C_sig"], rank_for(svs, CUB_TOL), U=Us)

        # linear HPROM keeps its own supports, built for the 39-mode basis
        hl_r = ReducedAssembly(full, HERE / "tm_hlr", sup["z_res"], sup["w_res"],
                               rve, prom.TPhi)
        hl_s = ReducedAssembly(full, HERE / "tm_hls", sup["z_sig"], sup["w_sig"],
                               rve, prom.TPhi)

        Tc = rve.T.tocsr()
        cnt = np.diff(Tc.indptr)
        Tr = np.full(rve.n_dof, -1, dtype=np.int64)
        nzm = cnt > 0
        Tr[nzm] = Tc.indices[Tc.indptr[:-1][nzm]]

        def restricted(sel):
            rows = Tr[sel]
            return dec.restrict(np.maximum(rows, 0)), (rows >= 0).astype(float)

        an_r = ReducedAssembly(full, HERE / "tm_anr", z_rc, w_rc, rve,
                               np.zeros((rve.n_dof, 1)))
        an_s = ReducedAssembly(full, HERE / "tm_ans", z_sc, w_sc, rve,
                               np.zeros((rve.n_dof, 1)))
        mw_r = ReducedAssembly(full, HERE / "tm_mwr", z_rm, np.ones(z_rm.size),
                               rve, np.zeros((rve.n_dof, 1)))
        mw_s = ReducedAssembly(full, HERE / "tm_mws", z_sm, np.ones(z_sm.size),
                               rve, np.zeros((rve.n_dof, 1)))
        base_r = np.array(mw_r.asm.w_detJ, dtype=float, copy=True)
        base_s = np.array(mw_s.asm.w_detJ, dtype=float, copy=True)

        dc_ar, mk_ar = restricted(an_r.sel)
        dc_as, mk_as = restricted(an_s.sel)
        dc_mr, mk_mr = restricted(mw_r.sel)
        dc_ms, mk_ms = restricted(mw_s.sel)

        def fom(E):
            rve.solve(E)
            return rve.homogenized_stress(E)

        def prom_run(E):
            return prom.solve(E)[0]

        def hprom_lin(E):
            n = max(1, int(np.ceil(SUB * np.linalg.norm(E))))
            q = np.zeros(Phi.shape[1])
            for kk in range(1, n + 1):
                # _g_at instead of _g(...)[sel]: bit-identical, 6.4x cheaper.
                gs = rve._g_at(E * (kk / n), hl_r.sel)
                for _ in range(MAX_IT):
                    K, R = hl_r.asm.Assemble(hl_r.TPhi @ q + gs)
                    dq = np.linalg.solve(hl_r.TPhi.T @ (K @ hl_r.TPhi),
                                         hl_r.TPhi.T @ R)
                    q = q + dq
                    if np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30) < TOL:
                        break
                else:
                    raise RuntimeError("HPROM Newton failed")
            hl_s.asm.Assemble(hl_s.TPhi @ q + rve._g_at(E, hl_s.sel))
            return rve.homogenized_stress(E, assembler=hl_s.asm)

        def manifold(E, adaptive):
            if adaptive:
                red, dcd, msk, base = mw_r, dc_mr, mk_mr, base_r
            else:
                red, dcd, msk = an_r, dc_ar, mk_ar
            n = max(1, int(np.ceil(SUB * np.linalg.norm(E))))
            q = np.zeros(3)
            for kk in range(1, n + 1):
                gs = rve._g_at(E * (kk / n), red.sel)
                for _ in range(MAX_IT):
                    if adaptive:
                        red.asm.w_detJ = base * L.field_weights(
                            m_res, q[None, :])[:, 0][:, None]
                    dd, PD = dcd.value_and_jac(q)
                    K, R = red.asm.Assemble(dd * msk + gs)
                    PD = PD * msk[:, None]
                    dq = np.linalg.solve(PD.T @ (K @ PD), PD.T @ R)
                    q = q + dq
                    if np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30) < TOL:
                        break
                else:
                    raise RuntimeError("manifold Newton failed")
            if adaptive:
                mw_s.asm.w_detJ = base_s * L.field_weights(
                    m_sig, q[None, :])[:, 0][:, None]
                dd, _ = dc_ms.value_and_jac(q)
                mw_s.asm.Assemble(dd * mk_ms + rve._g_at(E, mw_s.sel))
                return rve.homogenized_stress(E, assembler=mw_s.asm)
            dd, _ = dc_as.value_and_jac(q)
            an_s.asm.Assemble(dd * mk_as + rve._g_at(E, an_s.sel))
            return rve.homogenized_stress(E, assembler=an_s.asm)

        def dhprom(E, adaptive):
            """No solve at all: q comes straight from E through the network, and
            the only cost left is the homogenized stress -- which is why it must
            be evaluated on the HYPER-REDUCED stress mesh, not the full one.
            Assembling the full 1546-element mesh here would have thrown away
            the entire point of the model and inflated its cost by the ratio of
            the meshes.
            """
            with torch.no_grad():
                qs = net(torch.from_numpy(((E - mu_m) / mu_s)[None, :])).numpy()[0]
            # E IS the latent coordinate here -- that is what makes the model
            # "direct": no least-squares map from mu to q. PhiMA already holds
            # Phi_M @ A_M, so the input to it is E, NOT A_M @ E. Applying A_M
            # twice was worth a relative stress error of 1.5e+02.
            q = E
            if adaptive:
                mw_s.asm.w_detJ = base_s * L.field_weights(
                    m_sig, np.atleast_2d(q))[:, 0][:, None]
                red, dcd, msk = mw_s, dc_ms, mk_ms
            else:
                red, dcd, msk = an_s, dc_as, mk_as
            dd = dcd.PhiMA @ E + dcd.Phi_S @ qs
            red.asm.Assemble(dd * msk + rve._g_at(E, red.sel))
            return rve.homogenized_stress(E, assembler=red.asm)

        S_s = d["S_test"]
        ok = np.isfinite(S_s).all(axis=1)
        Es, Ss = d["E_test"][ok], S_s[ok]
        idx = np.random.default_rng(3).choice(
            Es.shape[0], min(a_.n_eval, Es.shape[0]), replace=False)
        Es, Ss = Es[idx], Ss[idx]

        MODELS = (
            ("FOM", 1546, fom),
            ("PROM (39 modes)", 1546, prom_run),
            ("HPROM (39 modes)", int(np.asarray(sup["z_res"]).size
                                     + np.asarray(sup["z_sig"]).size),
             hprom_lin),
            ("HPROM-ANN (classic cub)", int(z_rc.size + z_sc.size),
             lambda E: manifold(E, False)),
            ("MAW-HPROM-ANN", int(z_rm.size + z_sm.size),
             lambda E: manifold(E, True)),
            ("D-HPROM-ANN (classic cub)", int(z_sc.size),
             lambda E: dhprom(E, False)),
            ("MAW-D-HPROM-ANN", int(z_sm.size), lambda E: dhprom(E, True)),
        )

        out = {}
        for lb, nel, fn in MODELS:
            if fn is None:
                continue
            vals, t, nf = [], 0.0, 0
            keep = []
            try:
                for rep in range(REPEATS):
                    vals, keep, t = [], [], 0.0
                    for j in range(Es.shape[0]):
                        try:
                            t0 = time.perf_counter()
                            v = fn(Es[j])
                            t += time.perf_counter() - t0
                            vals.append(v)
                            keep.append(j)
                        except Exception:                      # noqa: BLE001
                            nf += 1
            except Exception as e:                             # noqa: BLE001
                print(f"  {lb}: unavailable ({type(e).__name__}: {e})")
                continue
            if not keep:
                print(f"  {lb}: all states failed")
                continue
            Aa, Bb = np.array(vals), Ss[keep]
            fr_ = np.linalg.norm(Aa - Bb) / np.linalg.norm(Bb)
            out[lb] = dict(t=t / len(keep), frob=fr_, n=len(keep),
                           nf=nf, nel=nel)
            print(f"  {lb:<26} {t / len(keep) * 1e3:9.2f} ms/state  "
                  f"frob {fr_:.4e}  n={len(keep)} fail={nf}", flush=True)

    if "FOM" not in out:
        print("\nNOTE: the FOM row is missing; speed-ups are relative to the "
              "slowest available model instead.")
    ref = out.get("FOM") or max(out.values(), key=lambda v: v["t"])
    print("\n=== hierarchy, in-envelope, serial timing ===\n")
    print(f"{'model':<26} {'elems':>7} {'ms/state':>10} {'speed-up':>10} "
          f"{'stress error':>13}")
    print("-" * 70)
    for lb, v in out.items():
        print(f"{lb:<26} {v['nel']:>7} {v['t'] * 1e3:>10.2f} "
              f"{ref['t'] / v['t']:>9.1f}x {v['frob']:>13.4e}")
    np.savez_compressed(HERE / "timing_hierarchy.npz",
                        **{f"{k}_t": v["t"] for k, v in out.items()},
                        **{f"{k}_e": v["frob"] for k, v in out.items()})
    print("\nTIMING_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
