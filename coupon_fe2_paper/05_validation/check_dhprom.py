#!/usr/bin/env python3
"""Independent verification of the D-HPROM-ANN's 12129x speed-up.

A number that large deserves to be disbelieved until it survives checks that
would each catch a different way of being wrong:

  1. ARITHMETIC PLAUSIBILITY. The FOM costs 8449 ms over ~143 Newton
     iterations, so ~59 ms per full-mesh assembly. A single assembly of 10
     elements should cost ~59 * 10/1546 = 0.38 ms. If the measured 0.70 ms is
     far BELOW that, work is being skipped.
  2. IS IT DOING ANYTHING? A routine that returns a stale or E-independent
     value would be both fast and superficially plausible. Perturbing E must
     change the stress, and roughly linearly for small perturbations.
  3. EXACTLY ONE ASSEMBLY PER STATE. The speed-up rests on there being no
     Newton loop and no load ramp. Counted, not assumed, by wrapping Assemble.
  4. AGREEMENT ACROSS MESHES. The same model evaluated on the full 1546-element
     mesh and on the 75- and 10-element reduced ones must give nearly the same
     stress. If the reduced numbers agree with the full-mesh one, the reduction
     is sound; if they only agree with each other, the cubature is wrong in a
     correlated way.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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
REPEATS = 5
CUB_TOL = 1.0e-4


def rank_for(sv, tol):
    tail = np.cumsum(sv[::-1] ** 2)[::-1]
    tot = np.sum(sv ** 2)
    for i in range(1, sv.size + 1):
        if np.sqrt((tail[i] if i < sv.size else 0.0) / tot) <= tol:
            return i
    return sv.size


def main():
    import torch

    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    from reduced_mesh import ReducedAssembly
    from numpy_decoder import NumpyDecoder
    from train_nslave import build_net

    bas = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    nsl = np.load(ROOT / "04_training" / "nslave.npz")
    lab = L.load()
    ne = lab["ne"]
    fi, _ = L.split(lab["q"].shape[0])
    Phi_M, Phi_S, A_M = bas["Phi_M"], bas["Phi_S"], bas["A_M"]
    PhiMA_full = Phi_M @ A_M
    mu_m, mu_s = nsl["mu_mean"], nsl["mu_std"]

    fs = np.load(HERE / "maw_phase2_sig.npz")
    z_sm = fs["sig_10_z"]
    m_sig = dict(state={k[len("sig_10") + 5:]: fs[k] for k in fs.files
                        if k.startswith("sig_10_net_")},
                 mu=fs["sig_10_mu"], sd=fs["sig_10_sd"],
                 act=str(fs["sig_10_act"]), target_sum=float(ne))

    net = build_net(3, Phi_S.shape[1], width=int(nsl["width"]),
                    depth=int(nsl["depth"]))
    net.load_state_dict({k: torch.from_numpy(nsl[k]) for k in net.state_dict()})
    net.eval()

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        dec = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                           ROOT / "04_training" / "nslave.npz")
        full = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"
        Us, svs = L.integrand_svd(lab["C_sig"], fi)
        z_sc, w_sc, _ = L.classic_ecm(lab["C_sig"], rank_for(svs, CUB_TOL), U=Us)

        Tc = rve.T.tocsr()
        cnt = np.diff(Tc.indptr)
        Tr = np.full(rve.n_dof, -1, dtype=np.int64)
        nzm = cnt > 0
        Tr[nzm] = Tc.indices[Tc.indptr[:-1][nzm]]

        def restricted(sel):
            rows = Tr[sel]
            return dec.restrict(np.maximum(rows, 0)), (rows >= 0).astype(float)

        r75 = ReducedAssembly(full, HERE / "ck_75", z_sc, w_sc, rve,
                              np.zeros((rve.n_dof, 1)))
        r10 = ReducedAssembly(full, HERE / "ck_10", z_sm, np.ones(z_sm.size),
                              rve, np.zeros((rve.n_dof, 1)))
        base10 = np.array(r10.asm.w_detJ, dtype=float, copy=True)
        d75, m75 = restricted(r75.sel)
        d10, m10 = restricted(r10.sel)

        # --- count assembler calls (check 3) ------------------------------
        counts = {"full": 0, "r75": 0, "r10": 0}

        def wrap(asm, key):
            orig = asm.Assemble

            def counted(*a, **k):
                counts[key] += 1
                return orig(*a, **k)
            asm.Assemble = counted
        wrap(rve.assembler, "full")
        wrap(r75.asm, "r75")
        wrap(r10.asm, "r10")

        def qs_of(E):
            with torch.no_grad():
                return net(torch.from_numpy(
                    ((E - mu_m) / mu_s)[None, :])).numpy()[0]

        def d_full(E):
            u = PhiMA_full @ E + Phi_S @ qs_of(E)
            rve.assembler.Assemble(rve.T @ u + rve._g(E))
            return rve.homogenized_stress(E)

        def d_red(E, which):
            red, dcd, msk = ((r75, d75, m75) if which == 75 else
                             (r10, d10, m10))
            if which == 10:
                red.asm.w_detJ = base10 * L.field_weights(
                    m_sig, np.atleast_2d(E))[:, 0][:, None]
            dd = dcd.PhiMA @ E + dcd.Phi_S @ qs_of(E)
            red.asm.Assemble(dd * msk + rve._g(E)[red.sel])
            return rve.homogenized_stress(E, assembler=red.asm)

        S_s = d["S_test"]
        ok = np.isfinite(S_s).all(axis=1)
        Es, Ss = d["E_test"][ok], S_s[ok]
        idx = np.random.default_rng(3).choice(
            Es.shape[0], min(N_EVAL, Es.shape[0]), replace=False)
        Es, Ss = Es[idx], Ss[idx]

        # --- check 2: does it respond to E? -------------------------------
        E0 = Es[0].copy()
        S0 = d_red(E0, 10)
        for h in (1e-4, 1e-3, 1e-2):
            Ep = E0 * (1.0 + h)
            rel = (np.linalg.norm(d_red(Ep, 10) - S0)
                   / max(np.linalg.norm(S0), 1e-300))
            print(f"  perturb E by {h:.0e} -> stress changes {rel:.3e} "
                  f"(ratio {rel / h:.2f})")
        print("  (a stale or E-independent result would give 0)\n", flush=True)

        # --- checks 1 and 4: time and agree -------------------------------
        counts.update(full=0, r75=0, r10=0)
        res = {}
        for lb, fn, nel in (("full mesh", d_full, 1546),
                            ("reduced 75", lambda E: d_red(E, 75), 75),
                            ("reduced 10", lambda E: d_red(E, 10), 10)):
            best = np.inf
            vals = None
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                v = [fn(Es[j]) for j in range(Es.shape[0])]
                dt = (time.perf_counter() - t0) / Es.shape[0]
                best = min(best, dt)
                vals = np.array(v)
            e = np.linalg.norm(vals - Ss) / np.linalg.norm(Ss)
            res[lb] = (best, e, vals, nel)
            print(f"  {lb:<11} {nel:>5} elems  {best * 1e3:8.3f} ms/state  "
                  f"error vs FOM {e:.4e}", flush=True)

        print(f"\n  assembler calls for {Es.shape[0]} states x {REPEATS} "
              f"repeats = {Es.shape[0] * REPEATS} expected each:")
        for k, v in counts.items():
            print(f"    {k:<5} {v:>6}  "
                  f"{'OK' if v == Es.shape[0] * REPEATS else 'MISMATCH'}")

        print("\n  agreement between meshes (relative Frobenius):")
        for a in ("reduced 75", "reduced 10"):
            dif = (np.linalg.norm(res[a][2] - res["full mesh"][2])
                   / np.linalg.norm(res["full mesh"][2]))
            print(f"    {a:<11} vs full mesh  {dif:.4e}")

        fom_ms = 8449.45
        print(f"\n  FOM reference {fom_ms:.0f} ms/state over ~143 Newton "
              f"iterations = {fom_ms / 143:.1f} ms per full assembly")
        for lb in ("full mesh", "reduced 75", "reduced 10"):
            t, _, _, nel = res[lb]
            pred = fom_ms / 143 * nel / 1546
            print(f"    {lb:<11} predicted {pred:8.3f} ms  "
                  f"measured {t * 1e3:8.3f} ms  ratio {t * 1e3 / pred:5.2f}x")
            print(f"    {'':<11} speed-up {fom_ms / (t * 1e3):9.0f}x")

    print("\nCHECK_DHPROM_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
