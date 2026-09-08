#!/usr/bin/env python3
"""The ITERATIVE HPROM-ANN: nonlinear-manifold ROM that SOLVES the reduced
equilibrium, with fixed-weight ECM rules for residual and stress.

WHAT CHANGES relative to the linear HPROM, and it is not just the unknown
count:

                 unknowns          tangent space
    HPROM        q in R^39         Phi, CONSTANT
    HPROM-ANN    q_M in R^3        Phi_D(q_M) = Phi_M A_M + Phi_S J_N(q_M)

so the projected equilibrium is Phi_D(q_M)^T f_int(d(q_M)) = 0, three
equations in three unknowns, and the projection itself moves along the
manifold.

TWO CONSEQUENCES.

First, the residual integrand now has 3 components instead of 39, so a much
lower-rank object is expected and the ECM should need far fewer points. That
is a genuine advantage of the manifold route, not just a smaller solve.

Second, this is exactly where MAW-ECM comes from: a FIXED-weight rule is being
fitted for a projection that VARIES with q_M. Hernandez's adaptive weights
exist to remove that mismatch, and the error left here is the measured
motivation for them rather than an argument for them.

LINEARIZATION. The consistent Newton tangent is

    dr/dq = Phi_D^T K Phi_D + (d Phi_D / dq)^T f_int

whose second term needs the decoder's second derivative -- the reason
twice-differentiability is required of it, and the reason tanh was chosen over
ReLU. This starts with GAUSS-NEWTON, dropping that term: the fixed point is
identical, convergence is linear rather than quadratic, and it is robust. If
convergence proves poor the second term is added, measured rather than
assumed.

The stress rule is reused unchanged: the stress integrand <P> does not involve
the projection at all, so the 73-element rule built for the linear HPROM
applies here without modification.
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

N_EVAL = 40
TOL = 1.0e-10
MAX_IT = 60
SUBSTEPS = 200.0
SVD_TOLS = (1e-3, 1e-4, 1e-5)
N_SNAP = 495


class Decoder:
    """d_red(q_M) = Phi_M A_M q_M + Phi_S N(q_M), with its Jacobian."""

    def __init__(self, basis_npz, net_npz):
        import torch
        from train_nslave import build_net
        b, m = np.load(basis_npz), np.load(net_npz)
        self.Phi_M, self.Phi_S, self.A_M = b["Phi_M"], b["Phi_S"], b["A_M"]
        self.mu_m, self.mu_s = m["mu_mean"], m["mu_std"]
        self.net = build_net(3, self.Phi_S.shape[1], width=int(m["width"]),
                             depth=int(m["depth"]))
        self.net.load_state_dict({k: torch.from_numpy(m[k])
                                  for k in self.net.state_dict()})
        self.net.eval()
        self._torch = torch

    def _x(self, q):
        return (q - self.mu_m) / self.mu_s

    def value_and_jac(self, q):
        torch = self._torch
        x = torch.from_numpy(self._x(np.asarray(q, dtype=float)))
        with torch.no_grad():
            ns = self.net(x[None, :]).numpy()[0]
        J = torch.autograd.functional.jacobian(
            lambda v: self.net(v[None, :]).reshape(-1), x).numpy()
        J = J / self.mu_s[None, :]                    # chain rule of the scaling
        d = self.Phi_M @ (self.A_M @ q) + self.Phi_S @ ns
        Phi_D = self.Phi_M @ self.A_M + self.Phi_S @ J
        return d, Phi_D


def build_manifold_integrand(rve, dec, q_train, E_train, n_snap, verbose=True):
    """Element-wise contribution to Phi_D(q)^T f_int, 3 components."""
    a = rve.assembler
    ne, ndl = a.n_elems, a.n_local_dof
    dofs = np.asarray(a.rows_R, dtype=np.int64).reshape(ne, ndl)
    step = max(1, q_train.shape[0] // n_snap)
    sel = np.arange(0, q_train.shape[0], step)[:n_snap]
    C = np.empty((ne, sel.size * 3))
    t0 = time.perf_counter()
    for k, j in enumerate(sel):
        q = q_train[j]
        d, Phi_D = dec.value_and_jac(q)
        TPhiD = np.asarray(rve.T @ Phi_D)
        a.Assemble(rve.T @ d + rve._g(E_train[j]))
        f_int = a._f_int.reshape(ne, ndl)
        C[:, 3 * k:3 * (k + 1)] = np.einsum("eld,el->ed", TPhiD[dofs], f_int)
    if verbose:
        print(f"  manifold integrand matrix {C.shape} in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    G = C @ C.T
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    return np.ascontiguousarray(V[:, order]), np.sqrt(np.clip(w[order], 0.0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    a_ = ap.parse_args()

    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    from reduced_mesh import ReducedAssembly
    from hprom_full import rank_for, run_ecm
    from numpy_decoder import NumpyDecoder

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    sup = np.load(HERE / "ecm_supports.npz")

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        # Closed-form NumPy decoder, no torch in the Newton loop: measured
        # 42.4 us/call restricted against 2959 us for torch's reverse-mode
        # autograd, i.e. 70x, and exact to 3.6e-16 on the Jacobian.
        dec = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                           ROOT / "04_training" / "nslave.npz")

        q_train = np.ascontiguousarray(b["q_M"].T)
        E_train = b["E_train"]
        print("building the manifold-projected residual integrand:", flush=True)
        U, sv = build_manifold_integrand(rve, dec, q_train, E_train, N_SNAP)
        print(f"  {'svd_tol':>8} {'rank':>5}")
        for st in SVD_TOLS:
            print(f"  {st:8.0e} {rank_for(sv, st):5d}", flush=True)

        full = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"
        # The dummy basis argument is deliberate: ReducedAssembly precomputes
        # TPhi[sel] for a FIXED basis, but the manifold tangent Phi_D varies
        # with q, so only its `sel` (reduced dof -> full dof map) is used here
        # and the projection is formed per iteration.
        red_sig = ReducedAssembly(full, HERE / "ann_sig", sup["z_sig"],
                                  sup["w_sig"], rve, np.zeros((rve.n_dof, 1)))
        # T maps independent dofs to all dofs, with AT MOST one nonzero per
        # row -- the PINNED dofs have an all-zero row, since their value comes
        # entirely from g(E). So T.argmax would map them to independent dof 0
        # instead of to nothing, silently injecting a wrong contribution at
        # those dofs. Only 2 dofs are pinned, but if either lands in a reduced
        # mesh the answer is wrong, so the map is built explicitly with -1
        # marking "no independent dof" and the decoder's contribution is
        # zeroed there.
        Tcsr = rve.T.tocsr()
        counts = np.diff(Tcsr.indptr)
        Trows = np.full(rve.n_dof, -1, dtype=np.int64)
        nz = counts > 0
        if int(counts.max()) != 1 or int(counts[nz].min()) != 1:
            raise RuntimeError(f"T has rows with {counts.max()} nonzeros; the "
                               "one-per-row assumption does not hold")
        Trows[nz] = Tcsr.indices[Tcsr.indptr[:-1][nz]]
        print(f"T: {int(np.sum(nz))} dofs map to an independent dof, "
              f"{int(np.sum(~nz))} are pinned", flush=True)

        def make_restricted(sel):
            rows = Trows[sel]
            mask = (rows >= 0).astype(float)
            return dec.restrict(np.maximum(rows, 0)), mask

        dec_sig, mask_sig = make_restricted(red_sig.sel)

        E_all = {}
        for nm in ("test", "probe"):
            S_s = d[f"S_{nm}"]
            ok = np.isfinite(S_s).all(axis=1)
            Es, Ss = d[f"E_{nm}"][ok], S_s[ok]
            idx = np.random.default_rng(3).choice(
                Es.shape[0], min(a_.n_eval, Es.shape[0]), replace=False)
            E_all[nm] = (Es[idx], Ss[idx])

        print(f"\n{'svd_tol':>8} {'elems':>6} | {'set':>6} {'frob':>11} "
              f"{'median':>11} {'max':>11} {'its':>6} {'s':>7} {'fail':>5}")
        for st in SVD_TOLS:
            r = rank_for(sv, st)
            z, w = run_ecm(np.ascontiguousarray(U[:, :r]))
            red = ReducedAssembly(full, HERE / f"ann_res_{r}", np.sort(z),
                                  w[np.argsort(z)], rve, np.zeros((rve.n_dof, 1)))
            dec_res, mask_res = make_restricted(red.sel)
            gsel_res, gsel_sig = red.sel, red_sig.sel
            for nm in ("test", "probe"):
                Es, Ss = E_all[nm]
                out, its, t, nfail = [], [], 0.0, 0
                keep = []
                for j in range(Es.shape[0]):
                    E = Es[j]
                    try:
                        t0 = time.perf_counter()
                        n_sub = max(1, int(np.ceil(SUBSTEPS * np.linalg.norm(E))))
                        q = np.zeros(3)
                        nit = 0
                        for k in range(1, n_sub + 1):
                            Et = E * (k / n_sub)
                            g = rve._g(Et)
                            for _ in range(MAX_IT):
                                dd, PD = dec_res.value_and_jac(q)
                                dd = dd * mask_res
                                PD = PD * mask_res[:, None]
                                K, R = red.asm.Assemble(dd + g[gsel_res])
                                Kr = PD.T @ (K @ PD)
                                rr = PD.T @ R
                                dq = np.linalg.solve(Kr, rr)
                                q = q + dq
                                nit += 1
                                if (np.linalg.norm(dq)
                                        / max(np.linalg.norm(q), 1e-30) < TOL):
                                    break
                            else:
                                raise RuntimeError("HPROM-ANN Newton failed")
                        dd, _ = dec_sig.value_and_jac(q)
                        red_sig.asm.Assemble(dd * mask_sig + rve._g(E)[gsel_sig])
                        S = rve.homogenized_stress(E, assembler=red_sig.asm)
                        t += time.perf_counter() - t0
                        its.append(nit)
                        keep.append(j)
                        out.append(S)
                    except (RuntimeError, np.linalg.LinAlgError):
                        nfail += 1
                A, B = np.array(out), Ss[keep]
                if len(keep) == 0:
                    print(f"{st:8.0e} {z.size:6d} | {nm:>6} {'all failed':>11}")
                    continue
                per = np.linalg.norm(A - B, axis=1) / np.linalg.norm(B, axis=1)
                fr = np.linalg.norm(A - B) / np.linalg.norm(B)
                print(f"{st:8.0e} {z.size:6d} | {nm:>6} {fr:11.4e} "
                      f"{np.median(per):11.4e} {per.max():11.4e} "
                      f"{np.mean(its):6.0f} {t:7.1f} {nfail:5d}", flush=True)
    print("\nHPROM_ANN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
