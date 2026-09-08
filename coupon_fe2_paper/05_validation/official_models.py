#!/usr/bin/env python3
"""The three official reduced models, defined once.

WHY THIS MODULE EXISTS. Through this whole study every script rebuilt its own
HPROM-ANN -- its own supports, its own restricted decoder, its own dof
bookkeeping -- and the duplication cost real errors: a stress rule paired with
a support in the wrong order, a D-HPROM-ANN evaluated on the full mesh, `A_M`
applied twice. Stage 06 (macro FE^2) and the paper's tables should import the
models, not re-derive them.

THE THREE, and nothing else:

    HPROM             linear 39-mode basis, classic fixed-weight ECM
                      208 elements (135 residual + 73 stress)
    MAW-HPROM-ANN     3-latent manifold, adaptive weights on both rules,
                      residual inside the Newton loop, 20 elements (10 + 10)
    MAW-D-HPROM-ANN   3-latent manifold, no solve at all: q comes from E
                      through the closure network, 10 elements (stress only)

FOM and PROM are exposed too, as references rather than deliverables.

MEASURED, in-envelope, 40 states, serially on an idle machine:

    model              elements   ms/state   speed-up   stress error
    FOM                    1546    8415.95        1.0x    4.1407e-15
    PROM (39 modes)        1546    1619.27        5.2x    4.7724e-07
    HPROM                   208     169.41       49.7x    1.5339e-04
    MAW-HPROM-ANN            20      72.61      115.9x    1.8761e-03
    MAW-D-HPROM-ANN          10       0.51    16542.4x    8.3409e-04

The direct model is both the fastest and MORE accurate in-envelope than the one
that solves, because its closure is regressed on FOM snapshots while the
HPROM-ANN additionally carries the error of an approximate reduced equilibrium.
That ordering reverses out of envelope, where the solve is what generalizes;
see MAW_FINDINGS.md.

ADAPTIVE WEIGHTS carry both structural guarantees at every query, in and out of
the training envelope, by construction rather than by fit:

    w(q) = n_elements * softmax(logits(q))   =>   w >= 0,  sum(w) = n_elements

The second is exactly the volume row of the integration conditions, so that row
is satisfied identically and only the physical rows are learned.

The residual's tangent FREEZES the weights within each Newton iteration, so it
is a modified Newton: same fixed point, 23% more iterations. dw/dq would remove
that, and the analytic Jacobian exists in the previous project's
`mawecm_ann_jacobian_claude.py`.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

import numpy as np

import maw_lab as L

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT / "00_rve"), str(ROOT / "04_training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TOL = 1.0e-10
MAX_IT = 200
RES_PTS = 10
SIG_PTS = 10


def _field(fl, key, ne):
    return dict(state={k[len(key) + 5:]: fl[k] for k in fl.files
                       if k.startswith(f"{key}_net_")},
                mu=fl[f"{key}_mu"], sd=fl[f"{key}_sd"],
                act=str(fl[f"{key}_act"]), target_sum=float(ne))


class OfficialModels:
    """Builds the three models on one shared RVE, decoder and dof map."""

    def __init__(self, n_eval_states=None):
        import torch

        import linear_prom as lp
        import periodic_fom as pf
        pf.NEWTON_TOL = TOL
        lp.NEWTON_TOL = TOL
        lp.SUBSTEPS_PER_UNIT_STRAIN = pf.SUBSTEPS_PER_UNIT_STRAIN
        self.SUB = float(pf.SUBSTEPS_PER_UNIT_STRAIN)
        # Asserted, not assumed: an earlier comparison in this project was
        # invalidated by a PROM running a coarser ramp than the FOM.
        assert lp.SUBSTEPS_PER_UNIT_STRAIN == pf.SUBSTEPS_PER_UNIT_STRAIN
        assert lp.NEWTON_TOL == pf.NEWTON_TOL == TOL

        from periodic_fom import PeriodicRVE
        from linear_prom import LinearPROM
        from reduced_mesh import ReducedAssembly
        from numpy_decoder import NumpyDecoder
        from train_nslave import build_net

        bas = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
        dat = np.load(ROOT / "03_data" / "data.npz")
        sup = np.load(HERE / "ecm_supports.npz")
        nsl = np.load(ROOT / "04_training" / "nslave.npz")
        lab = L.load()
        self.ne = lab["ne"]

        self.Phi_S, self.A_M = bas["Phi_S"], bas["A_M"]
        self.mu_m, self.mu_s = nsl["mu_mean"], nsl["mu_std"]

        fr = np.load(HERE / "maw_res_long10.npz")
        fs = np.load(HERE / "maw_phase2_sig.npz")
        self.z_res = fr[f"res_{RES_PTS}_z"]
        self.z_sig = fs[f"sig_{SIG_PTS}_z"]
        self.m_res = _field(fr, f"res_{RES_PTS}", self.ne)
        self.m_sig = _field(fs, f"sig_{SIG_PTS}", self.ne)

        self.net = build_net(3, self.Phi_S.shape[1], width=int(nsl["width"]),
                             depth=int(nsl["depth"]))
        self.net.load_state_dict({k: torch.from_numpy(nsl[k])
                                  for k in self.net.state_dict()})
        self.net.eval()
        self.torch = torch

        self.rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                               cell_area=float(dat["cell_area"]))
        self.prom = LinearPROM(self.rve, bas["Phi_ROM"])
        self.dec = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                                ROOT / "04_training" / "nslave.npz")
        full = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"
        rve = self.rve

        self.hl_r = ReducedAssembly(full, HERE / "of_hlr", sup["z_res"],
                                    sup["w_res"], rve, self.prom.TPhi)
        self.hl_s = ReducedAssembly(full, HERE / "of_hls", sup["z_sig"],
                                    sup["w_sig"], rve, self.prom.TPhi)
        self.mw_r = ReducedAssembly(full, HERE / "of_mwr", self.z_res,
                                    np.ones(self.z_res.size), rve,
                                    np.zeros((rve.n_dof, 1)))
        self.mw_s = ReducedAssembly(full, HERE / "of_mws", self.z_sig,
                                    np.ones(self.z_sig.size), rve,
                                    np.zeros((rve.n_dof, 1)))
        self.base_r = np.array(self.mw_r.asm.w_detJ, dtype=float, copy=True)
        self.base_s = np.array(self.mw_s.asm.w_detJ, dtype=float, copy=True)

        # T has AT MOST one nonzero per row; the pinned dofs have an all-zero
        # row, since their value comes entirely from g(E). argmax would map
        # those to independent dof 0 and inject a wrong contribution, so the
        # map is explicit with -1 marking "no independent dof".
        Tc = rve.T.tocsr()
        cnt = np.diff(Tc.indptr)
        Tr = np.full(rve.n_dof, -1, dtype=np.int64)
        nz = cnt > 0
        if int(cnt.max()) != 1 or int(cnt[nz].min()) != 1:
            raise RuntimeError("T is not one-nonzero-per-row")
        Tr[nz] = Tc.indices[Tc.indptr[:-1][nz]]
        self._Tr = Tr

        self.dc_r, self.mk_r = self._restrict(self.mw_r.sel)
        self.dc_s, self.mk_s = self._restrict(self.mw_s.sel)

    def _restrict(self, sel):
        rows = self._Tr[sel]
        return self.dec.restrict(np.maximum(rows, 0)), (rows >= 0).astype(float)

    # -- references ------------------------------------------------------
    def fom(self, E):
        self.rve.solve(E)
        return self.rve.homogenized_stress(E)

    def prom_stress(self, E):
        return self.prom.solve(E)[0]

    # -- the three official models ---------------------------------------
    def hprom(self, E):
        """Linear 39-mode basis, classic fixed-weight ECM, 208 elements."""
        n = max(1, int(np.ceil(self.SUB * np.linalg.norm(E))))
        q = np.zeros(self.prom.TPhi.shape[1])
        for kk in range(1, n + 1):
            gs = self.rve._g_at(E * (kk / n), self.hl_r.sel)
            for _ in range(MAX_IT):
                K, R = self.hl_r.asm.Assemble(self.hl_r.TPhi @ q + gs)
                dq = np.linalg.solve(self.hl_r.TPhi.T @ (K @ self.hl_r.TPhi),
                                     self.hl_r.TPhi.T @ R)
                q = q + dq
                if np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30) < TOL:
                    break
            else:
                raise RuntimeError("HPROM Newton failed")
        self.hl_s.asm.Assemble(self.hl_s.TPhi @ q
                               + self.rve._g_at(E, self.hl_s.sel))
        return self.rve.homogenized_stress(E, assembler=self.hl_s.asm)

    def maw_hprom_ann(self, E, return_iters=False):
        """3-latent manifold, adaptive weights on both rules, 10 + 10."""
        n = max(1, int(np.ceil(self.SUB * np.linalg.norm(E))))
        q = np.zeros(3)
        nit = 0
        for kk in range(1, n + 1):
            gs = self.rve._g_at(E * (kk / n), self.mw_r.sel)
            for _ in range(MAX_IT):
                self.mw_r.asm.w_detJ = self.base_r * L.field_weights(
                    self.m_res, q[None, :])[:, 0][:, None]
                dd, PD = self.dc_r.value_and_jac(q)
                K, R = self.mw_r.asm.Assemble(dd * self.mk_r + gs)
                PD = PD * self.mk_r[:, None]
                dq = np.linalg.solve(PD.T @ (K @ PD), PD.T @ R)
                q = q + dq
                nit += 1
                if np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30) < TOL:
                    break
            else:
                raise RuntimeError("MAW-HPROM-ANN Newton failed")
        S = self._maw_stress(E, q)
        return (S, nit) if return_iters else S

    def maw_d_hprom_ann(self, E):
        """No solve: E is the latent coordinate, the closure gives the slave
        amplitudes, and the only cost left is the homogenized stress on the
        10-element rule. PhiMA already contains A_M, so its input is E."""
        with self.torch.no_grad():
            qs = self.net(self.torch.from_numpy(
                ((E - self.mu_m) / self.mu_s)[None, :])).numpy()[0]
        return self._maw_stress(E, E, qs=qs)

    def _maw_stress(self, E, q, qs=None):
        self.mw_s.asm.w_detJ = self.base_s * L.field_weights(
            self.m_sig, np.atleast_2d(q))[:, 0][:, None]
        if qs is None:
            dd, _ = self.dc_s.value_and_jac(q)
        else:
            dd = self.dc_s.PhiMA @ q + self.dc_s.Phi_S @ qs
        self.mw_s.asm.Assemble(dd * self.mk_s
                               + self.rve._g_at(E, self.mw_s.sel))
        return self.rve.homogenized_stress(E, assembler=self.mw_s.asm)

    def elements(self):
        return dict(FOM=self.ne, PROM=self.ne,
                    HPROM=int(np.asarray(self.hl_r.n_elements)
                              + np.asarray(self.hl_s.n_elements)),
                    MAW_HPROM_ANN=int(self.z_res.size + self.z_sig.size),
                    MAW_D_HPROM_ANN=int(self.z_sig.size))
