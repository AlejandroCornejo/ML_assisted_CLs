"""Equivalent local/batched implementation of the iterative MAW-HPROM-ANN.

Keeps the original modified Newton, strain substeps, decoder, MAW fields,
10+10 physical MDPA supports and central-difference IFT tangent.  Optimization
only changes assembly: project element arrays directly into the three modes,
and evaluate the IFT stencil in batches without constructing unused stiffness.
The original MAWHPROMANN remains the independent regression reference.
"""
from __future__ import annotations

import numpy as np

from maw_hprom_ann_law import MAWHPROMANN, NEWTON_MAX_IT, NEWTON_TOL


class FastMAWHPROMANN(MAWHPROMANN):
    def __init__(self, work_dir=None):
        super().__init__(work_dir)
        self._arms = {}
        for mesh in (self.residual, self.stress):
            sel = mesh.sel
            root = self.rve.final_root[sel]
            arm = self.rve.final_off[sel].copy()
            pin = self.rve.pinned[root]
            arm[pin] += self.rve.dof_xy[root[pin]]
            self._arms[id(mesh)] = arm

    def _newton_state(self, E, q):
        """Identical modified Newton using local 12x12 element matrices."""
        a = self.residual.asm
        lifting = self.rve._g_at(E, self.residual.sel)
        for _ in range(NEWTON_MAX_IT):
            a.w_detJ = self.base_res * self._weights(self.field_res, q)[:, None]
            d, T = self.decoder_res.value_and_jac(q)
            T = T * self.mask_res[:, None]
            Ke, fint = a.ComputeLocalArrays(d * self.mask_res + lifting)
            Te = T[a.local_eq_ids]
            Kq = np.sum(np.swapaxes(Te, 1, 2) @ (Ke @ Te), axis=0)
            # The original assembler returns rhs=-f_int.
            rq = -np.einsum('eik,ei->k', Te, fint.reshape(a.n_elems, -1))
            dq = np.linalg.solve(Kq, rq)
            q += dq
            if np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30) < NEWTON_TOL:
                return q
        raise RuntimeError(f'MAW-HPROM-ANN local Newton failed, E={E}')

    @staticmethod
    def _macro_F(E):
        """Batched version of the reference symmetric square root of 2E+I."""
        C = np.empty((len(E), 2, 2))
        C[:, 0, 0], C[:, 1, 1] = 1 + 2 * E[:, 0], 1 + 2 * E[:, 1]
        C[:, 0, 1] = C[:, 1, 0] = E[:, 2]
        val, vec = np.linalg.eigh(C)
        if np.any(val <= 0):
            raise RuntimeError('invalid macro C in MAW tangent stencil')
        return (vec * np.sqrt(val)[:, None, :]) @ np.swapaxes(vec, 1, 2)

    def _batch_values(self, E, Q, *, residual):
        """Actual r(q,E) or S(q,E), vectorized over independent stencil states.

        Geometry, quadrature and equation maps are extracted from the actual
        reduced MDPA assembler.  No full-mesh element evaluation occurs here.
        """
        from maw_lab import field_weights

        E, Q = np.atleast_2d(E), np.atleast_2d(Q)
        if residual:
            mesh, dec, mask = self.residual, self.decoder_res, self.mask_res
            field, base = self.field_res, self.base_res
        else:
            mesh, dec, mask = self.stress, self.decoder_sig, self.mask_sig
            field, base = self.field_sig, self.base_sig
        a = mesh.asm
        n, ne, ng = len(E), a.n_elems, a.n_gauss
        weights = field_weights(field, Q).T
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise RuntimeError('invalid adaptive MAW weights')
        self.weight_min = min(self.weight_min, float(weights.min()))
        self.weight_sum_error = max(self.weight_sum_error,
            float(np.max(np.abs(weights.sum(axis=1) - self.n_full_elements))))
        w = base[None, :, :] * weights[:, :, None]
        N, J = dec.net_and_jac(Q)
        d = (Q @ dec.PhiMA.T + N @ dec.Phi_S.T) * mask
        Fbar = self._macro_F(E)
        lifting = np.einsum('bij,dj->bdi', Fbar - np.eye(2), self._arms[id(mesh)])
        d += lifting[:, np.arange(len(mesh.sel)), self.rve.dof_comp[mesh.sel]]
        un = d[:, a.local_eq_ids].reshape(n, ne, a.n_nodes, 2)
        F = np.einsum('beai,egaj->begij', un, a.DN) + np.eye(2)
        C = np.swapaxes(F, -1, -2) @ F
        Ev = np.stack((.5*(C[..., 0, 0]-1), .5*(C[..., 1, 1]-1),
                       C[..., 0, 1]), axis=-1)
        if np.any(np.linalg.det(F) <= 0):
            raise RuntimeError('inverted reduced element in MAW tangent stencil')
        # Reuse exactly the established microscopic law, including materials.
        if a._uniform_material:
            young, poisson = a._young_scalar, a._poisson_scalar
        else:
            young = np.broadcast_to(a.young[None, :, None], (n, ne, ng)).reshape(-1)
            poisson = np.broadcast_to(a.poisson[None, :, None], (n, ne, ng)).reshape(-1)
        Sv, _ = self.rve._fom._neo_hookean_pk2_2d_vectorized(Ev.reshape(-1, 3), young, poisson)
        Sv = Sv.reshape(n, ne, ng, 3)
        St = np.empty((n, ne, ng, 2, 2))
        St[..., 0, 0], St[..., 1, 1] = Sv[..., 0], Sv[..., 1]
        St[..., 0, 1] = St[..., 1, 0] = Sv[..., 2]
        P = F @ St
        if residual:
            T = (dec.PhiMA[None, :, :] + dec.Phi_S[None, :, :] @ J) * mask[None, :, None]
            fint = np.einsum('begij,egaj,beg->beai', P, a.DN, w).reshape(n, ne, -1)
            return -np.einsum('beik,bei->bk', T[:, a.local_eq_ids], fint)
        Pbar = np.einsum('beg,begij->bij', w, P) / self.rve.denom
        Sbar = np.linalg.solve(Fbar, Pbar)
        return np.stack((Sbar[:, 0, 0], Sbar[:, 1, 1],
                         .5*(Sbar[:, 0, 1]+Sbar[:, 1, 0])), axis=1)

    def _stress_from_state(self, E, q):
        return self._batch_values(E, q, residual=False)[0]

    def _residual_state(self, E, q):
        return self._batch_values(E, q, residual=True)[0]

    def _consistent_tangent(self, E, q):
        # Same +/- perturbations as the reference (first six q, then six E).
        EE, QQ = np.tile(E, (12, 1)), np.tile(q, (12, 1))
        hq = np.array([self._step(x) for x in q])
        he = np.array([self._step(x) for x in E])
        for k in range(3):
            QQ[2*k, k] += hq[k]
            QQ[2*k+1, k] -= hq[k]
            EE[6+2*k, k] += he[k]
            EE[7+2*k, k] -= he[k]
        R = self._batch_values(EE, QQ, residual=True)
        S = self._batch_values(EE, QQ, residual=False)
        rq = ((R[:6:2] - R[1:6:2]) / (2*hq[:, None])).T
        rE = ((R[6::2] - R[7::2]) / (2*he[:, None])).T
        sq = ((S[:6:2] - S[1:6:2]) / (2*hq[:, None])).T
        sE = ((S[6::2] - S[7::2]) / (2*he[:, None])).T
        return sE + sq @ np.linalg.solve(rq, -rE)

    @property
    def ecm_metadata(self):
        return dict(super().ecm_metadata,
                    implementation='local_projection_batched_ift',
                    micro_newton_tolerance=NEWTON_TOL,
                    macro_tangent='same_central_difference_ift_as_baseline')
