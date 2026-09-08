"""Batched stress integration on an actual reduced MDPA assembler.

Shares the existing microscopic Neo-Hookean law and periodic constraints.
Only kinematics and averaging are batched; no full-mesh element evaluations.
"""
import numpy as np


class ReducedStressBatch:
    def __init__(self, rve, mesh):
        self.rve, self.mesh, self.a = rve, mesh, mesh.asm
        self.base_weights = self.a.w_detJ.copy()
        root = rve.final_root[mesh.sel]
        self.arm = rve.final_off[mesh.sel].copy()
        pinned = rve.pinned[root]
        self.arm[pinned] += rve.dof_xy[root[pinned]]
        self.comp = rve.dof_comp[mesh.sel]

    @staticmethod
    def macro_F(E):
        C = np.empty((len(E), 2, 2))
        C[:, 0, 0], C[:, 1, 1] = 1+2*E[:, 0], 1+2*E[:, 1]
        C[:, 0, 1] = C[:, 1, 0] = E[:, 2]
        val, vec = np.linalg.eigh(C)
        if np.any(val <= 0):
            raise RuntimeError('invalid macro Green-Lagrange strain')
        return (vec * np.sqrt(val)[:, None, :]) @ np.swapaxes(vec, 1, 2)

    def lifting_jacobian(self, E):
        """Same analytic lifting derivative, restricted before computing it."""
        from deformation_gradient_jacobian_claude import deformation_gradient_and_jacobian_2d
        _, dF = deformation_gradient_and_jacobian_2d(np.asarray(E))
        values = np.einsum('ijk,dj->dik', dF, self.arm)
        return values[np.arange(len(self.comp)), self.comp]

    def evaluate(self, E, displacement, element_weights=None):
        E, d = np.atleast_2d(E), np.atleast_2d(displacement)
        a = self.a
        n, ne, ng = len(E), a.n_elems, a.n_gauss
        Fbar = self.macro_F(E)
        lift = np.einsum('bij,dj->bdi', Fbar-np.eye(2), self.arm)
        u = d + lift[:, np.arange(len(self.comp)), self.comp]
        un = u[:, a.local_eq_ids].reshape(n, ne, a.n_nodes, 2)
        F = np.einsum('beai,egaj->begij', un, a.DN) + np.eye(2)
        if np.any(np.linalg.det(F) <= 0):
            raise RuntimeError('inverted element in reduced stress batch')
        C = np.swapaxes(F, -1, -2) @ F
        Ev = np.stack((.5*(C[..., 0, 0]-1), .5*(C[..., 1, 1]-1), C[..., 0, 1]), axis=-1)
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
        w = self.base_weights[None, :, :]
        if element_weights is not None:
            w = w * np.asarray(element_weights)[:, :, None]
        Pbar = np.sum(w[..., None, None]*(F @ St), axis=(1, 2)) / self.rve.denom
        S = np.linalg.solve(Fbar, Pbar)
        return np.stack((S[:, 0, 0], S[:, 1, 1], .5*(S[:, 0, 1]+S[:, 1, 0])), axis=1)
