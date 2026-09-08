"""Local projection and batched IFT stress stencil for the fixed-ECM HPROM.

Identical 39-mode equilibrium and 135/73 physical MDPA supports.  The original
LinearHPROMECM remains the reference; no decoder or adaptive weights are used.
"""
import numpy as np
from linear_hprom_ecm import LinearHPROMECM, NEWTON_MAX_IT, NEWTON_TOL
from reduced_stress_batch import ReducedStressBatch


class FastLinearHPROMECM(LinearHPROMECM):
    def __init__(self, work_dir=None):
        super().__init__(work_dir)
        self.Te = np.ascontiguousarray(self.residual.TPhi[self.residual.asm.local_eq_ids])
        self.Tflat = self.Te.reshape(-1, self.n_modes)
        self.batch_res = ReducedStressBatch(self.rve, self.residual)
        self.batch_sig = ReducedStressBatch(self.rve, self.stress)

    def _project(self, Ke):
        return self.Tflat.T @ (Ke @ self.Te).reshape(-1, self.n_modes)

    def _solve_at(self, E, q):
        """Return local element K/force arrays instead of sparse global arrays."""
        a = self.residual.asm
        g = self.rve._g_at(E, self.residual.sel)
        for _ in range(NEWTON_MAX_IT):
            Ke, fint = a.ComputeLocalArrays(self.residual.TPhi @ q + g)
            Kq = self._project(Ke)
            rq = -(self.Tflat.T @ fint.reshape(-1))
            q += (dq := np.linalg.solve(Kq, rq))
            if np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30) < NEWTON_TOL:
                # Match the original reassembly at the returned state.
                return a.ComputeLocalArrays(self.residual.TPhi @ q + g)
        raise RuntimeError(f'local linear HPROM Newton failed, E={E}')

    def solve(self, E, q_init=None, E_start=None):
        """Return (q, element stiffness, element internal force) after convergence."""
        return super().solve(E, q_init=q_init, E_start=E_start)

    def _stress_from_state(self, E, q):
        return self.batch_sig.evaluate(E, self.stress.TPhi @ q)[0]

    def _consistent_tangent(self, E, q, K_res, heps=1e-6):
        Ge = self.batch_res.lifting_jacobian(E)[self.residual.asm.local_eq_ids]
        rhs = -(self.Tflat.T @ (K_res @ Ge).reshape(-1, 3))
        dq = np.linalg.solve(self._project(K_res), rhs)
        h = heps * np.maximum(1., np.abs(E))
        EE, QQ = np.tile(E, (6, 1)), np.tile(q, (6, 1))
        for j in range(3):
            EE[2*j, j] += h[j]
            EE[2*j+1, j] -= h[j]
            QQ[2*j] += h[j]*dq[:, j]
            QQ[2*j+1] -= h[j]*dq[:, j]
        S = self.batch_sig.evaluate(EE, QQ @ self.stress.TPhi.T)
        return ((S[::2]-S[1::2])/(2*h[:, None])).T

    @property
    def ecm_metadata(self):
        return dict(super().ecm_metadata,
                    implementation='local_projection_batched_stress_ift',
                    micro_newton_tolerance=NEWTON_TOL)
