"""Pure linear-HPROM constitutive law for the coupon FE2 driver.

This module deliberately contains neither an ANN nor a manifold decoder.  The
online RVE equilibrium is the 39-mode linear POD problem assembled on the
fixed 135-element residual ECM mesh, while the homogenized second-Piola stress
is assembled on its distinct fixed 73-element ECM mesh.  Both are actual
reduced ``.mdpa`` model parts built from the selected elements; the full RVE
exists here only to provide the already-validated periodic lifting and POD map.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VALIDATION = ROOT / "05_validation"

NEWTON_TOL = 1.0e-10
NEWTON_MAX_IT = 40
SUBSTEPS_PER_UNIT_STRAIN = 200.0


class LinearHPROMECM:
    """39-mode POD + two fixed ECM rules on genuine reduced MDPA meshes."""

    def __init__(self, work_dir: str | Path | None = None):
        # Imports are intentionally delayed: worker processes must be forked
        # before Kratos is imported by either the parent or a child.
        from periodic_fom import PeriodicRVE
        from reduced_mesh import ReducedAssembly

        data = np.load(ROOT / "03_data" / "data.npz")
        basis = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
        supports = np.load(VALIDATION / "ecm_supports.npz")
        self.rve = PeriodicRVE(
            str(ROOT / "03_data" / "rve_mesh"), cell_area=float(data["cell_area"])
        )
        self.TPhi = np.ascontiguousarray(self.rve.T @ basis["Phi_ROM"])
        self.n_modes = int(self.TPhi.shape[1])

        if work_dir is None:
            work_dir = Path("/tmp") / f"coupon_hprom_ecm_{os.getpid()}"
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        full_mdpa = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"
        self.residual = ReducedAssembly(
            full_mdpa, self.work_dir / "residual_ecm", supports["z_res"],
            supports["w_res"], self.rve, self.TPhi,
        )
        self.stress = ReducedAssembly(
            full_mdpa, self.work_dir / "stress_ecm", supports["z_sig"],
            supports["w_sig"], self.rve, self.TPhi,
        )
        if self.residual.n_elements != 135 or self.stress.n_elements != 73:
            raise RuntimeError(
                "unexpected ECM support sizes: "
                f"{self.residual.n_elements} residual, {self.stress.n_elements} stress"
            )

    @property
    def ecm_metadata(self) -> dict:
        return {
            "n_modes": self.n_modes,
            "residual_elements": int(self.residual.n_elements),
            "stress_elements": int(self.stress.n_elements),
            "residual_mdpa": str(self.work_dir / "residual_ecm.mdpa"),
            "stress_mdpa": str(self.work_dir / "stress_ecm.mdpa"),
        }

    def _solve_at(self, E: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Converge the 39-dimensional reduced equilibrium at one strain."""
        g_res = self.rve._g_at(E, self.residual.sel)
        for _ in range(NEWTON_MAX_IT):
            K, R = self.residual.asm.Assemble(self.residual.TPhi @ q + g_res)
            Kq = self.residual.TPhi.T @ (K @ self.residual.TPhi)
            Rq = self.residual.TPhi.T @ R
            try:
                dq = np.linalg.solve(Kq, Rq)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("singular linear-HPROM reduced tangent") from exc
            q += dq
            if np.linalg.norm(dq) / max(np.linalg.norm(q), 1.0e-30) < NEWTON_TOL:
                # Reassemble at the returned q: ``K`` must describe the state
                # used later by the implicit-function sensitivity.
                K, R = self.residual.asm.Assemble(self.residual.TPhi @ q + g_res)
                return K, R
        raise RuntimeError(
            "linear-HPROM Newton failed on the 135-element ECM mesh, "
            f"E={np.array2string(E, precision=5)}"
        )

    def solve(self, E, q_init=None, E_start=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return final ``(q, K_res, R_res)`` with exact warm continuation."""
        E = np.asarray(E, dtype=float).reshape(3)
        E0 = np.zeros(3) if E_start is None else np.asarray(E_start, dtype=float).reshape(3)
        q = np.zeros(self.n_modes) if q_init is None else np.asarray(q_init, dtype=float).copy()
        n_sub = max(1, int(np.ceil(SUBSTEPS_PER_UNIT_STRAIN * np.linalg.norm(E - E0))))
        K = R = None
        for k in range(1, n_sub + 1):
            Et = E0 + (E - E0) * (k / n_sub)
            K, R = self._solve_at(Et, q)
        assert K is not None and R is not None
        return q, K, R

    def _stress_from_state(self, E: np.ndarray, q: np.ndarray) -> np.ndarray:
        """PK2 stress from the dedicated 73-element ECM stress mesh only."""
        g_sig = self.rve._g_at(E, self.stress.sel)
        self.stress.asm.Assemble(self.stress.TPhi @ q + g_sig)
        return self.rve.homogenized_stress(E, assembler=self.stress.asm)

    def _consistent_tangent(self, E: np.ndarray, q: np.ndarray, K_res: np.ndarray,
                            heps: float = 1.0e-6) -> np.ndarray:
        """Differentiate the HPROM equilibrium, then the stress mesh response.

        The sensitivity of q follows the implicit function theorem on the
        residual ECM mesh.  The final stress derivative is centered only over
        the 73-element stress assembler; it does *not* launch six additional
        nonlinear micro-solves or use the full RVE assembly.
        """
        A = self.residual.TPhi.T @ (K_res @ self.residual.TPhi)
        g_E_res = self.rve._periodic_lifting_jacobian(E)[self.residual.sel]
        rhs = -(self.residual.TPhi.T @ (K_res @ g_E_res))
        try:
            dq_dE = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("singular HPROM implicit-function tangent") from exc

        C = np.empty((3, 3))
        for j in range(3):
            h = heps * max(1.0, abs(E[j]))
            Ep, Em = E.copy(), E.copy()
            Ep[j] += h
            Em[j] -= h
            Sp = self._stress_from_state(Ep, q + h * dq_dE[:, j])
            Sm = self._stress_from_state(Em, q - h * dq_dE[:, j])
            C[:, j] = (Sp - Sm) / (2.0 * h)
        return C

    def stress_and_tangent(self, E, q_init=None, E_start=None, return_state=False):
        """Return ``(S, dS/dE)`` (and q on request), all from ECM MDPA meshes."""
        E = np.asarray(E, dtype=float).reshape(3)
        q, K_res, _R_res = self.solve(E, q_init=q_init, E_start=E_start)
        S = self._stress_from_state(E, q)
        C = self._consistent_tangent(E, q, K_res)
        return (S, C, q) if return_state else (S, C)
