"""MAW-HPROM-ANN constitutive law on the 10 + 10 element reduced MDPA meshes.

This is the iterative, three-coordinate nonlinear-manifold model.  The
residual and homogenized-stress rules are the validated adaptive MAW-ECM
rules, not the 135 + 73 element fixed rules used by the linear HPROM.  Both
weight fields retain non-negative weights whose sum is exactly the number of
full RVE elements at every query.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VALIDATION = ROOT / "05_validation"
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for candidate in (ROOT / "00_rve", ROOT / "04_training", VALIDATION,
                  PROJ / "fe2_extension", PROJ / "core"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
_kratos_candidates = (
    Path("/home/sares/Kratos_Eigen_Check/bin/Release"),
    Path("/home/kratos/Kratos_Eigen_Check/bin/Release"),
)
KRATOS_PATH = next((path for path in _kratos_candidates if path.is_dir()), _kratos_candidates[0])
if str(KRATOS_PATH) not in sys.path:
    sys.path.append(str(KRATOS_PATH))


NEWTON_TOL = 1.0e-10
NEWTON_MAX_IT = 200
SUBSTEPS_PER_UNIT_STRAIN = 200.0
FD_EPS = 1.0e-6


def _field(data: np.lib.npyio.NpzFile, key: str, n_full_elements: int) -> dict:
    """Deserialize a MAW softmax field without importing PyTorch online."""
    return dict(
        state={name[len(key) + 5:]: data[name] for name in data.files
               if name.startswith(f"{key}_net_")},
        mu=np.asarray(data[f"{key}_mu"], dtype=float),
        sd=np.asarray(data[f"{key}_sd"], dtype=float),
        act=str(data[f"{key}_act"]),
        target_sum=float(n_full_elements),
    )


class MAWHPROMANN:
    """Iterative manifold equilibrium with a local IFT macro tangent."""

    def __init__(self, work_dir: str | Path | None = None):
        # These imports must remain local.  The macro driver forks workers
        # before any process constructs a Kratos model part.
        from periodic_fom import PeriodicRVE
        from reduced_mesh import ReducedAssembly
        from numpy_decoder import NumpyDecoder

        data = np.load(ROOT / "03_data" / "data.npz")
        residual_field = np.load(VALIDATION / "maw_res_long10.npz")
        stress_field = np.load(VALIDATION / "maw_phase2_sig.npz")
        self.rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                               cell_area=float(data["cell_area"]))
        self.n_full_elements = int(data["n_elements"])
        self.z_res = np.asarray(residual_field["res_10_z"], dtype=np.int64)
        self.z_sig = np.asarray(stress_field["sig_10_z"], dtype=np.int64)
        self.field_res = _field(residual_field, "res_10", self.n_full_elements)
        self.field_sig = _field(stress_field, "sig_10", self.n_full_elements)
        self.decoder = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                                    ROOT / "04_training" / "nslave.npz")

        if work_dir is None:
            work_dir = Path("/tmp") / f"coupon_maw_hprom_ann_{os.getpid()}"
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        full_mdpa = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"
        # The decoder tangent varies with q, hence ReducedAssembly's stored
        # fixed basis is deliberately a dummy.  Its exact reduced-to-full DOF
        # selection is what connects the actual reduced MDPA meshes to the
        # manifold decoder.
        dummy = np.zeros((self.rve.n_dof, 1))
        self.residual = ReducedAssembly(full_mdpa, self.work_dir / "residual_maw10",
                                        self.z_res, np.ones(self.z_res.size),
                                        self.rve, dummy)
        self.stress = ReducedAssembly(full_mdpa, self.work_dir / "stress_maw10",
                                      self.z_sig, np.ones(self.z_sig.size),
                                      self.rve, dummy)
        if self.residual.n_elements != 10 or self.stress.n_elements != 10:
            raise RuntimeError("expected the validated 10 + 10 MAW-ECM support")
        self.base_res = np.array(self.residual.asm.w_detJ, dtype=float, copy=True)
        self.base_sig = np.array(self.stress.asm.w_detJ, dtype=float, copy=True)
        self.decoder_res, self.mask_res = self._restricted_decoder(self.residual.sel)
        self.decoder_sig, self.mask_sig = self._restricted_decoder(self.stress.sel)
        self.n_modes = 3
        self.weight_min = np.inf
        self.weight_sum_error = 0.0
        self.q_min = np.full(3, np.inf)
        self.q_max = np.full(3, -np.inf)

    def _restricted_decoder(self, selected_full_dofs: np.ndarray):
        """Restrict decoder rows, preserving zero values on pinned DOFs."""
        Tcsr = self.rve.T.tocsr()
        counts = np.diff(Tcsr.indptr)
        if int(counts.max()) != 1:
            raise RuntimeError("periodic lifting no longer has one nonzero per row")
        independent_row = np.full(self.rve.n_dof, -1, dtype=np.int64)
        nonzero = counts > 0
        independent_row[nonzero] = Tcsr.indices[Tcsr.indptr[:-1][nonzero]]
        rows = independent_row[np.asarray(selected_full_dofs, dtype=np.int64)]
        return self.decoder.restrict(np.maximum(rows, 0)), (rows >= 0).astype(float)

    def _weights(self, field: dict, q: np.ndarray) -> np.ndarray:
        from maw_lab import field_weights

        weights = np.asarray(field_weights(field, np.asarray(q)[None, :])[:, 0], dtype=float)
        if np.any(weights < -1.0e-12):
            raise RuntimeError("MAW field violated non-negative cubature weights")
        self.weight_min = min(self.weight_min, float(weights.min()))
        self.weight_sum_error = max(self.weight_sum_error, abs(float(weights.sum()) - self.n_full_elements))
        return weights

    def _residual_state(self, E: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Actual three-component manifold residual r(q,E)."""
        weights = self._weights(self.field_res, q)
        self.residual.asm.w_detJ = self.base_res * weights[:, None]
        displacement, tangent = self.decoder_res.value_and_jac(q)
        tangent = tangent * self.mask_res[:, None]
        _K, internal = self.residual.asm.Assemble(
            displacement * self.mask_res + self.rve._g_at(E, self.residual.sel)
        )
        return tangent.T @ internal

    def _newton_state(self, E: np.ndarray, q: np.ndarray) -> np.ndarray:
        """One modified-Newton solve at one ramp point, with frozen weights."""
        for _ in range(NEWTON_MAX_IT):
            weights = self._weights(self.field_res, q)
            self.residual.asm.w_detJ = self.base_res * weights[:, None]
            displacement, tangent = self.decoder_res.value_and_jac(q)
            tangent = tangent * self.mask_res[:, None]
            K, internal = self.residual.asm.Assemble(
                displacement * self.mask_res + self.rve._g_at(E, self.residual.sel)
            )
            reduced_tangent = tangent.T @ (K @ tangent)
            residual = tangent.T @ internal
            try:
                increment = np.linalg.solve(reduced_tangent, residual)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("singular MAW-HPROM-ANN reduced tangent") from exc
            q += increment
            if np.linalg.norm(increment) / max(np.linalg.norm(q), 1.0e-30) < NEWTON_TOL:
                return q
        raise RuntimeError(
            "MAW-HPROM-ANN Newton failed on its 10-element residual MDPA, "
            f"E={np.array2string(E, precision=5)}"
        )

    def solve(self, E, q_init=None, E_start=None) -> np.ndarray:
        E = np.asarray(E, dtype=float).reshape(3)
        E0 = np.zeros(3) if E_start is None else np.asarray(E_start, dtype=float).reshape(3)
        q = np.zeros(3) if q_init is None else np.asarray(q_init, dtype=float).copy()
        n_substeps = max(1, int(np.ceil(SUBSTEPS_PER_UNIT_STRAIN * np.linalg.norm(E - E0))))
        for step in range(1, n_substeps + 1):
            q = self._newton_state(E0 + (E - E0) * (step / n_substeps), q)
        self.q_min = np.minimum(self.q_min, q)
        self.q_max = np.maximum(self.q_max, q)
        return q

    def _stress_from_state(self, E: np.ndarray, q: np.ndarray) -> np.ndarray:
        weights = self._weights(self.field_sig, q)
        self.stress.asm.w_detJ = self.base_sig * weights[:, None]
        displacement, _tangent = self.decoder_sig.value_and_jac(q)
        self.stress.asm.Assemble(
            displacement * self.mask_sig + self.rve._g_at(E, self.stress.sel)
        )
        return self.rve.homogenized_stress(E, assembler=self.stress.asm)

    @staticmethod
    def _step(value: float) -> float:
        return FD_EPS * max(1.0, abs(float(value)))

    def _consistent_tangent(self, E: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Numerical IFT tangent of the *actual* adaptive reduced equations.

        The micro Newton uses a Gauss--Newton approximation, but the macro
        tangent differentiates r(q,E)=Phi_D(q)^T R(q,E,w(q)) itself.  Thus the
        decoder curvature and weight-field derivative are both included, while
        avoiding any extra nonlinear micro-equilibrium solves.
        """
        r_q = np.empty((3, 3))
        r_E = np.empty((3, 3))
        s_q = np.empty((3, 3))
        s_E = np.empty((3, 3))
        for component in range(3):
            hq = self._step(q[component])
            qp, qm = q.copy(), q.copy()
            qp[component] += hq
            qm[component] -= hq
            r_q[:, component] = (self._residual_state(E, qp) - self._residual_state(E, qm)) / (2.0 * hq)
            s_q[:, component] = (self._stress_from_state(E, qp) - self._stress_from_state(E, qm)) / (2.0 * hq)

            hE = self._step(E[component])
            Ep, Em = E.copy(), E.copy()
            Ep[component] += hE
            Em[component] -= hE
            r_E[:, component] = (self._residual_state(Ep, q) - self._residual_state(Em, q)) / (2.0 * hE)
            s_E[:, component] = (self._stress_from_state(Ep, q) - self._stress_from_state(Em, q)) / (2.0 * hE)
        try:
            dq_dE = np.linalg.solve(r_q, -r_E)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("singular MAW-HPROM-ANN implicit tangent") from exc
        return s_E + s_q @ dq_dE

    def stress_and_tangent(self, E, q_init=None, E_start=None, return_state=False):
        E = np.asarray(E, dtype=float).reshape(3)
        q = self.solve(E, q_init=q_init, E_start=E_start)
        stress = self._stress_from_state(E, q)
        tangent = self._consistent_tangent(E, q)
        if not np.all(np.isfinite(stress)) or not np.all(np.isfinite(tangent)):
            raise RuntimeError("non-finite MAW-HPROM-ANN response")
        return (stress, tangent, q) if return_state else (stress, tangent)

    @property
    def ecm_metadata(self) -> dict:
        return dict(
            n_latent=3,
            residual_elements=int(self.residual.n_elements),
            stress_elements=int(self.stress.n_elements),
            total_distinct_ecm_elements=int(np.union1d(self.z_res, self.z_sig).size),
            residual_mdpa=str(self.work_dir / "residual_maw10.mdpa"),
            stress_mdpa=str(self.work_dir / "stress_maw10.mdpa"),
            residual_support_full_indices=self.z_res.tolist(),
            stress_support_full_indices=self.z_sig.tolist(),
            minimum_adaptive_weight=float(self.weight_min),
            maximum_weight_sum_error=float(self.weight_sum_error),
            q_minimum=self.q_min.tolist(),
            q_maximum=self.q_max.tolist(),
        )
