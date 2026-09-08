"""MAW-D-HPROM-ANN constitutive law on the validated 10-element stress MDPA.

The direct tier has the same nonlinear-manifold displacement representation as
MAW-HPROM-ANN, but it does *not* solve a reduced RVE equilibrium problem.  Its
closure evaluates the 36 slave amplitudes directly from the macro Green--
Lagrange strain, while the three master amplitudes are the strain itself:

    d(E) = Phi_M A_M E + Phi_S N(E).

Stress is then evaluated with the validated MAW stress rule on exactly ten
physical RVE elements.  ``PhiMA`` below is ``Phi_M @ A_M``; applying ``A_M`` a
second time would be a different (and wrong) model.

This tier is a direct surrogate, not a potential.  We therefore differentiate
the complete direct stress map numerically for the macro Newton tangent rather
than borrowing an equilibrium/envelope tangent that the model does not have.
Those derivatives require no micro Newton solves and no residual MDPA.
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


FD_EPS = 1.0e-6


def _field(data: np.lib.npyio.NpzFile, key: str, n_full_elements: int) -> dict:
    """Deserialize one MAW softmax field without loading PyTorch."""
    return dict(
        state={name[len(key) + 5:]: data[name] for name in data.files
               if name.startswith(f"{key}_net_")},
        mu=np.asarray(data[f"{key}_mu"], dtype=float),
        sd=np.asarray(data[f"{key}_sd"], dtype=float),
        act=str(data[f"{key}_act"]),
        target_sum=float(n_full_elements),
    )


class MAWDHPROMANN:
    """Closure-based manifold model evaluated on only the 10 stress elements."""

    def __init__(self, work_dir: str | Path | None = None):
        # Local imports ensure macro worker processes, rather than their parent,
        # own every Kratos model part.
        from periodic_fom import PeriodicRVE
        from reduced_mesh import ReducedAssembly
        from numpy_decoder import NumpyDecoder

        data = np.load(ROOT / "03_data" / "data.npz")
        stress_field = np.load(VALIDATION / "maw_phase2_sig.npz")
        self.rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                               cell_area=float(data["cell_area"]))
        self.n_full_elements = int(data["n_elements"])
        self.z_sig = np.asarray(stress_field["sig_10_z"], dtype=np.int64)
        self.field_sig = _field(stress_field, "sig_10", self.n_full_elements)
        self.decoder = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                                    ROOT / "04_training" / "nslave.npz")

        if work_dir is None:
            work_dir = Path("/tmp") / f"coupon_maw_dhprom_ann_{os.getpid()}"
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        full_mdpa = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"
        # The direct closure constructs its displacements itself.  ReducedAssembly
        # is used here solely to build the real optimized 10-element MDPA and
        # preserve the exact full-DOF ordering selected by it.
        dummy = np.zeros((self.rve.n_dof, 1))
        self.stress_assembly = ReducedAssembly(full_mdpa, self.work_dir / "stress_maw10",
                                               self.z_sig, np.ones(self.z_sig.size),
                                               self.rve, dummy)
        if self.stress_assembly.n_elements != 10:
            raise RuntimeError("expected the validated 10-element MAW stress support")
        self.base_sig = np.array(self.stress_assembly.asm.w_detJ, dtype=float, copy=True)
        self.decoder_sig, self.mask_sig = self._restricted_decoder(self.stress_assembly.sel)
        self.weight_min = np.inf
        self.weight_sum_error = 0.0

    def _restricted_decoder(self, selected_full_dofs: np.ndarray):
        """Restrict decoder rows; leave exactly pinned dofs to the lifting."""
        Tcsr = self.rve.T.tocsr()
        counts = np.diff(Tcsr.indptr)
        if int(counts.max()) != 1:
            raise RuntimeError("periodic lifting no longer has one nonzero per row")
        independent_row = np.full(self.rve.n_dof, -1, dtype=np.int64)
        nonzero = counts > 0
        independent_row[nonzero] = Tcsr.indices[Tcsr.indptr[:-1][nonzero]]
        rows = independent_row[np.asarray(selected_full_dofs, dtype=np.int64)]
        return self.decoder.restrict(np.maximum(rows, 0)), (rows >= 0).astype(float)

    def _weights(self, E: np.ndarray) -> np.ndarray:
        from maw_lab import field_weights

        weights = np.asarray(field_weights(self.field_sig, E[None, :])[:, 0], dtype=float)
        if np.any(weights < -1.0e-12):
            raise RuntimeError("MAW field violated non-negative cubature weights")
        self.weight_min = min(self.weight_min, float(weights.min()))
        self.weight_sum_error = max(self.weight_sum_error,
                                    abs(float(weights.sum()) - self.n_full_elements))
        return weights

    def evaluate_stress(self, E: np.ndarray) -> np.ndarray:
        """Direct stress evaluation; no reduced or full RVE nonlinear solve."""
        E = np.asarray(E, dtype=float).reshape(3)
        self.stress_assembly.asm.w_detJ = self.base_sig * self._weights(E)[:, None]
        # The closure net maps E -> slave coordinates.  ``PhiMA`` already
        # includes A_M, so this is exactly OfficialModels.maw_d_hprom_ann.
        slave, _jac = self.decoder_sig.net_and_jac(E[None, :])
        displacement = self.decoder_sig.PhiMA @ E + self.decoder_sig.Phi_S @ slave[0]
        u = displacement * self.mask_sig + self.rve._g_at(E, self.stress_assembly.sel)
        # Homogenized stress requires F and S at Gauss points, not a force or
        # stiffness assembly.  Avoiding the latter is material to this direct
        # online tier.
        self.stress_assembly.asm.ComputeStrainStressOnly(u)
        result = self.rve.homogenized_stress(E, assembler=self.stress_assembly.asm)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("non-finite MAW-D-HPROM-ANN stress")
        return result

    @staticmethod
    def _step(value: float) -> float:
        return FD_EPS * max(1.0, abs(float(value)))

    def stress_and_tangent(self, E: np.ndarray, q_init=None, E_start=None,
                           return_state=False):
        """Direct stress and its central-difference macro tangent.

        The seven stress evaluations (center plus +/- for three components)
        are all 10-element, solve-free evaluations.  This intentionally gives
        macro Newton the derivative of the direct *model actually used*, not
        the tangent of an unrelated equilibrium model.
        """
        # q_init/E_start are deliberately accepted so the common FE2 adapter
        # can serve iterative and direct manifold tiers.  A direct closure has
        # no internal state or continuation path to solve.
        del q_init, E_start
        E = np.asarray(E, dtype=float).reshape(3)
        result = self.evaluate_stress(E)
        tangent = np.empty((3, 3), dtype=float)
        for component in range(3):
            h = self._step(E[component])
            Ep, Em = E.copy(), E.copy()
            Ep[component] += h
            Em[component] -= h
            tangent[:, component] = (self.evaluate_stress(Ep) - self.evaluate_stress(Em)) / (2.0 * h)
        if not np.all(np.isfinite(tangent)):
            raise RuntimeError("non-finite MAW-D-HPROM-ANN tangent")
        return (result, tangent, E.copy()) if return_state else (result, tangent)

    @property
    def ecm_metadata(self) -> dict:
        return dict(
            n_latent=3,
            closure="E_to_36_slave_amplitudes",
            stress_ecm_elements=int(self.stress_assembly.n_elements),
            total_distinct_ecm_elements=int(self.stress_assembly.n_elements),
            stress_mdpa=str(self.work_dir / "stress_maw10.mdpa"),
            stress_support_full_indices=self.z_sig.tolist(),
            minimum_adaptive_weight=float(self.weight_min),
            maximum_weight_sum_error=float(self.weight_sum_error),
            macro_tangent="central_difference_of_direct_10_element_stress",
            finite_difference_epsilon=FD_EPS,
        )
