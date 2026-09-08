"""Direct-stress regression baseline for the coupon macro FE solve.

This tier intentionally has no energy potential.  Its consistent Newton
tangent is therefore the Jacobian of the predicted stress, not an energy
Hessian; it is not expected to be symmetric.
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np

from flexible_pann_law import sha256


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ANISOTROPIC_PANN = ROOT.parent / "RVE_NeoHookean_Homogenization" / "pann" / "anisotropic"
if str(ANISOTROPIC_PANN) not in sys.path:
    sys.path.insert(0, str(ANISOTROPIC_PANN))


class CouponRegressionLaw:
    """Batched physical stress and its direct-Jacobian tangent."""

    def __init__(self, checkpoint: str | Path):
        import torch
        from anisotropic_pann_model_regression_claude import AnisotropicRegressionStress

        self.checkpoint_path = Path(checkpoint).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if self.checkpoint.get("kind") != "regression":
            raise ValueError(f"Expected the coupon regression PANN, got {self.checkpoint.get('kind')!r}.")
        state = self.checkpoint["state_dict"]
        if "net.0.weight" not in state or "strain_scale" not in state:
            raise ValueError("Regression checkpoint does not contain its expected network state.")
        self.model = AnisotropicRegressionStress(
            strain_scale=float(self.checkpoint["strain_scale"]), widths=(128, 128, 64)
        ).double()
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self._torch = torch
        self.strain_scale = float(self.checkpoint["strain_scale"])
        self.energy_scale = float(self.checkpoint["energy_scale"])
        self.calls = 0
        self.points = 0
        self.seconds = 0.0

    @property
    def metadata(self) -> dict:
        return dict(
            checkpoint=str(self.checkpoint_path),
            checkpoint_sha256=sha256(self.checkpoint_path),
            core="direct_stress_regression_mlp",
            features=3,
            feature_names=["E11", "E22", "gamma12"],
            widths=[128, 128, 64],
            strain_scale=self.strain_scale,
            energy_scale=self.energy_scale,
            checkpoint_test_stress_relative_l2=float(self.checkpoint["err_test"]),
            checkpoint_probe_stress_relative_l2=float(self.checkpoint["err_probe"]),
            structural_guarantee=(
                "direct stress regression only; no energy potential, exact reference "
                "normalization, tangent symmetry, or polyconvexity certificate"
            ),
        )

    def response(self, E, *, tangent: bool):
        """Return physical PK2 stress and optionally its physical Jacobian."""
        E = np.asarray(E, dtype=np.float64).reshape(-1, 3)
        torch = self._torch
        with torch.enable_grad():
            physical_E = torch.as_tensor(E, dtype=self.model.strain_scale.dtype)
            x = (physical_E / self.strain_scale).detach().requires_grad_(True)
            stress = self.model.stress(x)
            ans = dict(stress=stress.detach().cpu().numpy() * self.energy_scale / self.strain_scale)
            if tangent:
                derivative = torch.stack(
                    [torch.autograd.grad(stress[:, j].sum(), x, retain_graph=True)[0]
                     for j in range(3)], dim=1
                )
                ans["tangent"] = (
                    derivative.detach().cpu().numpy() * self.energy_scale / self.strain_scale**2
                )
        for name, value in ans.items():
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"non-finite regression-PANN {name}")
        return ans

    def __call__(self, E_flat, young=None, poisson=None):
        del young, poisson
        E = np.asarray(E_flat, dtype=np.float64).reshape(-1, 3)
        tic = time.perf_counter()
        ans = self.response(E, tangent=True)
        stress, tangent = ans["stress"], ans["tangent"]
        if stress.shape != E.shape or tangent.shape != (E.shape[0], 3, 3):
            raise RuntimeError("regression-PANN adapter returned an invalid batch shape")
        self.calls += 1
        self.points += int(E.shape[0])
        self.seconds += time.perf_counter() - tic
        return stress, tangent


def finite_difference_tangent_check(law: CouponRegressionLaw, states, h: float = 1.0e-6):
    """Check the stress Jacobian; report rather than require its asymmetry."""
    states = np.asarray(states, dtype=np.float64).reshape(-1, 3)
    reference = law.response(states, tangent=True)
    worst = 0.0
    for component in range(3):
        shift = np.zeros_like(states)
        shift[:, component] = h * np.maximum(1.0, np.abs(states[:, component]))
        plus = law.response(states + shift, tangent=False)["stress"]
        minus = law.response(states - shift, tangent=False)["stress"]
        fd = (plus - minus) / (2.0 * shift[:, component, None])
        error = np.linalg.norm(fd - reference["tangent"][:, :, component])
        worst = max(worst, float(error / max(np.linalg.norm(fd), 1.0)))
    asymmetry = np.linalg.norm(reference["tangent"] - np.swapaxes(reference["tangent"], 1, 2))
    asymmetry /= max(np.linalg.norm(reference["tangent"]), 1.0)
    return dict(tangent_fd_relative=worst, tangent_asymmetry_relative=float(asymmetry))
