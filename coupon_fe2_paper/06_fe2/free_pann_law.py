"""Direct macro adapter for the coupon's unconstrained free-energy PANN.

The checkpoint predates the selected flexible ICNN/ICKAN models.  It receives
the four material-frame features of ``C`` and has an affine reference
correction, hence it is energy-consistent and exactly stress-free at ``E=0``.
It deliberately makes *no* polyconvexity or non-negative-energy claim.
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


class CouponFreePANNLaw:
    """Batched physical-units free-energy PANN for the macro assembler."""

    def __init__(self, checkpoint: str | Path):
        import torch
        from anisotropic_pann_model import AnisotropicFreeEnergy

        self.checkpoint_path = Path(checkpoint).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if self.checkpoint.get("kind") != "free":
            raise ValueError(f"Expected the coupon free PANN, got {self.checkpoint.get('kind')!r}.")
        state = self.checkpoint["state_dict"]
        required = {"strain_scale", "feature_scale", "base_energy.network.0.weight"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Free-PANN checkpoint misses {sorted(missing)}.")

        # This is the architecture used by ``06_pann/train_pann.py``.  Reading
        # the feature scale from the state dict preserves the exact scaler used
        # during training instead of reconstructing it from a later data file.
        self.model = AnisotropicFreeEnergy(
            strain_scale=float(self.checkpoint["strain_scale"]),
            feature_scale=state["feature_scale"], widths=(128, 128, 64),
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
            core="free_energy_mlp",
            features=4,
            feature_names=["a0.C.a0 - 1", "b0.C.b0 - 1", "a0.C.b0", "J - 1"],
            widths=[128, 128, 64],
            strain_scale=self.strain_scale,
            energy_scale=self.energy_scale,
            checkpoint_test_stress_relative_l2=float(self.checkpoint["err_test"]),
            checkpoint_probe_stress_relative_l2=float(self.checkpoint["err_probe"]),
            structural_guarantee=(
                "energy potential with an exact affine reference correction; "
                "no polyconvexity or non-negative-energy certificate"
            ),
        )

    def response(self, E, *, tangent: bool):
        """Return physical energy, PK2 stress, and (when requested) dS/dE."""
        E = np.asarray(E, dtype=np.float64).reshape(-1, 3)
        torch = self._torch
        with torch.enable_grad():
            physical_E = torch.as_tensor(E, dtype=self.model.strain_scale.dtype)
            x = (physical_E / self.strain_scale).detach().requires_grad_(True)
            energy, stress = self.model.energy_and_stress(x, create_graph=tangent)
            ans = dict(
                energy=energy[:, 0].detach().cpu().numpy() * self.energy_scale,
                stress=stress.detach().cpu().numpy() * self.energy_scale / self.strain_scale,
            )
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
                raise RuntimeError(f"non-finite free-PANN {name}")
        return ans

    def __call__(self, E_flat, young=None, poisson=None):
        del young, poisson
        E = np.asarray(E_flat, dtype=np.float64).reshape(-1, 3)
        tic = time.perf_counter()
        ans = self.response(E, tangent=True)
        stress, tangent = ans["stress"], ans["tangent"]
        if stress.shape != E.shape or tangent.shape != (E.shape[0], 3, 3):
            raise RuntimeError("free-PANN adapter returned an invalid batch shape")
        self.calls += 1
        self.points += int(E.shape[0])
        self.seconds += time.perf_counter() - tic
        return stress, tangent


def finite_difference_tangent_check(law: CouponFreePANNLaw, states, h: float = 1.0e-6):
    """Return the independent physical finite-difference tangent audit."""
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
