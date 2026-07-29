"""Tier-1 baseline: direct strain-to-stress regression, no energy potential.

This is the anisotropic-RVE analogue of As'ad, Avery and Farhat's (2022,
IJNME 123:2738-2759) "standard regression ANN": a plain MLP that maps the
normalized strain directly to a predicted stress, with no scalar potential
anywhere in the architecture. Stress is NOT obtained by differentiating
anything.

Admissibility status, in the checklist of PANN_anisotropic_claude.tex:
  (C1) thermodynamic consistency: NOT guaranteed. Nothing forces the
       predicted stress field to be a gradient, so there is no reason for
       the cyclic integral (oint S . dE) to vanish around a closed strain
       loop; Section "A regression-ANN baseline: violating (C1) directly"
       verifies this fails in practice.
  (C2) objectivity: guaranteed, for the same reason as every other model
       in this project -- the input e=[E11,E22,gamma12] is already built
       from the objective right Cauchy-Green tensor C, not from F.
  (C3) symmetry: holds trivially (only three independent stress numbers
       are ever produced).
  (C4) normalization: NOT guaranteed architecturally (no reference-state
       correction is applied).
  (C5), (C6): not applicable; there is no energy functional to be
       polyconvex or coercive.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class AnisotropicRegressionStress(nn.Module):
    """Direct normalized-strain -> normalized-stress regression MLP."""

    def __init__(self, *, strain_scale: float, widths: tuple[int, ...] = (128, 128, 64)) -> None:
        super().__init__()
        if strain_scale <= 0.0:
            raise ValueError("strain_scale must be positive.")
        self.register_buffer("strain_scale", torch.tensor(float(strain_scale), dtype=torch.float32))
        self.widths = tuple(widths)
        layers: list[nn.Module] = []
        previous = 3
        for width in widths:
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        layers.append(nn.Linear(previous, 3))
        self.net = nn.Sequential(*layers)

    def stress(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        """Predicted stress, directly, with no differentiation involved."""

        return self.net(normalised_strain)


def load_anisotropic_regression(
    checkpoint_path: Path, device: torch.device
) -> tuple[AnisotropicRegressionStress, float, float, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["model_configuration"]
    if configuration["kind"] != "anisotropic_regression_baseline":
        raise ValueError("The supplied checkpoint is not the direct-regression baseline.")
    model = AnisotropicRegressionStress(
        strain_scale=float(checkpoint["strain_scale"]),
        widths=tuple(configuration["widths"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, float(checkpoint["strain_scale"]), float(checkpoint["energy_scale"]), checkpoint
