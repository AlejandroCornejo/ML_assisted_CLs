"""The selected PANN-D4 energy architecture.

The input is ``[E_xx, E_yy, gamma_xy]``, with ``gamma_xy = 2 E_xy``.
The scalar MLP is evaluated on the four distinct actions of the square
symmetry group and averaged.  A reference correction then makes energy and
stress exactly zero at ``E = 0``.  Stress is the derivative of that energy.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SmoothEnergyMLP(nn.Module):
    """Scalar MLP used by the selected PANN-D4."""

    def __init__(self, widths: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = 3
        for width in widths:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.Softplus(beta=4.0))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, strain: torch.Tensor) -> torch.Tensor:
        return self.network(strain)


def square_orbit(strain: torch.Tensor) -> torch.Tensor:
    """Return the four distinct D4 transforms of a batch of strains.

    For ``(E_xx, E_yy, gamma_xy)`` the four distinct vectors are

    ``(E_xx, E_yy, gamma_xy)``, ``(E_yy, E_xx, -gamma_xy)``,
    ``(E_xx, E_yy, -gamma_xy)``, and ``(E_yy, E_xx, gamma_xy)``.

    The eight matrices of D4 contain each of these twice when applied to a
    symmetric second-order tensor; averaging these four is therefore exactly
    the same group average without duplicated work.
    """

    if strain.ndim != 2 or strain.shape[1] != 3:
        raise ValueError("strain must have shape (n_samples, 3).")
    exx, eyy, gamma_xy = strain[:, 0], strain[:, 1], strain[:, 2]
    return torch.stack(
        (
            torch.stack((exx, eyy, gamma_xy), dim=1),
            torch.stack((eyy, exx, -gamma_xy), dim=1),
            torch.stack((exx, eyy, -gamma_xy), dim=1),
            torch.stack((eyy, exx, gamma_xy), dim=1),
        ),
        dim=1,
    )


class PANN_D4Energy(nn.Module):
    """Square-symmetric scalar energy with exact reference correction."""

    def __init__(self, widths: tuple[int, ...], invariant_feature_scale: torch.Tensor) -> None:
        super().__init__()
        self.base_energy = SmoothEnergyMLP(widths)
        # It is not used by the direct-input MLP, but it is retained so that
        # the selected checkpoint remains self-describing and loads verbatim.
        self.register_buffer(
            "invariant_feature_scale",
            torch.as_tensor(invariant_feature_scale, dtype=torch.float32),
        )

    def group_average_raw_energy(self, strain: torch.Tensor) -> torch.Tensor:
        orbit = square_orbit(strain)
        n_samples, n_actions, _ = orbit.shape
        raw_energy = self.base_energy(orbit.reshape(n_samples * n_actions, 3))
        return raw_energy.reshape(n_samples, n_actions, 1).mean(dim=1)

    def reference_terms(self, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros((1, 3), dtype=self.invariant_feature_scale.dtype, device=self.invariant_feature_scale.device)
        zero.requires_grad_(True)
        energy_zero = self.group_average_raw_energy(zero)
        stress_zero = torch.autograd.grad(
            energy_zero,
            zero,
            grad_outputs=torch.ones_like(energy_zero),
            create_graph=create_graph,
        )[0]
        return energy_zero, stress_zero

    def energy_and_stress(self, strain: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dimensionless energy and its derivative in strain coordinates."""

        if not strain.requires_grad:
            strain = strain.requires_grad_(True)
        raw_energy = self.group_average_raw_energy(strain)
        energy_zero, stress_zero = self.reference_terms(create_graph=create_graph)
        energy = raw_energy - energy_zero - torch.sum(stress_zero * strain, dim=1, keepdim=True)
        stress = torch.autograd.grad(
            energy,
            strain,
            grad_outputs=torch.ones_like(energy),
            create_graph=create_graph,
        )[0]
        return energy, stress


def load_selected_pann(checkpoint_path: Path, device: torch.device) -> tuple[PANN_D4Energy, float, float, dict]:
    """Load the one selected all-trajectory PANN checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["model_configuration"]
    if configuration["kind"] != "d4_mlp":
        raise ValueError("The supplied checkpoint is not the selected direct-input PANN-D4.")
    model = PANN_D4Energy(
        tuple(configuration["widths"]),
        torch.tensor(configuration["invariant_feature_scale"], dtype=torch.float32),
    ).to(device)
    state = dict(checkpoint["model_state_dict"])
    # The historical training wrapper stored the eight D4 matrices as a
    # buffer.  This compact implementation uses their four unique actions.
    state.pop("group_matrices", None)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, float(checkpoint["strain_scale"]), float(checkpoint["energy_scale"]), checkpoint
