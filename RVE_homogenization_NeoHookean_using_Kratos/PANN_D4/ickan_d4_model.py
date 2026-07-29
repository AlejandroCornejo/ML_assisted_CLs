"""D4-symmetric ICKAN energies on the same strain/energy/stress protocol.

This module deliberately separates two questions that are easy to conflate:

* ``direct`` is a flexible spline ICKAN of the three normalized Green--
  Lagrange strain components.  D4 averaging, the reference state and
  energy--stress consistency are exact.  It is a fair predictive comparison
  with the selected direct PANN-D4.
* ``minor_features`` supplies the ICKAN with objective directional measures
  of ``F`` and ``cof(F)`` (computed from ``C=I+2E``), plus ``J``.  In this
  mode its stress-free reference correction is ``-r log(J)`` and it has a
  positive volumetric growth term.  It is consequently polyconvex by design.

The ICKAN package projects spline coefficients to increasing convex sequences.
With ``base_fun='zero'``, positive spline scales, no symbolic branch, and
fixed positive affine scales, every scalar edge is convex and non-decreasing.
Sums and compositions preserve those two properties.  Thus the direct mode
is convex in its strain input (not automatically polyconvex), whereas the
minor mode has the required convex, coordinatewise non-decreasing function of
``(F, cof(F), J)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ICKAN_REPOSITORY = Path("/home/kratos/ICKANs")
if str(ICKAN_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(ICKAN_REPOSITORY))

try:
    import ickan as KAN
except ImportError as error:  # pragma: no cover - an environment diagnostic
    raise ImportError(
        "The actual ICKAN package was not found at /home/kratos/ICKANs. "
        "This experiment must not silently fall back to pykan."
    ) from error


def square_orbit(strain: torch.Tensor) -> torch.Tensor:
    """Return the four distinct D4 actions on ``[E_xx,E_yy,gamma_xy]``."""

    if strain.ndim != 2 or strain.shape[1] != 3:
        raise ValueError("strain must have shape (n_samples, 3).")
    exx, eyy, gamma = strain[:, 0], strain[:, 1], strain[:, 2]
    return torch.stack(
        (
            torch.stack((exx, eyy, gamma), dim=1),
            torch.stack((eyy, exx, -gamma), dim=1),
            torch.stack((exx, eyy, -gamma), dim=1),
            torch.stack((eyy, exx, gamma), dim=1),
        ),
        dim=1,
    )


class D4ICKANEnergy(nn.Module):
    """A true spline ICKAN, wrapped as an exact D4 energy.

    ``strain`` always denotes the strain normalized by ``strain_scale``.
    The direct formulation therefore sees three inputs.  The minor-informed
    formulation reconstructs the physical strain internally and sees ten
    scalar features: four directional measures of ``C``, four of
    ``cof(C)``, ``J``, and ``J**2``.  The axes and diagonals form a D4-closed
    direction set.  The group average is retained even though the feature set
    itself is closed, because it prevents a KAN's ordered input channels from
    selecting a preferred square orientation.
    """

    MODES = {"direct", "minor_features"}

    def __init__(
        self,
        *,
        strain_scale: float,
        mode: str = "direct",
        widths: tuple[int, ...] = (12,),
        grid: int = 20,
        spline_order: int = 3,
        seed: int = 20260803,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}, got {mode!r}.")
        if strain_scale <= 0.0:
            raise ValueError("strain_scale must be positive.")
        self.mode = mode
        # An empty tuple is a legitimate one-layer KAN.  It is useful here as
        # a numerically tractable, genuinely convex-in-input spline baseline.
        self.widths = tuple(int(width) for width in widths)
        self.grid = int(grid)
        self.spline_order = int(spline_order)
        self.seed = int(seed)
        self.register_buffer("strain_scale", torch.tensor(float(strain_scale), dtype=torch.float32))
        # Fixed, physically interpretable normalization for the minor inputs:
        # every directional measure and J is one in the reference state; J^2
        # is one too.  The scales cover the prescribed Stage--1 loading range
        # without peeking at Stage--10 and keep the fixed ICKAN grids useful.
        self.register_buffer(
            "minor_feature_center",
            torch.ones(10, dtype=torch.float32),
        )
        self.register_buffer(
            "minor_feature_scale",
            torch.tensor([4.0] * 8 + [4.0, 24.0], dtype=torch.float32),
        )
        self.raw_volumetric_quadratic = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(1.0e-2, dtype=torch.float32)))
        )
        n_input = 3 if mode == "direct" else 10
        # ``zero`` avoids adding SiLU residual edges, whose nonconvexity would
        # obscure even the limited convex-spline property of the ICKAN package.
        self.base_energy = KAN.MultKAN(
            width=[n_input, *self.widths, 1],
            grid=self.grid,
            k=self.spline_order,
            base_fun="zero",
            noise_scale=0.01,
            symbolic_enabled=False,
            affine_trainable=False,
            grid_eps=0.15,
            grid_range=[-1.0, 1.0],
            auto_save=False,
            save_act=False,
            seed=self.seed,
            device="cpu",
        )

    @property
    def volumetric_quadratic(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_volumetric_quadratic) + 1.0e-8

    def _minor_physical_features(self, normalised_strain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return physical minor features and ``J`` for a normalized strain."""

        strain = normalised_strain * self.strain_scale
        c11 = 1.0 + 2.0 * strain[:, 0]
        c22 = 1.0 + 2.0 * strain[:, 1]
        c12 = strain[:, 2]  # C_12 = gamma_xy.
        determinant_c = c11 * c22 - c12.square()
        if torch.any(determinant_c <= 0.0):
            raise ValueError("minor_features ICKAN is defined only for det(I+2E)>0.")
        # a=e1,e2,(e1+e2)/sqrt(2),(e1-e2)/sqrt(2): a.C.a
        q_c = torch.stack(
            (
                c11,
                c22,
                0.5 * (c11 + c22 + 2.0 * c12),
                0.5 * (c11 + c22 - 2.0 * c12),
            ),
            dim=1,
        )
        # cof(C) = [[C22,-C12],[-C12,C11]].
        q_cof = torch.stack(
            (
                c22,
                c11,
                0.5 * (c11 + c22 - 2.0 * c12),
                0.5 * (c11 + c22 + 2.0 * c12),
            ),
            dim=1,
        )
        j = torch.sqrt(determinant_c)
        return torch.cat((q_c, q_cof, j.unsqueeze(1), j.square().unsqueeze(1)), dim=1), j

    def feature_vector(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        """Return direct inputs or objective directional-minor features."""

        if self.mode == "direct":
            return normalised_strain

        physical_features, _ = self._minor_physical_features(normalised_strain)
        return (physical_features - self.minor_feature_center) / self.minor_feature_scale

    def initialise_grid(self, normalised_strain: torch.Tensor) -> None:
        """Adapt the fixed spline grids once, using Stage--1 data only."""

        with torch.no_grad():
            orbit = square_orbit(normalised_strain)
            n_samples, n_actions, _ = orbit.shape
            features = self.feature_vector(orbit.reshape(n_samples * n_actions, 3))
            self.base_energy.update_grid_from_samples(features)

    def group_average_raw_energy(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        orbit = square_orbit(normalised_strain)
        n_samples, n_actions, _ = orbit.shape
        features = self.feature_vector(orbit.reshape(n_samples * n_actions, 3))
        raw = self.base_energy(features)
        return raw.reshape(n_samples, n_actions, 1).mean(dim=1)

    def reference_terms(self, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference terms for the direct-strain ICKAN correction."""
        zero = torch.zeros((1, 3), dtype=self.strain_scale.dtype, device=self.strain_scale.device)
        zero.requires_grad_(True)
        energy_zero = self.group_average_raw_energy(zero)
        stress_zero = torch.autograd.grad(
            energy_zero,
            zero,
            grad_outputs=torch.ones_like(energy_zero),
            create_graph=create_graph,
        )[0]
        return energy_zero, stress_zero

    def minor_reference_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``H(0)`` and the positive physical pressure coefficient r."""

        zero = torch.zeros((1, 3), dtype=self.strain_scale.dtype, device=self.strain_scale.device, requires_grad=True)
        energy_zero = self.group_average_raw_energy(zero)
        gradient_normalised = torch.autograd.grad(
            energy_zero, zero, grad_outputs=torch.ones_like(energy_zero), create_graph=True
        )[0]
        pressure = 0.5 * (gradient_normalised[0, 0] + gradient_normalised[0, 1]) / self.strain_scale
        return energy_zero, pressure

    def energy(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        """Return the normalized stored energy before differentiating it."""

        raw = self.group_average_raw_energy(normalised_strain)
        if self.mode == "direct":
            energy_zero, stress_zero = self.reference_terms(create_graph=True)
            return raw - energy_zero - torch.sum(stress_zero * normalised_strain, dim=1, keepdim=True)
        energy_zero, pressure = self.minor_reference_terms()
        _, j = self._minor_physical_features(normalised_strain)
        return (
                raw - energy_zero
                - pressure * torch.log(j).unsqueeze(1)
                + 0.5 * self.volumetric_quadratic * (j - 1.0).square().unsqueeze(1)
        )

    def energy_and_stress(
        self, normalised_strain: torch.Tensor, *, create_graph: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reference-corrected normalized energy and its derivative."""

        if not normalised_strain.requires_grad:
            normalised_strain = normalised_strain.requires_grad_(True)
        energy = self.energy(normalised_strain)
        stress = torch.autograd.grad(
            energy,
            normalised_strain,
            grad_outputs=torch.ones_like(energy),
            create_graph=create_graph,
        )[0]
        return energy, stress


def load_ickan_d4(
    checkpoint_path: Path, device: torch.device
) -> tuple[D4ICKANEnergy, float, float, dict]:
    """Load an ICKAN-D4 checkpoint made by ``train_ickan_d4.py``."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["model_configuration"]
    if configuration["kind"] != "d4_ickan":
        raise ValueError("The supplied checkpoint is not an ICKAN-D4 checkpoint.")
    model = D4ICKANEnergy(
        strain_scale=float(checkpoint["strain_scale"]),
        mode=str(configuration["mode"]),
        widths=tuple(int(value) for value in configuration["widths"]),
        grid=int(configuration["grid"]),
        spline_order=int(configuration["spline_order"]),
        seed=int(configuration["seed"]),
    ).to(device)
    state = dict(checkpoint["model_state_dict"])
    # Checkpoints produced before minor-feature normalization was introduced
    # are direct-input models and legitimately lack these inert buffers.
    for name in ("minor_feature_center", "minor_feature_scale"):
        state.setdefault(name, getattr(model, name))
    state.setdefault("raw_volumetric_quadratic", model.raw_volumetric_quadratic)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, float(checkpoint["strain_scale"]), float(checkpoint["energy_scale"]), checkpoint
