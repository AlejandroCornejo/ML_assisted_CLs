"""A D4-symmetric polyconvex energy, evaluated from Green--Lagrange strain.

The public input is ``e = [E_xx, E_yy, gamma_xy]`` with
``gamma_xy = 2 E_xy``.  Internally the model constructs

    C = I + 2 E,       J = sqrt(det(C)),
    q_a^F   = |F a|^2       = a . C a,
    q_a^H   = |cof(F) a|^2 = a . cof(C) a.

The vector of direct and cofactor directional stretches is passed to an ICNN
whose weights are strictly positive.  It is therefore convex and
coordinate-wise non-decreasing in q.  Averaging this base potential over the
four distinct actions of D4 makes the material symmetry exact.  Since every
q component is convex either in F or in cof(F), the resulting structural term
has a convex representation in (F, cof(F)).  Adding a convex function of J
gives a polyconvex energy.

The reference stress is cancelled only with ``-r log(J)``.  Unlike a generic
affine correction in E, this retains the polyconvex representation.  The
positive quadratic term in J gives energy growth at large volume, while the
logarithmic term is an exact barrier at J -> 0+.
"""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

import torch
from torch import nn


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Return x such that softplus(x)=value for positive ``value``."""

    return torch.log(torch.expm1(value))


def _positive(raw: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    return torch.nn.functional.softplus(raw) + floor


class PositiveICNN(nn.Module):
    """ICNN convex and non-decreasing in every input component.

    All weights are strictly positive and Softplus is convex and increasing.
    Therefore each hidden unit, and ultimately the scalar output, is convex
    and coordinate-wise non-decreasing in its input vector q.
    """

    def __init__(self, n_inputs: int, widths: tuple[int, ...]) -> None:
        super().__init__()
        if not widths:
            raise ValueError("At least one hidden layer is required.")
        self.raw_input_weights = nn.ParameterList()
        self.raw_hidden_weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        def positive_parameter(shape: tuple[int, ...], mean: float = 0.010) -> nn.Parameter:
            # Breaking neuron permutation symmetry is essential: identical
            # positive weights would make every neuron receive the same
            # gradient and would collapse a wide ICNN to one effective unit.
            target = mean * torch.exp(0.05 * torch.randn(shape, dtype=torch.float64))
            return nn.Parameter(_inverse_softplus(target))

        previous = 0
        for width in widths:
            self.raw_input_weights.append(positive_parameter((width, n_inputs)))
            if previous:
                self.raw_hidden_weights.append(positive_parameter((width, previous)))
            self.biases.append(nn.Parameter(0.01 * torch.randn(width, dtype=torch.float64)))
            previous = width
        self.raw_output_input = positive_parameter((1, n_inputs))
        self.raw_output_hidden = positive_parameter((1, previous))
        self.output_bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        hidden = None
        hidden_index = 0
        for layer_index, (raw_input, bias) in enumerate(zip(self.raw_input_weights, self.biases)):
            preactivation = q @ _positive(raw_input, floor=1.0e-10).T + bias
            if hidden is not None:
                preactivation = preactivation + hidden @ _positive(self.raw_hidden_weights[hidden_index], floor=1.0e-10).T
                hidden_index += 1
            hidden = torch.nn.functional.softplus(preactivation)
        return q @ _positive(self.raw_output_input, floor=1.0e-10).T + hidden @ _positive(self.raw_output_hidden, floor=1.0e-10).T + self.output_bias


def _direction_rows() -> list[tuple[float, float]]:
    """D4-closed set of unoriented material directions.

    It contains axial, diagonal and two off-axis D4 orbits.  Signs are
    immaterial because only squared directional stretches are used.
    """

    rows: list[tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0)]
    inv_sqrt_two = 2.0 ** -0.5
    rows.extend(((inv_sqrt_two, inv_sqrt_two), (inv_sqrt_two, -inv_sqrt_two)))
    for angle in (pi / 8.0, pi / 12.0):
        c, s = cos(angle), sin(angle)
        rows.extend(((c, s), (c, -s), (s, c), (s, -c)))
    return rows


class PolyconvexD4Energy(nn.Module):
    """Objective, D4-symmetric energy with analytic polyconvexity certificate.

    The output is normalised energy.  Physical Green--Lagrange strain is
    recovered inside the model before the nonlinear kinematic map is applied.
    """

    def __init__(
        self,
        *,
        strain_scale: float,
        widths: tuple[int, ...] = (64, 64, 32),
        volumetric_floor: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if strain_scale <= 0.0:
            raise ValueError("strain_scale must be positive.")
        directions = torch.tensor(_direction_rows(), dtype=torch.float64)
        self.register_buffer("directions", directions)
        self.register_buffer("strain_scale", torch.tensor(float(strain_scale), dtype=torch.float64))
        self.register_buffer("volumetric_floor", torch.tensor(float(volumetric_floor), dtype=torch.float64))
        self.widths = tuple(widths)
        # Besides directional direct/cofactor measures, J and J^2 give the
        # monotone convex network an explicitly volumetric degree of freedom.
        # Both are convex functions of J>0, so the certificate is unchanged.
        self.base_icnn = PositiveICNN(2 * len(directions) + 2, self.widths)
        self.raw_volumetric_quadratic = nn.Parameter(_inverse_softplus(torch.tensor(1.0e-2, dtype=torch.float64)))

    @property
    def volumetric_quadratic(self) -> torch.Tensor:
        return _positive(self.raw_volumetric_quadratic, floor=float(self.volumetric_floor))

    @staticmethod
    def _square_orbit(strain: torch.Tensor) -> torch.Tensor:
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

    def _kinematics(self, normalised_strain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if normalised_strain.ndim != 2 or normalised_strain.shape[1] != 3:
            raise ValueError("normalised_strain must have shape (n_samples, 3).")
        strain = normalised_strain * self.strain_scale
        c11 = 1.0 + 2.0 * strain[:, 0]
        c22 = 1.0 + 2.0 * strain[:, 1]
        c12 = strain[:, 2]  # C_12 = 2 E_12 = gamma_xy.
        determinant_c = c11 * c22 - c12.square()
        if torch.any(determinant_c <= 0.0):
            raise ValueError("The model is defined only for physical states I + 2 E positive definite.")
        c = torch.stack(
            (torch.stack((c11, c12), dim=1), torch.stack((c12, c22), dim=1)),
            dim=1,
        )
        cof_c = torch.stack(
            (torch.stack((c22, -c12), dim=1), torch.stack((-c12, c11), dim=1)),
            dim=1,
        )
        return c, cof_c, torch.sqrt(determinant_c)

    def _features(self, normalised_strain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c, cof_c, j = self._kinematics(normalised_strain)
        q_direct = torch.einsum("di,nij,dj->nd", self.directions, c, self.directions)
        q_cofactor = torch.einsum("di,nij,dj->nd", self.directions, cof_c, self.directions)
        return torch.cat((q_direct, q_cofactor, j.unsqueeze(1), j.square().unsqueeze(1)), dim=1), j

    def structural_energy(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        """D4-average of the convex, monotone ICNN structural potential."""

        orbit = self._square_orbit(normalised_strain)
        n_samples, n_actions, _ = orbit.shape
        features, _ = self._features(orbit.reshape(n_samples * n_actions, 3))
        value = self.base_icnn(features)
        return value.reshape(n_samples, n_actions, 1).mean(dim=1)

    def _reference_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return H(0) and r so H'(0)-r (log J)'(0)=0.

        D4 symmetry makes the normal components identical and the shear
        component zero.  We nevertheless average the two normal components to
        remove insignificant floating-point roundoff.
        """

        zero = torch.zeros((1, 3), dtype=self.strain_scale.dtype, device=self.strain_scale.device, requires_grad=True)
        h_zero = self.structural_energy(zero)
        gradient_wrt_normalised_strain = torch.autograd.grad(
            h_zero,
            zero,
            grad_outputs=torch.ones_like(h_zero),
            create_graph=True,
        )[0]
        pressure = 0.5 * (gradient_wrt_normalised_strain[0, 0] + gradient_wrt_normalised_strain[0, 1]) / self.strain_scale
        return h_zero, pressure

    def energy(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        """Return normalised stored energy on the physical domain det(I+2E)>0."""

        structural = self.structural_energy(normalised_strain)
        _, j = self._features(normalised_strain)
        h_zero, pressure = self._reference_terms()
        value = structural - h_zero
        value = value - pressure * torch.log(j).unsqueeze(1)
        value = value + 0.5 * self.volumetric_quadratic * (j - 1.0).square().unsqueeze(1)
        return value

    def energy_and_stress(self, normalised_strain: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalised energy and its derivative in normalised strain."""

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

    def certificate_summary(self) -> dict[str, float | int | list[int]]:
        _, pressure = self._reference_terms()
        return {
            "logarithmic_barrier_coefficient": float(pressure.detach().cpu()),
            "quadratic_volumetric_coefficient": float(self.volumetric_quadratic.detach().cpu()),
            "number_of_structural_directions": int(len(self.directions)),
            "icnn_widths": list(self.widths),
        }


def load_polyconvex_pann(checkpoint_path: Path, device: torch.device) -> tuple[PolyconvexD4Energy, float, float, dict]:
    """Load a trained polyconvex D4 PANN checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["model_configuration"]
    if configuration["kind"] != "polyconvex_d4_directional_minors":
        raise ValueError("The supplied checkpoint is not a polyconvex D4 directional-minors PANN.")
    model = PolyconvexD4Energy(
        strain_scale=float(checkpoint["strain_scale"]),
        widths=tuple(configuration["icnn_widths"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, float(checkpoint["strain_scale"]), float(checkpoint["energy_scale"]), checkpoint
