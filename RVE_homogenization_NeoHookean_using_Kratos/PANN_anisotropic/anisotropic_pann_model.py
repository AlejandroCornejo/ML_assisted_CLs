"""Objective anisotropic energy surrogates for the two-dimensional RVE.

This module deliberately contains no D4 group average.  The material frame is
described by the two physical reference directions ``a0`` and ``b0``.  In that
frame the independent components of ``C = F.T @ F`` are

    a0.C.a0 = C11,  b0.C.b0 = C22,  a0.C.b0 = C12.

They are *joint invariants*: if a numerical reference basis is rotated, both
``C`` and the material directions are transformed and the three contractions
remain unchanged.  ``J = sqrt(det(C))`` is supplied as a redundant, but
physically explicit, volumetric feature.

``AnisotropicFreeEnergy`` is the high-fidelity PANN: a free Softplus MLP
with an affine reference correction.  ``AnisotropicPolyconvexEnergy`` is a
separate certified construction.  It uses directional stretches of ``F`` and
``cof(F)`` internally; its public input remains Green--Lagrange strain so its
stress is directly comparable to the FOM second Piola stress.
"""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

import torch
from torch import nn


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Return the preimage of a strictly positive Softplus value."""

    return torch.log(torch.expm1(value))


def _positive(raw: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    return torch.nn.functional.softplus(raw) + floor


def strain_to_c(normalised_strain: torch.Tensor, strain_scale: torch.Tensor) -> torch.Tensor:
    """Recover ``C`` from normalized ``[E11, E22, gamma12]`` samples.

    ``gamma12 = 2 E12``.  Thus ``C12 = gamma12`` and ``C = I + 2E``.
    """

    if normalised_strain.ndim != 2 or normalised_strain.shape[1] != 3:
        raise ValueError("normalised_strain must have shape (n_samples, 3).")
    strain = normalised_strain * strain_scale
    c11 = 1.0 + 2.0 * strain[:, 0]
    c22 = 1.0 + 2.0 * strain[:, 1]
    c12 = strain[:, 2]
    return torch.stack(
        (torch.stack((c11, c12), dim=1), torch.stack((c12, c22), dim=1)), dim=1
    )


def c_to_strain(c: torch.Tensor) -> torch.Tensor:
    """Return physical ``[E11, E22, gamma12]`` from a batch of 2x2 ``C``."""

    if c.ndim != 3 or c.shape[1:] != (2, 2):
        raise ValueError("c must have shape (n_samples, 2, 2).")
    return torch.stack(
        (0.5 * (c[:, 0, 0] - 1.0), 0.5 * (c[:, 1, 1] - 1.0), c[:, 0, 1]), dim=1
    )


def material_c_features(c: torch.Tensor, a0: torch.Tensor | None = None, b0: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``[a.C.a-1, b.C.b-1, a.C.b, J-1]``.

    The optional directions make the coordinate covariance explicit.  They
    are fixed to the RVE axes when omitted.  Directions must be transformed
    together with ``C`` under a relabelling of the reference basis.
    """

    if c.ndim != 3 or c.shape[1:] != (2, 2):
        raise ValueError("c must have shape (n_samples, 2, 2).")
    dtype, device = c.dtype, c.device
    if a0 is None:
        a0 = torch.tensor((1.0, 0.0), dtype=dtype, device=device)
    else:
        a0 = torch.as_tensor(a0, dtype=dtype, device=device)
    if b0 is None:
        b0 = torch.tensor((0.0, 1.0), dtype=dtype, device=device)
    else:
        b0 = torch.as_tensor(b0, dtype=dtype, device=device)
    if a0.shape != (2,) or b0.shape != (2,):
        raise ValueError("a0 and b0 must be two-dimensional vectors.")
    caa = torch.einsum("i,nij,j->n", a0, c, a0)
    cbb = torch.einsum("i,nij,j->n", b0, c, b0)
    cab = torch.einsum("i,nij,j->n", a0, c, b0)
    det_c = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1].square()
    if torch.any(det_c <= 0.0):
        raise ValueError("The constitutive models are defined only for C positive definite.")
    j = torch.sqrt(det_c)
    return torch.stack((caa - 1.0, cbb - 1.0, cab, j - 1.0), dim=1)


class SmoothEnergyMLP(nn.Module):
    """Unconstrained smooth scalar regressor for the free anisotropic PANN."""

    def __init__(self, widths: tuple[int, ...], n_inputs: int = 4) -> None:
        super().__init__()
        if n_inputs < 1:
            raise ValueError("n_inputs must be positive.")
        layers: list[nn.Module] = []
        previous = n_inputs
        for width in widths:
            layers.extend((nn.Linear(previous, width), nn.Softplus(beta=4.0)))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class AnisotropicFreeEnergy(nn.Module):
    """Free energy PANN in the material components of ``C`` and ``J``.

    The reference correction is allowed here because no polyconvexity claim is
    made for this model.  It enforces ``W(I)=0`` and ``S(I)=0`` exactly.
    """

    def __init__(self, *, strain_scale: float, feature_scale: torch.Tensor, widths: tuple[int, ...] = (128, 128, 64)) -> None:
        super().__init__()
        if strain_scale <= 0.0:
            raise ValueError("strain_scale must be positive.")
        scale = torch.as_tensor(feature_scale, dtype=torch.float32)
        if scale.shape != (4,) or torch.any(scale <= 0.0):
            raise ValueError("feature_scale must contain four positive entries.")
        self.register_buffer("strain_scale", torch.tensor(float(strain_scale), dtype=torch.float32))
        self.register_buffer("feature_scale", scale)
        self.widths = tuple(widths)
        self.base_energy = SmoothEnergyMLP(self.widths)

    def raw_energy(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        c = strain_to_c(normalised_strain, self.strain_scale)
        features = material_c_features(c) / self.feature_scale
        return self.base_energy(features)

    def reference_terms(self, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros((1, 3), dtype=self.strain_scale.dtype, device=self.strain_scale.device, requires_grad=True)
        raw_zero = self.raw_energy(zero)
        stress_zero = torch.autograd.grad(
            raw_zero, zero, grad_outputs=torch.ones_like(raw_zero), create_graph=create_graph
        )[0]
        return raw_zero, stress_zero

    def energy(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        """Return normalised energy, including the affine reference correction."""

        raw = self.raw_energy(normalised_strain)
        # The reference gradient depends on the network parameters during
        # training, so this graph must be kept for energy--stress consistency.
        raw_zero, stress_zero = self.reference_terms(create_graph=True)
        return raw - raw_zero - torch.sum(stress_zero * normalised_strain, dim=1, keepdim=True)

    def energy_and_stress(self, normalised_strain: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not normalised_strain.requires_grad:
            normalised_strain = normalised_strain.requires_grad_(True)
        energy = self.energy(normalised_strain)
        stress = torch.autograd.grad(
            energy, normalised_strain, grad_outputs=torch.ones_like(energy), create_graph=create_graph
        )[0]
        return energy, stress


class MetricPreconditionedFreeEnergy(nn.Module):
    """Free energy PANN in the locally isotropised tangent coordinates.

    ``metric_transform`` is the fixed three-by-three map ``T`` obtained from
    the FOM tangent at the reference state, such that

    ``D_0 = T.T @ D_hat @ T``

    with ``D_hat`` the closest isotropic tangent in the engineering-strain
    coordinates ``[E11,E22,gamma12]``.  The network receives
    ``[T e / e_scale, J-1]``.  It is a conditioning transformation only: this
    class has the same energy, reference-state and objectivity guarantees as
    the free C-PANN, and deliberately makes no polyconvexity claim.
    """

    def __init__(
        self,
        *,
        strain_scale: float,
        metric_transform: torch.Tensor,
        widths: tuple[int, ...] = (128, 128, 64),
    ) -> None:
        super().__init__()
        if strain_scale <= 0.0:
            raise ValueError("strain_scale must be positive.")
        transform = torch.as_tensor(metric_transform, dtype=torch.float32)
        if transform.shape != (3, 3):
            raise ValueError("metric_transform must have shape (3, 3).")
        self.register_buffer("strain_scale", torch.tensor(float(strain_scale), dtype=torch.float32))
        self.register_buffer("metric_transform", transform)
        self.widths = tuple(widths)
        self.base_energy = SmoothEnergyMLP(self.widths, n_inputs=4)

    def raw_energy(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        physical_strain = normalised_strain * self.strain_scale
        mapped_strain = physical_strain @ self.metric_transform.T / self.strain_scale
        c = strain_to_c(normalised_strain, self.strain_scale)
        j_minus_one = material_c_features(c)[:, 3:4]
        return self.base_energy(torch.cat((mapped_strain, j_minus_one), dim=1))

    def reference_terms(self, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros((1, 3), dtype=self.strain_scale.dtype, device=self.strain_scale.device, requires_grad=True)
        raw_zero = self.raw_energy(zero)
        stress_zero = torch.autograd.grad(
            raw_zero, zero, grad_outputs=torch.ones_like(raw_zero), create_graph=create_graph
        )[0]
        return raw_zero, stress_zero

    def energy(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        raw = self.raw_energy(normalised_strain)
        raw_zero, stress_zero = self.reference_terms(create_graph=True)
        return raw - raw_zero - torch.sum(stress_zero * normalised_strain, dim=1, keepdim=True)

    def energy_and_stress(self, normalised_strain: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not normalised_strain.requires_grad:
            normalised_strain = normalised_strain.requires_grad_(True)
        energy = self.energy(normalised_strain)
        stress = torch.autograd.grad(
            energy, normalised_strain, grad_outputs=torch.ones_like(energy), create_graph=create_graph
        )[0]
        return energy, stress


class PositiveICNN(nn.Module):
    """ICNN convex and coordinate-wise non-decreasing in its inputs."""

    def __init__(self, n_inputs: int, widths: tuple[int, ...]) -> None:
        super().__init__()
        if not widths:
            raise ValueError("At least one hidden layer is required.")
        self.raw_input_weights = nn.ParameterList()
        self.raw_hidden_weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        def positive_parameter(shape: tuple[int, ...], mean: float = 0.015) -> nn.Parameter:
            target = mean * torch.exp(0.07 * torch.randn(shape, dtype=torch.float64))
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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = None
        hidden_index = 0
        for raw_input, bias in zip(self.raw_input_weights, self.biases):
            preactivation = values @ _positive(raw_input, floor=1.0e-10).T + bias
            if hidden is not None:
                preactivation = preactivation + hidden @ _positive(self.raw_hidden_weights[hidden_index], floor=1.0e-10).T
                hidden_index += 1
            hidden = torch.nn.functional.softplus(preactivation)
        return (
            values @ _positive(self.raw_output_input, floor=1.0e-10).T
            + hidden @ _positive(self.raw_output_hidden, floor=1.0e-10).T
            + self.output_bias
        )


class AnisotropicPolyconvexEnergy(nn.Module):
    """Objective anisotropic PANN with a constructive polyconvex certificate.

    Each structural system ``k`` contains three fixed material directions and
    positive weights chosen so that

    ``sum_i w_ki d_ki outer d_ki = I``.

    Its four directional-minor invariants are

    ``Q^F_{k,2} = sum_i w_ki |F d_ki|^4``,
    ``Q^F_{k,3} = sum_i w_ki |F d_ki|^6``, and their two cofactor analogues.

    They are convex functions of ``F`` and ``cof(F)``, respectively.  More
    importantly, the balanced second moment above makes *each* invariant have
    an isotropic first derivative at ``F=I``.  Consequently the learned ICNN
    may weight the structural systems independently without producing an
    anisotropic reference stress.  The common isotropic pressure is cancelled
    exactly by ``-r log(J)``.

    The three direction systems are deliberately generic: they are not closed
    under a 90 degree rotation and no D4 group average appears anywhere in the
    model.  The result is polyconvex, stress-free at the reference state, and
    has no imposed square material symmetry.
    """

    def __init__(
        self,
        *,
        strain_scale: float,
        widths: tuple[int, ...] = (24, 24),
        feature_scale: torch.Tensor | None = None,
        volumetric_floor: float = 1.0e-10,
    ) -> None:
        super().__init__()
        if strain_scale <= 0.0:
            raise ValueError("strain_scale must be positive.")
        # Each row is a generic, non-D4-closed structural direction system.
        # The corresponding positive weights solve
        # sum_i w_i d_i outer d_i = I exactly (to numerical precision).
        # Thus no individual system has an anisotropic first derivative at the
        # reference state, although its finite-strain response is anisotropic.
        angle_sets = (
            (0.0, 50.0, 130.0),
            (20.0, 83.0, 151.0),
            (12.0, 76.0, 143.0),
        )
        weights = (
            (0.2959118089581524, 0.8520440955209238, 0.8520440955209238),
            (0.5570762841746417, 0.7941383556738040, 0.6487853601515543),
            (0.5760205291589184, 0.7929697991090714, 0.6310096717320103),
        )
        self.register_buffer(
            "directions",
            torch.tensor(
                tuple(
                    tuple((cos(angle * pi / 180.0), sin(angle * pi / 180.0)) for angle in angles)
                    for angles in angle_sets
                ),
                dtype=torch.float64,
            ),
        )
        self.register_buffer("direction_weights", torch.tensor(weights, dtype=torch.float64))
        self.register_buffer("strain_scale", torch.tensor(float(strain_scale), dtype=torch.float64))
        self.register_buffer("volumetric_floor", torch.tensor(float(volumetric_floor), dtype=torch.float64))
        self.widths = tuple(widths)
        # One isotropic direct minor; quartic and sixth-power direct/cofactor
        # invariants for each balanced system; and J/J^2.  The sixth powers
        # are essential: in 2D the balanced quartic sums alone happen to be
        # invariant under a 90-degree material rotation.
        n_features = 1 + 4 * len(self.directions) + 2
        if feature_scale is None:
            scale = torch.ones(n_features, dtype=torch.float64)
        else:
            scale = torch.as_tensor(feature_scale, dtype=torch.float64)
            if scale.shape != (n_features,) or torch.any(scale <= 0.0):
                raise ValueError(f"feature_scale must contain {n_features} positive entries.")
        self.register_buffer("feature_scale", scale)
        self.base_icnn = PositiveICNN(n_features, self.widths)
        self.raw_quadratic = nn.Parameter(_inverse_softplus(torch.tensor(1.0e-2, dtype=torch.float64)))

    @property
    def quadratic_coefficient(self) -> torch.Tensor:
        return _positive(self.raw_quadratic, floor=float(self.volumetric_floor))

    def _kinematics(self, normalised_strain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = strain_to_c(normalised_strain, self.strain_scale)
        determinant_c = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1].square()
        if torch.any(determinant_c <= 0.0):
            raise ValueError("The polyconvex PANN is defined only for det(C)>0.")
        cof_c = torch.stack(
            (torch.stack((c[:, 1, 1], -c[:, 0, 1]), dim=1), torch.stack((-c[:, 1, 0], c[:, 0, 0]), dim=1)),
            dim=1,
        )
        return c, cof_c, torch.sqrt(determinant_c)

    def structural_features(self, normalised_strain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c, cof_c, j = self._kinematics(normalised_strain)
        trace_c = (c[:, 0, 0] + c[:, 1, 1]).unsqueeze(1)
        # If t=d.C.d=|F d|^2, then t^2=|F d|^4 and t^3=|F d|^6 are convex
        # in F.  The positive weighted sums remain convex.  ``cof_c``
        # represents |cof(F)d|^2.
        direct_components = torch.einsum("kdi,nij,kdj->nkd", self.directions, c, self.directions)
        cofactor_components = torch.einsum("kdi,nij,kdj->nkd", self.directions, cof_c, self.directions)
        direct_quartic = torch.sum(self.direction_weights.unsqueeze(0) * direct_components.square(), dim=2)
        direct_sixth = torch.sum(self.direction_weights.unsqueeze(0) * direct_components.pow(3), dim=2)
        cofactor_quartic = torch.sum(self.direction_weights.unsqueeze(0) * cofactor_components.square(), dim=2)
        cofactor_sixth = torch.sum(self.direction_weights.unsqueeze(0) * cofactor_components.pow(3), dim=2)
        return torch.cat(
            (trace_c, direct_quartic, direct_sixth, cofactor_quartic, cofactor_sixth, j.unsqueeze(1), j.square().unsqueeze(1)),
            dim=1,
        ), j

    def _reference_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return H(0) and its necessarily isotropic reference pressure.

        The trace and J features have an isotropic reference derivative.  The
        balanced directional systems do too because their weighted second
        moments are exactly I.  The normal components are averaged only to
        eliminate roundoff.
        """

        zero = torch.zeros((1, 3), dtype=self.strain_scale.dtype, device=self.strain_scale.device, requires_grad=True)
        features, _ = self.structural_features(zero)
        h_zero = self.base_icnn(features / self.feature_scale)
        gradient = torch.autograd.grad(h_zero, zero, grad_outputs=torch.ones_like(h_zero), create_graph=True)[0]
        pressure = 0.5 * (gradient[0, 0] + gradient[0, 1]) / self.strain_scale
        return h_zero, pressure

    def energy(self, normalised_strain: torch.Tensor) -> torch.Tensor:
        features, j = self.structural_features(normalised_strain)
        # Positive constant rescaling is only a numerical preconditioner.  It
        # preserves convexity, monotonicity, polyconvexity, and the isotropic
        # reference-derivative argument.
        structural = self.base_icnn(features / self.feature_scale)
        structural_zero, pressure = self._reference_terms()
        quadratic = 0.5 * self.quadratic_coefficient * (j - 1.0).square()
        return structural - structural_zero - pressure * torch.log(j).unsqueeze(1) + quadratic.unsqueeze(1)

    def energy_and_stress(self, normalised_strain: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not normalised_strain.requires_grad:
            normalised_strain = normalised_strain.requires_grad_(True)
        energy = self.energy(normalised_strain)
        stress = torch.autograd.grad(
            energy, normalised_strain, grad_outputs=torch.ones_like(energy), create_graph=create_graph
        )[0]
        return energy, stress

    def certificate_summary(self) -> dict[str, float | int | list[int] | str]:
        _, pressure = self._reference_terms()
        second_moments = torch.einsum(
            "ki,kia,kib->kab", self.direction_weights, self.directions, self.directions
        )
        identity = torch.eye(2, dtype=self.directions.dtype, device=self.directions.device).unsqueeze(0)
        return {
            "direction_systems_degrees": [[0.0, 50.0, 130.0], [20.0, 83.0, 151.0], [12.0, 76.0, 143.0]],
            "weights": self.direction_weights.detach().cpu().tolist(),
            "number_of_structural_systems": int(len(self.directions)),
            "maximum_balanced_second_moment_error": float(torch.max(torch.abs(second_moments - identity)).detach().cpu()),
            "icnn_widths": list(self.widths),
            "barrier_coefficient": float(pressure.detach().cpu()),
            "quadratic_volumetric_coefficient": float(self.quadratic_coefficient.detach().cpu()),
            "feature_scale": self.feature_scale.detach().cpu().tolist(),
            "reference_mechanism": "all balanced directional-minor invariants and the trace/J inputs have isotropic reference derivatives; their common pressure is cancelled by -r log(J)",
        }


def load_anisotropic_free(checkpoint_path: Path, device: torch.device) -> tuple[AnisotropicFreeEnergy, float, float, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["model_configuration"]
    if configuration["kind"] != "anisotropic_c_mlp":
        raise ValueError("The supplied checkpoint is not the free anisotropic C-based PANN.")
    model = AnisotropicFreeEnergy(
        strain_scale=float(checkpoint["strain_scale"]),
        feature_scale=torch.tensor(configuration["feature_scale"], dtype=torch.float32),
        widths=tuple(configuration["widths"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, float(checkpoint["strain_scale"]), float(checkpoint["energy_scale"]), checkpoint


def load_metric_preconditioned_free(
    checkpoint_path: Path, device: torch.device
) -> tuple[MetricPreconditionedFreeEnergy, float, float, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["model_configuration"]
    if configuration["kind"] != "anisotropic_metric_mlp":
        raise ValueError("The supplied checkpoint is not the metric-preconditioned anisotropic PANN.")
    model = MetricPreconditionedFreeEnergy(
        strain_scale=float(checkpoint["strain_scale"]),
        metric_transform=torch.tensor(configuration["metric_transform"], dtype=torch.float32),
        widths=tuple(configuration["widths"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, float(checkpoint["strain_scale"]), float(checkpoint["energy_scale"]), checkpoint


def load_anisotropic_polyconvex(checkpoint_path: Path, device: torch.device) -> tuple[AnisotropicPolyconvexEnergy, float, float, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["model_configuration"]
    if configuration["kind"] != "anisotropic_polyconvex_directional_minors":
        raise ValueError("The supplied checkpoint is not the anisotropic polyconvex PANN.")
    model = AnisotropicPolyconvexEnergy(
        strain_scale=float(checkpoint["strain_scale"]),
        widths=tuple(configuration["icnn_widths"]),
        feature_scale=torch.tensor(configuration["feature_scale"], dtype=torch.float64)
        if "feature_scale" in configuration else None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict="feature_scale" in configuration)
    model.eval()
    return model, float(checkpoint["strain_scale"]), float(checkpoint["energy_scale"]), checkpoint
