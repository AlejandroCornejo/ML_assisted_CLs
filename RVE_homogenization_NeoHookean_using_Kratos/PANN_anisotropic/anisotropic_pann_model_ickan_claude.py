"""Input-convex Kolmogorov-Arnold Network (ICKAN) variant of the certified
anisotropic polyconvex PANN.

This reuses ``AnisotropicPolyconvexEnergy`` from ``anisotropic_pann_model.py``
verbatim -- the balanced structural direction systems, the 15-dimensional
feature vector ``z_pc``, the reference-state cancellation, and the
volumetric barrier -- and only swaps the convex core ``Phi_theta`` from a
Softplus input-convex neural network (ICNN, Amos et al. 2017) to an
input-convex Kolmogorov-Arnold network (ICKAN, Thakolkaran et al. 2025,
"Can KAN CANs?", CMAME 443:118089), using the reference implementation at
https://github.com/mmc-group/ICKANs.

Every proof in ``PANN_anisotropic_claude.tex`` (Propositions 1-4: the
isotropic reference derivative, the exact stress cancellation, polyconvexity
of the energy, and non-negativity of the barrier coefficient) is stated for
"any Phi_theta that is convex and coordinate-wise non-decreasing in its
inputs" -- it never assumes Phi_theta is specifically an ICNN. The ICKAN
core has that same property by a different mechanism: each edge is a
B-spline whose control points are reparameterized (in
``ickan.spline.coef2curve``) as a running double cumulative sum of
non-negative increments, which makes every edge simultaneously convex and
non-decreasing in its own scalar input; the "zero" base function and
non-trainable (identity) affine transforms between layers mean the sum
over edges, and hence the whole network, inherits both properties by the
same composition rule used for the ICNN. So every proposition in the
memorandum applies unchanged; only the numerical results (Section 11 of
the memorandum) can differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from anisotropic_pann_model import AnisotropicPolyconvexEnergy

_ICKAN_REPO = Path("/home/kratos/ICKANs")


def _load_ickan_kan_class():
    if str(_ICKAN_REPO) not in sys.path:
        if not _ICKAN_REPO.exists():
            raise FileNotFoundError(
                f"Expected the reference ICKAN implementation at {_ICKAN_REPO} "
                "(github.com/mmc-group/ICKANs); it is missing."
            )
        sys.path.insert(0, str(_ICKAN_REPO))
    from ickan import KAN  # noqa: PLC0415

    return KAN


class ICKANCore(nn.Module):
    """Drop-in replacement for ``PositiveICNN``: convex, coordinate-wise
    non-decreasing scalar function of its input, built from an input-convex
    Kolmogorov-Arnold network instead of a Softplus ICNN.
    """

    def __init__(
        self,
        n_inputs: int,
        widths: tuple[int, ...],
        *,
        grid: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (0.0, 1.3),
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not widths:
            raise ValueError("At least one hidden layer is required.")
        KAN = _load_ickan_kan_class()
        full_width = [n_inputs, *widths, 1]
        self.core = KAN(
            width=full_width,
            grid=grid,
            k=spline_order,
            seed=seed,
            device="cpu",
            base_fun="zero",
            grid_eps=1.0,
            grid_range_0=list(grid_range),
            sp_trainable=True,
            sb_trainable=False,
            symbolic_enabled=False,
            affine_trainable=False,
            auto_save=False,
        ).to(torch.float64)
        # ``base_fun="zero"`` makes scale_base provably inert (it multiplies a
        # constant zero), and the symbolic-regression branch is unused here;
        # freeze both so the optimizer does not carry dead state for them.
        for name, parameter in self.core.named_parameters():
            if "scale_base" in name or "symbolic_fun" in name:
                parameter.requires_grad_(False)
        self.grid = grid
        self.spline_order = spline_order
        self.grid_range = tuple(grid_range)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.core(values)


class AnisotropicPolyconvexEnergyICKAN(AnisotropicPolyconvexEnergy):
    """Same construction and certificate as ``AnisotropicPolyconvexEnergy``,
    with an ICKAN core in place of the ICNN core.
    """

    def __init__(
        self,
        *,
        strain_scale: float,
        ickan_hidden: tuple[int, ...] = (8,),
        ickan_grid: int = 6,
        ickan_spline_order: int = 3,
        ickan_grid_range: tuple[float, float] = (0.0, 1.3),
        ickan_seed: int = 0,
        feature_scale: torch.Tensor | None = None,
        volumetric_floor: float = 1.0e-10,
    ) -> None:
        super().__init__(
            strain_scale=strain_scale,
            widths=ickan_hidden,
            feature_scale=feature_scale,
            volumetric_floor=volumetric_floor,
        )
        n_features = int(self.feature_scale.shape[0])
        self.base_icnn = ICKANCore(
            n_features,
            self.widths,
            grid=ickan_grid,
            spline_order=ickan_spline_order,
            grid_range=ickan_grid_range,
            seed=ickan_seed,
        )
        self.ickan_grid = ickan_grid
        self.ickan_spline_order = ickan_spline_order
        self.ickan_grid_range = tuple(ickan_grid_range)

    def certificate_summary(self) -> dict[str, float | int | list[int] | str]:
        summary = super().certificate_summary()
        summary["core"] = "input-convex Kolmogorov-Arnold network (ICKAN)"
        summary["ickan_hidden_widths"] = list(self.widths)
        summary["ickan_grid"] = self.ickan_grid
        summary["ickan_spline_order"] = self.ickan_spline_order
        summary["ickan_grid_range"] = list(self.ickan_grid_range)
        summary.pop("icnn_widths", None)
        return summary


def load_anisotropic_polyconvex_ickan(
    checkpoint_path: Path, device: torch.device
) -> tuple[AnisotropicPolyconvexEnergyICKAN, float, float, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["model_configuration"]
    if configuration["kind"] != "anisotropic_polyconvex_ickan_directional_minors":
        raise ValueError("The supplied checkpoint is not the anisotropic polyconvex ICKAN.")
    model = AnisotropicPolyconvexEnergyICKAN(
        strain_scale=float(checkpoint["strain_scale"]),
        ickan_hidden=tuple(configuration["ickan_hidden_widths"]),
        ickan_grid=int(configuration["ickan_grid"]),
        ickan_spline_order=int(configuration["ickan_spline_order"]),
        ickan_grid_range=tuple(configuration["ickan_grid_range"]),
        feature_scale=torch.tensor(configuration["feature_scale"], dtype=torch.float64),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, float(checkpoint["strain_scale"]), float(checkpoint["energy_scale"]), checkpoint
