"""Vectorized PANN constitutive law: drop-in replacement for
core/fom_solver_rve.py's _neo_hookean_pk2_2d_vectorized, evaluating PK2
stress and the material tangent from a trained PANN (certified ICNN or
free/uncertified) via PyTorch autodiff instead of the closed-form
Neo-Hookean law. Same I/O convention: E_voigt (N,3) = [E11, E22, gamma12]
with gamma12 = 2*E12 = C12, in; S_voigt (N,3), CC (N,3,3) out, both numpy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PANN_DIR = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/pann/anisotropic")
sys.path.insert(0, str(PANN_DIR))

from anisotropic_pann_model import load_anisotropic_free, load_anisotropic_polyconvex  # noqa: E402
from anisotropic_pann_model_ickan_claude import load_anisotropic_polyconvex_ickan  # noqa: E402
from anisotropic_pann_model_regression_claude import load_anisotropic_regression  # noqa: E402

_DEVICE = torch.device("cpu")


class PannLaw:
    """Wraps a loaded PANN model for vectorized stress/tangent evaluation."""

    def __init__(self, checkpoint_name: str, kind: str):
        path = PANN_DIR / "checkpoints" / checkpoint_name
        if kind == "polyconvex":
            model, strain_scale, energy_scale, _ = load_anisotropic_polyconvex(path, _DEVICE)
        elif kind == "free":
            model, strain_scale, energy_scale, _ = load_anisotropic_free(path, _DEVICE)
        elif kind == "ickan":
            model, strain_scale, energy_scale, _ = load_anisotropic_polyconvex_ickan(path, _DEVICE)
        elif kind == "regression":
            model, strain_scale, energy_scale, _ = load_anisotropic_regression(path, _DEVICE)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        model.eval()
        self.model = model
        self.kind = kind
        self.strain_scale = float(strain_scale)
        self.energy_scale = float(energy_scale)

    def pk2_and_tangent(self, e_voigt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """E_voigt (N,3) physical strain [E11,E22,gamma12] -> (S_voigt (N,3), CC (N,3,3))."""
        model_dtype = self.model.strain_scale.dtype
        raw = torch.as_tensor(e_voigt, dtype=model_dtype)
        normalised = (raw / self.strain_scale).clone().requires_grad_(True)

        if self.kind == "regression":
            # No energy potential: stress is a direct forward pass, and the
            # tangent is its own (not-necessarily-symmetric) Jacobian --
            # one autodiff pass per output component, no double backprop.
            stress_hat = self.model.stress(normalised)
            rows = []
            for i in range(3):
                grad_out = torch.zeros_like(stress_hat)
                grad_out[:, i] = 1.0
                row = torch.autograd.grad(stress_hat, normalised, grad_outputs=grad_out, retain_graph=True)[0]
                rows.append(row)
            cc_hat = torch.stack(rows, dim=1)
            # Trained target was stress_phys * (strain_scale/energy_scale) (train_anisotropic_regression_claude.py).
            stress_phys = stress_hat.detach().numpy() * (self.energy_scale / self.strain_scale)
            cc_phys = cc_hat.detach().numpy() * (self.energy_scale / self.strain_scale ** 2)
            return stress_phys, cc_phys

        energy = self.model.energy(normalised)
        stress_hat = torch.autograd.grad(
            energy, normalised, grad_outputs=torch.ones_like(energy), create_graph=True
        )[0]

        rows = []
        for i in range(3):
            grad_out = torch.zeros_like(stress_hat)
            grad_out[:, i] = 1.0
            row = torch.autograd.grad(
                stress_hat, normalised, grad_outputs=grad_out, retain_graph=True
            )[0]
            rows.append(row)
        cc_hat = torch.stack(rows, dim=1)  # (N,3,3), cc_hat[:,i,:] = d(stress_hat_i)/d(normalised)

        stress_phys = (stress_hat.detach().numpy()) * (self.energy_scale / self.strain_scale)
        cc_phys = (cc_hat.detach().numpy()) * (self.energy_scale / self.strain_scale ** 2)

        # symmetrize (should already be symmetric analytically; guards against roundoff)
        cc_phys = 0.5 * (cc_phys + np.swapaxes(cc_phys, 1, 2))
        return stress_phys, cc_phys


_CACHE: dict[str, PannLaw] = {}


def get_law(which: str) -> PannLaw:
    if which not in _CACHE:
        if which == "certified":
            _CACHE[which] = PannLaw("PANN_anisotropic_polyconvex_final_claude.pt", "polyconvex")
        elif which == "free":
            _CACHE[which] = PannLaw("PANN_anisotropic_free_compw000_claude.pt", "free")
        elif which == "ickan":
            _CACHE[which] = PannLaw("PANN_anisotropic_polyconvex_ickan_final_claude.pt", "ickan")
        elif which == "regression":
            _CACHE[which] = PannLaw("PANN_anisotropic_regression_claude.pt", "regression")
        else:
            raise ValueError(which)
    return _CACHE[which]


def pann_pk2_2d_vectorized(e_voigt: np.ndarray, which: str) -> tuple[np.ndarray, np.ndarray]:
    return get_law(which).pk2_and_tangent(e_voigt)


if __name__ == "__main__":
    # Quick self-check: finite-difference the tangent at a handful of random
    # strain states and compare against the autodiff tangent.
    rng = np.random.default_rng(0)
    e_test = 0.05 * rng.standard_normal((5, 3))
    for which in ("certified", "free", "ickan", "regression"):
        law = get_law(which)
        s0, cc0 = law.pk2_and_tangent(e_test)
        h = 1.0e-3 if which in ("free", "regression") else 1.0e-6  # float32 internally
        cc_fd = np.zeros_like(cc0)
        for j in range(3):
            e_plus = e_test.copy(); e_plus[:, j] += h
            e_minus = e_test.copy(); e_minus[:, j] -= h
            s_plus, _ = law.pk2_and_tangent(e_plus)
            s_minus, _ = law.pk2_and_tangent(e_minus)
            cc_fd[:, :, j] = (s_plus - s_minus) / (2 * h)
        err = np.max(np.abs(cc_fd - cc0)) / max(np.max(np.abs(cc0)), 1e-30)
        print(f"{which}: max |S| = {np.max(np.abs(s0)):.4e}, tangent rel err vs FD = {err:.3e}")
