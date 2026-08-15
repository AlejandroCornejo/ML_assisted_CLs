#!/usr/bin/env python3
"""Rank-one (Legendre-Hadamard) convexity audit for D-HPROM-ANN's
homogenized macro map, using the exact same sampling protocol as
pann/anisotropic/evaluate_free_ellipticity_claude.py's audit of the
uncertified "free" PANN (same F/a/b distribution, same n_samples=2000
default) -- so the two audits are directly comparable.

That script's curvature check differentiates a SCALAR ENERGY twice via
autograd: d^2/dt^2 W(F+t*a(x)b) at t=0. D-HPROM-ANN (like HPROM--ANN and
the "regression" PANN tier) has no energy potential -- it produces
homogenized STRESS directly. This module derives and uses the equivalent
stress/tangent-only formula (standard in finite-strain elasticity, no
energy needed):

    d^2W/dt^2|_0 = Edot_voigt . CC . Edot_voigt  +  S_voigt . Eddot_voigt

where S=S_hom(E0), CC=dS_hom/dE|_{E0} (both from a plain (E)->(S,CC)
evaluator, energy or not), and Edot/Eddot are the first/second t-derivatives
of the Green-Lagrange strain along F(t)=F0+t*(a⊗b) (both closed-form,
since C(t)=F(t)^T F(t) is an exact quadratic in t):
    Edot  = 0.5*(F0^T A + A^T F0),   A := a⊗b
    Eddot = A^T A                     (t-independent)
converted to the project's Voigt convention [E11,E22,gamma12=2*E12] (and
correspondingly S_voigt=[S11,S22,S12], not doubled, so that S.E_voigt is a
plain dot product -- confirmed against pann_constitutive_law_claude.py's
own S_voigt = d(energy)/d(E_voigt) construction).

Before trusting this on D-HPROM-ANN (which has no energy to cross-check
against), __main__ first validates it against the ENERGY-based double-
autograd curvature on the "certified" PANN (which has both), at several
sampled (F,a,b) -- only after that passes does it run the full audit on
D-HPROM-ANN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
PANN_DIR = ROOT / "pann" / "anisotropic"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)


def sample_F_a_b(generator, dtype, stretch_log_range=1.0):
    """Exactly mirrors evaluate_free_ellipticity_claude.py's rank_one_curvature_audit
    sampling of F (diag=exp(uniform(-r,r)), off-diag=uniform(-0.4,0.4)) and unit a,b."""
    f = torch.tensor(
        ((float(torch.exp(torch.empty((), dtype=dtype).uniform_(-stretch_log_range, stretch_log_range, generator=generator))),
          float(torch.empty((), dtype=dtype).uniform_(-0.4, 0.4, generator=generator))),
         (float(torch.empty((), dtype=dtype).uniform_(-0.4, 0.4, generator=generator)),
          float(torch.exp(torch.empty((), dtype=dtype).uniform_(-stretch_log_range, stretch_log_range, generator=generator)))),),
        dtype=dtype,
    )
    a = torch.randn(2, dtype=dtype, generator=generator)
    b = torch.randn(2, dtype=dtype, generator=generator)
    a = a / torch.linalg.vector_norm(a)
    b = b / torch.linalg.vector_norm(b)
    return f, a, b


def strain_voigt_from_F(F):
    """F (2,2) numpy -> E_voigt (3,) = [E11,E22,gamma12=2*E12]."""
    C = F.T @ F
    E = 0.5 * (C - np.eye(2))
    return np.array([E[0, 0], E[1, 1], 2.0 * E[0, 1]])


def rank_one_curvature_from_S_CC(F0, a, b, S0, CC0):
    """d^2/dt^2 W(F0 + t*a⊗b) at t=0, given ALREADY-EVALUATED (S0,CC0) at F0 --
    lets a caller sampling many (a,b) directions at the SAME base state F0
    (e.g. audit_at_real_states) evaluate the law once per state instead of
    once per (state,direction) pair. See module docstring for the derivation."""
    F0 = np.asarray(F0, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    A = np.outer(a, b)

    Edot_t = 0.5 * (F0.T @ A + A.T @ F0)
    Edot_voigt = np.array([Edot_t[0, 0], Edot_t[1, 1], 2.0 * Edot_t[0, 1]])

    Eddot_t = A.T @ A
    Eddot_voigt = np.array([Eddot_t[0, 0], Eddot_t[1, 1], 2.0 * Eddot_t[0, 1]])

    S0 = np.asarray(S0, dtype=float).reshape(3)
    CC0 = np.asarray(CC0, dtype=float).reshape(3, 3)

    return float(Edot_voigt @ CC0 @ Edot_voigt + S0 @ Eddot_voigt)


def rank_one_curvature_from_stress_tangent(F0, a, b, eval_S_and_CC):
    """d^2/dt^2 W(F0 + t*a⊗b) at t=0, using only (S,CC)=eval_S_and_CC(E_voigt) --
    no energy potential required. See module docstring for the derivation."""
    E0_voigt = strain_voigt_from_F(np.asarray(F0, dtype=float))
    S0, CC0 = eval_S_and_CC(E0_voigt)
    return rank_one_curvature_from_S_CC(F0, a, b, S0, CC0)


def _validate_against_energy_autograd(n_checks=8, seed=1):
    """Cross-checks rank_one_curvature_from_stress_tangent against the
    energy-based double-autograd formula (evaluate_free_ellipticity_claude
    .py's own method) on the 'certified' PANN, which has both an energy
    and a (S,CC) evaluator -- the only tier where both are available."""
    import pann_constitutive_law_claude as pann_law
    from anisotropic_pann_model import load_anisotropic_polyconvex, c_to_strain

    law = pann_law.get_law("certified")

    def eval_S_and_CC(E_voigt):
        S, CC = law.pk2_and_tangent(E_voigt.reshape(1, 3))
        return S[0], CC[0]

    model, strain_scale, energy_scale, _ck = load_anisotropic_polyconvex(
        PANN_DIR / "checkpoints" / "PANN_anisotropic_polyconvex_final_claude.pt", torch.device("cpu"),
    )
    model = model.double()
    dtype = torch.float64

    def energy_from_F(F_t):
        c = F_t.transpose(-1, -2) @ F_t
        physical_strain = c_to_strain(c.reshape(-1, 2, 2))
        return model.energy(physical_strain / strain_scale) * energy_scale

    generator = torch.Generator().manual_seed(seed)
    print(f"[validate] comparing stress/tangent curvature formula vs. energy-autograd, "
          f"{n_checks} random (F,a,b) samples on 'certified' ...")
    worst_rel = 0.0
    n_ok = 0
    for i in range(n_checks):
        f, a, b = sample_F_a_b(generator, dtype)
        t = torch.zeros((), dtype=dtype, requires_grad=True)
        f_t = f + t * torch.outer(a, b)
        if torch.det(f_t).detach() <= 0.0:
            continue
        value = energy_from_F(f_t.reshape(1, 2, 2)).sum()
        first = torch.autograd.grad(value, t, create_graph=True)[0]
        second = torch.autograd.grad(first, t)[0]
        curvature_energy = float(second.detach())

        curvature_formula = rank_one_curvature_from_stress_tangent(
            f.numpy(), a.numpy(), b.numpy(), eval_S_and_CC,
        )
        rel = abs(curvature_formula - curvature_energy) / max(abs(curvature_energy), 1e-8)
        worst_rel = max(worst_rel, rel)
        n_ok += 1
        print(f"    sample {i}: energy-autograd={curvature_energy:.6e}, "
              f"stress/tangent-formula={curvature_formula:.6e}, rel_err={rel:.3e}")

    ok = worst_rel < 1e-4
    print(f"[validate] worst relative error over {n_ok} valid samples: {worst_rel:.3e} "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def run_audit(eval_S_and_CC, label, n_samples=2000, stretch_log_range=1.0, seed=20260828):
    """Same protocol/output shape as evaluate_free_ellipticity_claude.py's
    rank_one_curvature_audit, using the stress/tangent-only formula."""
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    curvatures = []
    failures = []
    for i in range(n_samples):
        f, a, b = sample_F_a_b(generator, dtype, stretch_log_range=stretch_log_range)
        if torch.det(f).item() <= 0.0:
            continue
        curvature = rank_one_curvature_from_stress_tangent(f.numpy(), a.numpy(), b.numpy(), eval_S_and_CC)
        curvatures.append(curvature)
        if curvature < 0.0:
            failures.append({"F": f.tolist(), "a": a.tolist(), "b": b.tolist(), "curvature": curvature})
        if (i + 1) % 200 == 0:
            print(f"    [{label}] {i + 1}/{n_samples} samples done, "
                  f"{len(failures)} violations so far, {len(curvatures)} valid")

    curvatures_arr = np.asarray(curvatures)
    n_valid = len(curvatures)
    result = {
        "n_requested": n_samples,
        "n_valid_paths": n_valid,
        "n_violations": int((curvatures_arr < 0.0).sum()) if n_valid else 0,
        "fraction_violations": float((curvatures_arr < 0.0).mean()) if n_valid else float("nan"),
        "minimum_curvature": float(curvatures_arr.min()) if n_valid else float("nan"),
        "median_curvature": float(np.median(curvatures_arr)) if n_valid else float("nan"),
        "example_violations": failures[:5],
    }
    print(f"\n[{label}] n_valid={result['n_valid_paths']}, n_violations={result['n_violations']} "
          f"({result['fraction_violations']:.3%}), min_curvature={result['minimum_curvature']:.6e}, "
          f"median_curvature={result['median_curvature']:.6e}")
    return result


if __name__ == "__main__":
    ok = _validate_against_energy_autograd()
    if not ok:
        print("[main] ABORTING: stress/tangent curvature formula did not match the energy-based "
              "reference closely enough; not safe to trust on D-HPROM-ANN (no energy to cross-check).")
        sys.exit(1)

    print("\n[main] formula validated -- running the full rank-one audit on D-HPROM-ANN ...")
    from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw

    law = DHpromAnnDirectLaw()

    def eval_dhprom(E_voigt):
        _eps, sig, _dEps, dSig = law.evaluate_with_tangent(E_voigt)
        return sig, dSig

    run_audit(eval_dhprom, label="D-HPROM-ANN", n_samples=2000, stretch_log_range=1.0, seed=20260828)
