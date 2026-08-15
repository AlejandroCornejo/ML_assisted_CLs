#!/usr/bin/env python3
"""Consolidated, reusable evaluation battery for a single smallstrain
checkpoint (ICNN or ICKAN): held-out stage-10 trajectory (full E/S rel L2),
Cook step-1 exact-Gauss-point comparison against the true FOM, rank-one
admissibility audit, and a full 20-step Cook (nx=8) structural run. Prints
one clean summary block; used identically for every weight in the sweep so
results are directly comparable and nothing is retyped/re-derived per run.

Usage: python3 eval_smallstrain_checkpoint_claude.py <kind: polyconvex|ickan> <checkpoint_filename> <label>
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
CORE_DIR = ROOT / "core"
for p in (str(CORE_DIR), str(COOK_DIR), str(PANN_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

DATA_PATH = PANN_DIR.parent / "data" / "alltraj_stage10_direct_energy.npz"


def relative_l2(prediction, reference):
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def load_model(kind, checkpoint_name):
    if kind == "polyconvex":
        from anisotropic_pann_model import load_anisotropic_polyconvex
        return load_anisotropic_polyconvex(PANN_DIR / "checkpoints" / checkpoint_name, torch.device("cpu"))
    elif kind == "ickan":
        from anisotropic_pann_model_ickan_claude import load_anisotropic_polyconvex_ickan
        return load_anisotropic_polyconvex_ickan(PANN_DIR / "checkpoints" / checkpoint_name, torch.device("cpu"))
    raise ValueError(kind)


def predict(model, strain, strain_scale, energy_scale, dtype, batch_size=512):
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            stop = min(start + batch_size, len(strain))
            x = torch.as_tensor(strain[start:stop] / strain_scale, dtype=dtype).requires_grad_(True)
            energy, stress = model.energy_and_stress(x, create_graph=False)
            energies.append((energy.detach().numpy() * energy_scale).reshape(-1))
            stresses.append(stress.detach().numpy() * (energy_scale / strain_scale))
    return np.concatenate(energies), np.concatenate(stresses, axis=0)


def main(kind: str, checkpoint_name: str, label: str):
    model, strain_scale, energy_scale, _ = load_model(kind, checkpoint_name)
    model.eval()
    dtype = next(model.parameters()).dtype

    with np.load(DATA_PATH) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float64)
        energy_true = np.asarray(data["stage10_energy"], dtype=np.float64)
        stress_true = np.asarray(data["stage10_stress"], dtype=np.float64)
    e_pred, s_pred = predict(model, strain, strain_scale, energy_scale, dtype)
    full_e = relative_l2(e_pred, energy_true)
    full_s = relative_l2(s_pred, stress_true)

    sample_path = Path("/tmp/claude-1000/-home-kratos-ML-assisted-CLs-clean/4a3da423-8dfc-40e5-a7fa-8fa40d3c8b2f/scratchpad/sample_strains.npy")
    true_path = Path("/tmp/claude-1000/-home-kratos-ML-assisted-CLs-clean/4a3da423-8dfc-40e5-a7fa-8fa40d3c8b2f/scratchpad/true_fom_sample_SCC.npz")
    sample = np.load(sample_path)
    S_true = np.load(true_path)["S_true"]
    with torch.enable_grad():
        x = torch.as_tensor(sample / strain_scale, dtype=dtype).requires_grad_(True)
        _energy, stress = model.energy_and_stress(x, create_graph=False)
    S = stress.detach().numpy() * (energy_scale / strain_scale)
    rels = [np.linalg.norm(S[i] - S_true[i]) / max(np.linalg.norm(S_true[i]), 1e-6) for i in range(len(sample))]
    cook_mean, cook_lo, cook_hi = float(np.mean(rels)), float(min(rels)), float(max(rels))

    from rank_one_convexity_check_claude import run_audit

    def eval_fn(E_voigt):
        with torch.enable_grad():
            x1 = torch.as_tensor(E_voigt.reshape(1, 3) / strain_scale, dtype=dtype).requires_grad_(True)
            energy, stress = model.energy_and_stress(x1, create_graph=True)
            rows = []
            for i in range(3):
                grad_out = torch.zeros_like(stress)
                grad_out[:, i] = 1.0
                row = torch.autograd.grad(stress, x1, grad_outputs=grad_out, retain_graph=True)[0]
                rows.append(row.detach())
            cc_hat = torch.stack(rows, dim=1)[0].numpy()
        s_phys = stress.detach().numpy()[0] * (energy_scale / strain_scale)
        cc_phys = cc_hat * (energy_scale / strain_scale ** 2)
        cc_phys = 0.5 * (cc_phys + cc_phys.T)  # guard against roundoff asymmetry, same as pann_constitutive_law_claude.py
        return s_phys, cc_phys

    audit = run_audit(eval_fn, label=label, n_samples=2000, stretch_log_range=1.0, seed=20260828)

    import run_cook_hprom_ann_claude as cook_driver

    def cook_material_func(e_voigt, young=None, poisson=None):
        # Batched over all N Gauss points in one call (unlike eval_fn above,
        # which the rank-one audit calls once per single state) -- matches
        # pann_constitutive_law_claude.py's own PannLaw.pk2_and_tangent
        # pattern. The unbatched, per-row version cost 751s here (mostly
        # Python-loop/autograd-call overhead) vs. ~2.4s batched.
        n = e_voigt.shape[0]
        with torch.enable_grad():
            x = torch.as_tensor(e_voigt / strain_scale, dtype=dtype).requires_grad_(True)
            _energy, stress = model.energy_and_stress(x, create_graph=True)
            rows = []
            for i in range(3):
                grad_out = torch.zeros_like(stress)
                grad_out[:, i] = 1.0
                row = torch.autograd.grad(stress, x, grad_outputs=grad_out, retain_graph=True)[0]
                rows.append(row.detach())
            cc_hat = torch.stack(rows, dim=1).numpy()  # (n,3,3)
        s_phys = stress.detach().numpy() * (energy_scale / strain_scale)
        cc_phys = cc_hat * (energy_scale / strain_scale ** 2)
        cc_phys = 0.5 * (cc_phys + np.swapaxes(cc_phys, 1, 2))
        return s_phys, cc_phys

    which_key = f"smallstrain_eval_{label}"
    cook_driver.MATERIAL_FUNCS[which_key] = cook_material_func
    cook_res = cook_driver.run_newton_fe2(which_key, nx=8, ny=8, verbose=False, save_npz=False)

    print(f"\n=== SUMMARY [{label}] ===")
    print(f"held-out full trajectory: E={full_e:.4%}  S={full_s:.4%}")
    print(f"Cook step-1 exact points: mean={cook_mean:.4%}  range=({cook_lo:.4%},{cook_hi:.4%})")
    print(f"rank-one admissibility (2000 samples): violations={audit['n_violations']} ({audit['fraction_violations']:.3%})")
    print(f"Cook full 20-step: fully_converged={cook_res['fully_converged']}  tip_uy={cook_res['tip_uy_range']}  "
          f"wall={cook_res['wall_time']:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
