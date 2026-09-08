#!/usr/bin/env python3
"""Rank-one convexity check for Regression/Free (PANN tiers 1/2) at the
REAL macro Gauss-point states visited across the Cruciform delta sweeps
already run (sweep_cruciform_delta_uncertified_claude.py's magnitude
scan, delta in [1.2,4.8]; sweep_cruciform_asymmetric_uncertified_claude.py's
direction scan, delta_x=2.4 fixed, delta_y in [-2.4,2.4]) -- both found
clean convergence throughout, but that only shows the DISPLACEMENT-
CONTROLLED macro solve never got blocked; it says nothing about whether
the material response itself loses local stability (rank-one convexity)
somewhere along the way. Reuses this project's OWN validated, energy-
free curvature formula (rank_one_convexity_check_claude.py's
rank_one_curvature_from_S_CC), the exact method the paper already uses
for this same audit on Cook's own visited states (Table 8,
tab:cook-rankone) -- same protocol, now pointed at Cruciform's states
instead, for a directly comparable result.

Re-runs each (delta_x,delta_y) configuration once more (cheap, PANN-only)
with save_npz=True under throwaway keys so the real final-state e_gp is
available to load; deletes those throwaway npz files at the end."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cruciform_fe2_claude as rc  # noqa: E402
from rank_one_check_at_cook_states_claude import audit_at_real_states, deformation_gradients_from_strains  # noqa: E402
import pann_constitutive_law_claude as pann_law  # noqa: E402

N_BODY, N_ARM_LEN = 6, 4
N_STEPS = 20
CONFIGS = (
    # (delta_x, delta_y) -- magnitude scan (equibiaxial) + asymmetric scan
    (1.2, 1.2), (1.8, 1.8), (2.4, 2.4), (3.0, 3.0), (3.6, 3.6), (4.2, 4.2), (4.8, 4.8),
    (2.4, 1.8), (2.4, 0.6), (2.4, 0.0), (2.4, -0.6), (2.4, -1.2), (2.4, -1.8), (2.4, -2.4),
)
TIERS = {"pann_regression": "regression", "pann_free": "free"}


def collect_e_gp(which):
    # Register a SEPARATE throwaway key pointing at the identical, already-
    # registered material function -- guarantees the output filename below
    # can never collide with (and therefore can never overwrite) the
    # established cruciform_results_{which}_claude.npz this exact "which"
    # string already names (the paper's own Table 8 data).
    throwaway_key = f"{which}_stabcheck_claude"
    assert throwaway_key not in rc.MATERIAL_FUNCS
    rc.MATERIAL_FUNCS[throwaway_key] = rc.MATERIAL_FUNCS[which]

    all_e = []
    path = HERE / f"cruciform_results_{throwaway_key}_claude.npz"
    for dx, dy in CONFIGS:
        rc.run_newton_fe2_cruciform(
            throwaway_key, n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
            delta_x_final=dx, delta_y_final=dy, verbose=False, use_line_search=True, save_npz=True,
        )
        d = np.load(path)
        all_e.append(d["e_gp"].copy())
        d.close()
    path.unlink()  # throwaway file, never collides with the established which-named one
    return np.concatenate(all_e, axis=0)


def main():
    results = {}
    for which, tier in TIERS.items():
        print(f"\n=== {which} ({len(CONFIGS)} configs x 600 GP = {len(CONFIGS) * 600} states) ===", flush=True)
        e_gp = collect_e_gp(which)
        F_all = deformation_gradients_from_strains(e_gp)
        law = pann_law.get_law(tier)

        def eval_pann(E_voigt, _law=law):
            S, CC = _law.pk2_and_tangent(E_voigt.reshape(1, 3))
            return S[0], CC[0]

        results[which] = audit_at_real_states(F_all, eval_pann, which)

    print("\n\n=== SUMMARY: rank-one check across all Cruciform sweep states ===", flush=True)
    for name, r in results.items():
        print(f"{name:20s} n_gp={r['n_gp']:6d}  n_total={r['n_total']:7d}  "
              f"violations={r['n_violations']:6d} ({r['fraction']:.4%})  worst={r['worst_curvature']:.4e}",
              flush=True)
    print("RANKONE_CRUCIFORM_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
