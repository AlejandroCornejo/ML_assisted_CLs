#!/usr/bin/env python3
"""Same "Table-8-style" rank-one check (at each model's own real
Cook-visited nx=8 states) as rank_one_check_at_cook_states_claude.py,
for the three PANN tiers not covered there: certified, ickan, free --
so all six laws end up checked with the identical nx=8 mesh and protocol.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rank_one_check_at_cook_states_claude import audit_at_real_states, deformation_gradients_from_strains  # noqa: E402

if __name__ == "__main__":
    import pann_constitutive_law_claude as pann_law

    results = {}
    files = {
        "certified": HERE / "cook_results_certified_nx8_claude.npz",
        "ickan": HERE / "cook_results_ickan_nx8_claude.npz",
        "free": HERE / "cook_results_pann_free_claude.npz",
    }
    for tier, path in files.items():
        print(f"\n=== {tier} (nx=8) ===")
        d = np.load(path)
        e_gp = d["e_gp"]
        F_all = deformation_gradients_from_strains(e_gp)
        law = pann_law.get_law(tier)

        def eval_pann(E_voigt, _law=law):
            S, CC = _law.pk2_and_tangent(E_voigt.reshape(1, 3))
            return S[0], CC[0]

        results[f"{tier}_nx8"] = audit_at_real_states(F_all, eval_pann, f"{tier} (nx=8)")

    print("\n\n=== SUMMARY: rank-one check at real Cook-visited nx=8 states (PANN tiers) ===")
    for name, r in results.items():
        print(f"{name:24s} n_gp={r['n_gp']:5d}  n_total={r['n_total']:6d}  "
              f"violations={r['n_violations']:5d} ({r['fraction']:.3%})  worst={r['worst_curvature']:.4e}")
