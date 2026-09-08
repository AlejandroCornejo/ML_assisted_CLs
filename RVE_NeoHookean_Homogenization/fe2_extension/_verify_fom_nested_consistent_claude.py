#!/usr/bin/env python3
"""Verify fom_nested_consistent_law_claude.py's two corrections:

1. VALUE check: its reaction-force stress at a real trajectory state
   should match DirectStressGenerator's own independent computation at
   that SAME (E, U) pair (pann/direct_energy/reaction_force_direct_
   stress.py, already validated this project against the paper's own
   training-data ground truth). Note: DirectStressGenerator uses the
   trajectory's OWN pre-computed U (from an incremental multi-step
   solve); fom_nested_consistent_law_claude.py does a completely FRESH
   2-waypoint solve from E=0 straight to this E. These are two
   DIFFERENT solve paths to (should be) the SAME unique equilibrium --
   agreement here checks both the reaction-force formula AND that a
   direct-ramp solve reaches the same state an incremental one does, not
   just that the formula is self-consistent.

2. TANGENT check: the analytic (implicit-function-theorem) CC should
   match a direct central finite difference of THIS SAME reaction-force
   stress formula (not the old naive-volume-average one) at E+h/E-h,
   each a fresh full nonlinear solve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

sys.path.insert(0, str(HERE.parent / "pann" / "direct_energy"))

import fom_nested_consistent_law_claude as m  # noqa: E402
from reaction_force_direct_stress import DirectStressGenerator  # noqa: E402

TRAJ_DIR = HERE.parent / "trajectories" / "stage_1_training_set_fom" / "trajectory_1"
U_all = np.load(TRAJ_DIR / "trajectory_1_U.npy")
e_all = np.load(TRAJ_DIR / "trajectory_1_applied_strain.npy")

IDX = 20
E_test = e_all[IDX].copy()
U_test = U_all[IDX].copy()
print(f"[verify] test state: idx={IDX}, E={E_test}", flush=True)

# --- Value check against DirectStressGenerator ---
gen = DirectStressGenerator()
S_ref = gen.direct_stress_history(U_test.reshape(1, -1), E_test.reshape(1, 3))[0]
gen.close()
print(f"[verify] DirectStressGenerator (existing U): S_ref={S_ref}", flush=True)

S_mine, CC_mine = m.evaluate_with_tangent(E_test, verbose=False)
print(f"[verify] fom_nested_consistent (fresh solve): S_mine={S_mine}", flush=True)

s_err = np.max(np.abs(S_mine - S_ref))
s_rel = s_err / max(np.max(np.abs(S_ref)), 1e-300)
print(f"[verify] VALUE max abs diff={s_err:.3e}, rel={s_rel:.3e}", flush=True)

# --- Tangent check: FD of the SAME reaction-force stress formula ---
h = 1.0e-4
CC_fd = np.zeros((3, 3), dtype=float)
for k in range(3):
    Ep, Em = E_test.copy(), E_test.copy()
    Ep[k] += h
    Em[k] -= h
    Sp, _ = m.evaluate_with_tangent(Ep, verbose=False)
    Sm, _ = m.evaluate_with_tangent(Em, verbose=False)
    CC_fd[:, k] = (Sp - Sm) / (2.0 * h)
    print(f"[verify] FD tangent column {k} done", flush=True)

cc_err = np.max(np.abs(CC_mine - CC_fd))
cc_rel = cc_err / max(np.max(np.abs(CC_fd)), 1e-300)
print(f"[verify] analytic CC=\n{CC_mine}", flush=True)
print(f"[verify] FD CC=\n{CC_fd}", flush=True)
print(f"[verify] TANGENT max abs diff={cc_err:.3e}, rel={cc_rel:.3e}", flush=True)

ok = s_rel < 1.0e-2 and cc_rel < 1.0e-2
print("VERIFY_PASS" if ok else "VERIFY_FAIL", flush=True)
