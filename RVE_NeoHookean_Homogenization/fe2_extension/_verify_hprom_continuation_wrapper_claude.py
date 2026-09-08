#!/usr/bin/env python3
"""Verify the updated HpromIterativeContinuationWrapper (now driving
evaluate_with_tangent_batch) reproduces a reference wrapper using the
ORIGINAL per-point evaluate_with_tangent loop, across a SEQUENCE of calls
(so the cross-call q_prev_by_point warm-start bookkeeping is genuinely
exercised, not just a single call), on real macro strains."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import hprom_ann_iterative_law_float64_claude as m  # noqa: E402
from run_cruciform_fe2_claude import HpromIterativeContinuationWrapper  # noqa: E402

HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"
d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

N = 30
rng = np.random.default_rng(2)
base_idx = rng.choice(e_gp.shape[0], size=N, replace=False)
E_base = e_gp[base_idx]


class ReferenceWrapper:
    """Exact copy of the ORIGINAL (pre-batching) HpromIterativeContinuationWrapper
    logic -- per-point evaluate_with_tangent loop, same warm-start/commit
    rule -- used here purely as the ground truth to compare against."""

    def __init__(self, law):
        self.law = law
        self.q_prev_by_point = {}

    def __call__(self, e_voigt):
        e_voigt = np.asarray(e_voigt, dtype=float)
        n = e_voigt.shape[0]
        S = np.zeros((n, 3), dtype=float)
        CC = np.zeros((n, 3, 3), dtype=float)
        n_nonconverged = 0
        for i in range(n):
            q_prev = self.q_prev_by_point.get(i)
            if q_prev is None:
                q_prev = np.zeros(self.law.n_primary, dtype=float)
                step_index = 1
            else:
                step_index = 2
            _, S[i], q_p_new, _n_it, converged, _, CC[i], _, _ = self.law.evaluate_with_tangent(
                e_voigt[i], q_prev=q_prev, step_index=step_index,
            )
            if converged:
                self.q_prev_by_point[i] = q_p_new
            else:
                n_nonconverged += 1
        return S, CC


law_ref = m.HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR), qp_init_mode="continuation")
law_new = m.HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR), qp_init_mode="continuation")

ref_wrapper = ReferenceWrapper(law_ref)
new_wrapper = HpromIterativeContinuationWrapper(law_new)

N_CALLS = 6
ok = True
for call_idx in range(N_CALLS):
    # Small evolving perturbation per call, standing in for successive
    # macro Newton iterations/load steps visiting nearby (not identical)
    # strain states -- exercises real warm-starting, not a degenerate
    # "same E every time" case.
    E_call = E_base * (1.0 + 0.03 * call_idx) + 0.01 * call_idx * np.array([1.0, -0.5, 0.3])

    S_ref, CC_ref = ref_wrapper(E_call)
    S_new, CC_new = new_wrapper(E_call)

    s_err = np.max(np.abs(S_ref - S_new))
    s_rel = s_err / max(np.max(np.abs(S_ref)), 1e-300)
    cc_err = np.max(np.abs(CC_ref - CC_new))
    cc_rel = cc_err / max(np.max(np.abs(CC_ref)), 1e-300)

    n_prev_ref = len(ref_wrapper.q_prev_by_point)
    n_prev_new = len(new_wrapper.q_prev_by_point)
    q_prev_err = 0.0
    for i in ref_wrapper.q_prev_by_point:
        if i in new_wrapper.q_prev_by_point:
            q_prev_err = max(q_prev_err, np.max(np.abs(
                ref_wrapper.q_prev_by_point[i] - new_wrapper.q_prev_by_point[i]
            )))

    print(f"[verify] call {call_idx}: S rel err={s_rel:.3e}, CC rel err={cc_rel:.3e}, "
          f"n_prev(ref/new)={n_prev_ref}/{n_prev_new}, q_prev_by_point max abs diff={q_prev_err:.3e}",
          flush=True)

    if s_rel > 1e-8 or cc_rel > 1e-8 or n_prev_ref != n_prev_new or q_prev_err > 1e-8:
        ok = False

print("VERIFY_PASS" if ok else "VERIFY_FAIL", flush=True)
