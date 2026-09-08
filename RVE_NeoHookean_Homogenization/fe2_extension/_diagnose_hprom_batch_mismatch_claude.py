#!/usr/bin/env python3
"""Diagnose the batched-vs-serial mismatch found by
_verify_hprom_batched_claude.py's cold_start case: are the large hom_sig/
dSig_hom_dE errors confined to the small number of points whose n_iters
differ between the two paths (expected: tiny floating-point noise from a
different vmap-vs-serial call order flips a convergence check that's
sitting exactly at its threshold, giving a legitimately different but
still-converged iteration path for JUST those points), or spread across
points whose n_iters actually match (which would mean something in the
batching itself is wrong, not just convergence-threshold sensitivity)?"""
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

HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"
d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

law = m.HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR), qp_init_mode="continuation")
_ = law.evaluate_with_tangent(e_gp[0], q_prev=np.zeros(law.n_primary), step_index=1)

N = 40
rng = np.random.default_rng(1)
idx = rng.choice(e_gp.shape[0], size=N, replace=False)
E_batch = e_gp[idx]
q_prev_zero = np.zeros((N, law.n_primary))
step_index_ones = np.ones(N, dtype=int)

old = [law.evaluate_with_tangent(E_batch[i], q_prev=q_prev_zero[i], step_index=1) for i in range(N)]
new = law.evaluate_with_tangent_batch(E_batch, q_prev_batch=q_prev_zero, step_index_batch=step_index_ones)

n_iters_old = np.array([o[3] for o in old])
n_iters_new = new[3]
hom_sig_old = np.stack([o[1] for o in old])
hom_sig_new = new[1]
dsig_old = np.stack([o[6] for o in old])
dsig_new = new[6]
q_p_old = np.stack([o[2] for o in old])
q_p_new = new[2]

print(f"{'i':>3} {'n_it_old':>8} {'n_it_new':>8} {'q_p_reldiff':>12} {'hom_sig_reldiff':>16} {'dsig_reldiff':>13}")
mismatch_idx = []
for i in range(N):
    q_diff = np.max(np.abs(q_p_old[i] - q_p_new[i])) / max(np.max(np.abs(q_p_old[i])), 1e-300)
    s_diff = np.max(np.abs(hom_sig_old[i] - hom_sig_new[i])) / max(np.max(np.abs(hom_sig_old[i])), 1e-300)
    d_diff = np.max(np.abs(dsig_old[i] - dsig_new[i])) / max(np.max(np.abs(dsig_old[i])), 1e-300)
    flag = " <<<" if n_iters_old[i] != n_iters_new[i] else ""
    if flag or s_diff > 1e-8:
        mismatch_idx.append(i)
    print(f"{i:>3} {n_iters_old[i]:>8} {n_iters_new[i]:>8} {q_diff:>12.3e} {s_diff:>16.3e} {d_diff:>13.3e}{flag}")

print(f"\n[diag] {len(mismatch_idx)}/{N} points flagged (n_iters differs OR hom_sig rel err > 1e-8): {mismatch_idx}")

ok_idx = [i for i in range(N) if i not in mismatch_idx]
if ok_idx:
    s_diff_ok = max(
        np.max(np.abs(hom_sig_old[i] - hom_sig_new[i])) / max(np.max(np.abs(hom_sig_old[i])), 1e-300)
        for i in ok_idx
    )
    d_diff_ok = max(
        np.max(np.abs(dsig_old[i] - dsig_new[i])) / max(np.max(np.abs(dsig_old[i])), 1e-300)
        for i in ok_idx
    )
    print(f"[diag] among the {len(ok_idx)} NON-flagged points: worst hom_sig rel err = {s_diff_ok:.3e}, "
          f"worst dSig_hom_dE rel err = {d_diff_ok:.3e}")

print("DIAG_DONE_MARKER", flush=True)
