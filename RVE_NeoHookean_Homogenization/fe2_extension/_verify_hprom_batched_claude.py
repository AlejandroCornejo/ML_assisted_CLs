#!/usr/bin/env python3
"""Verify evaluate_with_tangent_batch (batches the decoder-Jacobian and
weighted-Hessian torch calls across points at each shared Newton-iteration
depth, via _drive_evaluate_impl_gens_batch) reproduces a per-point loop of
evaluate_with_tangent (unbatched, exact original control flow) to
floating-point precision, on real macro strains from the n_body=6
continuation run -- both at cold start (step_index=1 for all points) AND
at a MIXED warm-start (some points step_index=2 with a real, nonzero
q_prev), since that mixed-iteration-depth case is exactly what exercises
the active-point-mask logic and is what the real continuation wrapper
actually produces call to call."""
from __future__ import annotations

import sys
import time
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
print(f"[verify] loaded {e_gp.shape[0]} real macro strains", flush=True)

law = m.HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR), qp_init_mode="continuation")
_ = law.evaluate_with_tangent(e_gp[0], q_prev=np.zeros(law.n_primary), step_index=1)  # warm-up


def compare(name, E_batch, q_prev_batch, step_index_batch):
    t0 = time.perf_counter()
    old = [
        law.evaluate_with_tangent(E_batch[i], q_prev=q_prev_batch[i], step_index=int(step_index_batch[i]))
        for i in range(len(E_batch))
    ]
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    new = law.evaluate_with_tangent_batch(E_batch, q_prev_batch=q_prev_batch, step_index_batch=step_index_batch)
    t_new = time.perf_counter() - t0

    print(f"\n[verify:{name}] OLD (per-point loop): {t_old:.3f}s, {t_old / len(E_batch) * 1000:.3f} ms/point",
          flush=True)
    print(f"[verify:{name}] NEW (batched): {t_new:.3f}s, {t_new / len(E_batch) * 1000:.3f} ms/point", flush=True)
    print(f"[verify:{name}] speedup: {t_old / max(t_new, 1e-9):.2f}x", flush=True)

    field_names = ["hom_eps", "hom_sig", "q_p", "n_iters", "converged",
                   "dEps_hom_dE", "dSig_hom_dE", "hom_eps_check", "hom_sig_check"]
    ok = True
    for fi, fname in enumerate(field_names):
        old_stack = np.stack([np.asarray(o[fi]).astype(float) for o in old])
        new_stack = np.asarray(new[fi]).astype(float)
        if old_stack.shape != new_stack.shape:
            print(f"[verify:{name}] SHAPE MISMATCH {fname}: old {old_stack.shape} vs new {new_stack.shape}",
                  flush=True)
            ok = False
            continue
        diff = np.max(np.abs(old_stack - new_stack))
        rel = diff / max(np.max(np.abs(old_stack)), 1e-300)
        print(f"[verify:{name}] max abs diff {fname}: {diff:.3e} (rel {rel:.3e})", flush=True)
        if rel > 1e-8 and diff > 1e-10:
            ok = False
    return ok


N = 40
rng = np.random.default_rng(1)
idx = rng.choice(e_gp.shape[0], size=N, replace=False)
E_batch = e_gp[idx]

# --- Case 1: all cold start ---
q_prev_zero = np.zeros((N, law.n_primary))
step_index_ones = np.ones(N, dtype=int)
ok1 = compare("cold_start", E_batch, q_prev_zero, step_index_ones)

# --- Case 2: mixed warm start -- half the points get a REAL nonzero
# q_prev (each point's own cold-start converged q_p, used as a stand-in
# warm start) with step_index=2 (continuation branch), the other half
# stay cold (step_index=1) -- this forces generators to run genuinely
# different numbers of iterations AND different code branches within the
# SAME batched call, exactly like the real continuation wrapper produces
# call to call once some points have history and others don't yet.
q_prev_mixed = np.zeros((N, law.n_primary))
step_index_mixed = np.ones(N, dtype=int)
half = N // 2
for j in range(half):
    _, _, q_p_conv, _, conv = law.evaluate(E_batch[j], q_prev=np.zeros(law.n_primary), step_index=1)
    if conv:
        q_prev_mixed[j] = q_p_conv
        step_index_mixed[j] = 2
ok2 = compare("mixed_warm_start", E_batch, q_prev_mixed, step_index_mixed)

print("\nVERIFY_PASS" if (ok1 and ok2) else "\nVERIFY_FAIL", flush=True)
