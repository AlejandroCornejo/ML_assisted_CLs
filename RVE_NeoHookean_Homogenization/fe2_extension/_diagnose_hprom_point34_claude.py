#!/usr/bin/env python3
"""Deep-dive point 34 (the one showing zero improvement from the
K_total_support fix, 17.1% dSig_hom_dE relative error, well-conditioned
K_r): does the SERIAL path's own cache get populated (converged cleanly)
for this exact point, and does the BATCHED path's cache ALSO get
populated -- and if both do, do the cached K_r/J_manifold/w_res_iter/
K_total_support themselves actually agree?"""
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
E_point = e_gp[idx[34]]
q_prev = np.zeros(law.n_primary)

# --- SERIAL cache (via the thin driver _evaluate_impl over _evaluate_impl_gen) ---
cache_serial = {}
result_serial = law._evaluate_impl(E_point, q_prev=q_prev, step_index=1, _cache_out=cache_serial)
print(f"[diag] serial: converged_cleanly (cache populated) = {bool(cache_serial)}")
print(f"[diag] serial: n_iters={result_serial[3]}, converged={result_serial[4]}")

# --- BATCHED cache (via _drive_evaluate_impl_gens_batch on a batch of 1) ---
per_point = law._drive_evaluate_impl_gens_batch(
    E_point.reshape(1, 3), q_prev.reshape(1, -1), np.array([1]),
)
cache_batched = per_point[0][5]
print(f"[diag] batched (n=1): converged_cleanly (cache populated) = {bool(cache_batched)}")
print(f"[diag] batched (n=1): n_iters={per_point[0][3]}, converged={per_point[0][4]}")

if cache_serial and cache_batched:
    for key in ["q_p", "J_manifold", "K_r", "w_res_iter", "u_eq_curr", "r_full", "K_total_support"]:
        a, b = np.asarray(cache_serial[key]), np.asarray(cache_batched[key])
        diff = np.max(np.abs(a - b))
        rel = diff / max(np.max(np.abs(a)), 1e-300)
        print(f"[diag] cache[{key}]: max abs diff = {diff:.3e} (rel {rel:.3e})")

# Now also compare against a batch of 40 (the real repro case) to see if
# batch SIZE itself matters (n=1 batch vs n=40 batch) -- if n=1 batch
# matches serial exactly but n=40 doesn't, that would point at cross-point
# contamination specific to multi-point batches.
E_batch40 = e_gp[idx]
q_prev40 = np.zeros((N, law.n_primary))
per_point_40 = law._drive_evaluate_impl_gens_batch(E_batch40, q_prev40, np.ones(N, dtype=int))
cache_batched_40 = per_point_40[34][5]
print(f"\n[diag] batched (n=40), point 34: converged_cleanly = {bool(cache_batched_40)}")
print(f"[diag] batched (n=40), point 34: n_iters={per_point_40[34][3]}, converged={per_point_40[34][4]}")
if cache_serial and cache_batched_40:
    for key in ["q_p", "J_manifold", "K_r", "w_res_iter", "u_eq_curr", "r_full", "K_total_support"]:
        a, b = np.asarray(cache_serial[key]), np.asarray(cache_batched_40[key])
        diff = np.max(np.abs(a - b))
        rel = diff / max(np.max(np.abs(a)), 1e-300)
        print(f"[diag] cache40[{key}]: max abs diff = {diff:.3e} (rel {rel:.3e})")

print("DIAG_DONE_MARKER", flush=True)
