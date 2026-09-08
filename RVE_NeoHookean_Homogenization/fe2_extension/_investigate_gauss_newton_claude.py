#!/usr/bin/env python3
"""Investigate whether HPROM-ANN's inner Newton correction actually needs
the weighted-decoder-Hessian curvature term (include_manifold_curvature=
True, 22.0% of per-point cost) or whether a cheaper Gauss-Newton
approximation (include_manifold_curvature=False, K_r = K_std only) is
sufficient -- on ALL 600 real macro strains from the n_body=6 continuation
run's own final (most-deformed, historically-hardest) step, at COLD START
(no warm-starting help, a harder test than actual production use).

Reports, for each variant: convergence rate, iteration-count distribution,
wall-clock, and (where both converge) agreement on the final q_p/hom_eps/
hom_sig -- dropping the curvature term only changes the Newton UPDATE
direction, not the equilibrium equation being solved, so a converged
answer should match the full-Newton one closely regardless of which path
got there."""
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
print(f"[gn] loaded {e_gp.shape[0]} real macro strains "
      f"(n_body=6 continuation run's own FINAL step -- most-deformed states)", flush=True)

results = {}
for label, curvature in [("newton", True), ("gauss_newton", False)]:
    law = m.HpromAnnIterativeLawFloat64(
        hprom_ann_dir=str(HPROMANN_DIR), qp_init_mode="continuation",
        include_manifold_curvature=curvature,
    )
    q_prev_zero = np.zeros((e_gp.shape[0], law.n_primary))
    step_index_ones = np.ones(e_gp.shape[0], dtype=int)

    t0 = time.perf_counter()
    out = law.evaluate_with_tangent_batch(e_gp, q_prev_batch=q_prev_zero, step_index_batch=step_index_ones)
    wall = time.perf_counter() - t0

    hom_eps, hom_sig, q_p, n_iters, converged = out[0], out[1], out[2], out[3], out[4]
    results[label] = dict(hom_eps=hom_eps, hom_sig=hom_sig, q_p=q_p, n_iters=n_iters,
                           converged=converged, wall=wall)
    n_conv = int(np.sum(converged))
    print(f"\n[gn:{label}] wall={wall:.3f}s ({wall / e_gp.shape[0] * 1000:.3f} ms/point), "
          f"converged={n_conv}/{e_gp.shape[0]}", flush=True)
    print(f"[gn:{label}] n_iters: min={n_iters.min()}, max={n_iters.max()}, mean={n_iters.mean():.2f}, "
          f"median={np.median(n_iters):.1f}", flush=True)
    hist = {int(k): int(v) for k, v in zip(*np.unique(n_iters, return_counts=True))}
    print(f"[gn:{label}] n_iters histogram: {hist}", flush=True)
    if n_conv < e_gp.shape[0]:
        bad_idx = np.where(~converged)[0]
        print(f"[gn:{label}] NON-CONVERGED point indices: {bad_idx.tolist()}", flush=True)

n_ref = results["newton"]
g = results["gauss_newton"]
both_converged = n_ref["converged"] & g["converged"]
print(f"\n[gn] both converged: {int(np.sum(both_converged))}/{e_gp.shape[0]}", flush=True)

for field in ["hom_eps", "hom_sig", "q_p"]:
    a = n_ref[field][both_converged]
    b = g[field][both_converged]
    diff = np.max(np.abs(a - b)) if a.size else 0.0
    rel = diff / max(np.max(np.abs(a)), 1e-300) if a.size else 0.0
    print(f"[gn] where both converged, max abs diff {field}: {diff:.3e} (rel {rel:.3e})", flush=True)

speedup = n_ref["wall"] / max(g["wall"], 1e-9)
print(f"\n[gn] wall-clock: newton={n_ref['wall']:.3f}s, gauss_newton={g['wall']:.3f}s "
      f"({speedup:.2f}x)", flush=True)
print(f"[gn] mean iters: newton={n_ref['n_iters'].mean():.2f}, gauss_newton={g['n_iters'].mean():.2f}",
      flush=True)
print("GN_DONE_MARKER", flush=True)
