#!/usr/bin/env python3
"""Profile HpromAnnIterativeLawFloat64.evaluate_with_tangent on real macro
strains (from the already-converged, non-diverging n_body=6 continuation
run's own final-step e_gp), breaking down wall-clock time by sub-piece via
manual instrumentation -- cProfile alone can't cleanly separate "decoder
calls inside the inner Newton loop" from "decoder calls in the redundant
final _residual_jacobian_at call" since they go through the same method;
manual wrapping with per-name accumulators does.

Profiles COLD START (step_index=1, q_prev=zeros) for every point -- this
measures the per-iteration cost breakdown correctly (independent of warm
vs cold start: cold start only changes HOW MANY iterations run, not the
relative cost of each piece within one iteration), and reports the
iteration-count distribution explicitly so the caveat is visible.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
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
print(f"[profile] loaded {e_gp.shape[0]} real macro strains "
      f"(n_body=6 continuation run, fully_converged={bool(d['fully_converged'])})", flush=True)

law = m.HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR), qp_init_mode="continuation")
print(f"[profile] n_primary={law.n_primary}, n_secondary={law.n_secondary}, "
      f"Z_res support size={law.Z_res.size}, n_current_elements={law.n_current_elements}, "
      f"include_manifold_curvature={law.include_manifold_curvature}", flush=True)

timers = defaultdict(float)
counts = defaultdict(int)


def wrap_bound(obj, name):
    orig = getattr(obj, name)

    def wrapped(*a, **kw):
        t0 = time.perf_counter()
        r = orig(*a, **kw)
        timers[name] += time.perf_counter() - t0
        counts[name] += 1
        return r

    setattr(obj, name, wrapped)


for name in [
    "_eval_qs_and_jac", "_eval_qs_only", "_compute_weighted_decoder_hessian",
    "_residual_weights_and_jacobian", "_reaction_force_hom_sig", "_reaction_force_c_e",
    "_dirichlet_sensitivity", "_residual_jacobian_at", "dqp_dE_at",
    "_hom_weights", "_hom_weights_and_jacobian", "_evaluate_impl", "_solve_reduced_system",
]:
    wrap_bound(law, name)

orig_assemble = law.dyn_res_assembler.assemble_reduced_action


def wrapped_assemble(*a, **kw):
    t0 = time.perf_counter()
    r = orig_assemble(*a, **kw)
    timers["assemble_reduced_action(Z_res)"] += time.perf_counter() - t0
    counts["assemble_reduced_action(Z_res)"] += 1
    return r


law.dyn_res_assembler.assemble_reduced_action = wrapped_assemble

orig_cla = law.vec_assembler.ComputeLocalArrays


def wrapped_cla(*a, **kw):
    t0 = time.perf_counter()
    r = orig_cla(*a, **kw)
    timers["vec_assembler.ComputeLocalArrays(full_mesh)"] += time.perf_counter() - t0
    counts["vec_assembler.ComputeLocalArrays(full_mesh)"] += 1
    return r


law.vec_assembler.ComputeLocalArrays = wrapped_cla

for modname in ["InitializeNonLinearIteration", "FinalizeNonLinearIteration",
                "reaction_force_hom_sig_and_jacobian", "CalculateHomogenizedFromAssemblerWithElementWeights"]:
    orig_fn = getattr(m, modname)

    def make_wrapped(orig_fn=orig_fn, modname=modname):
        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            r = orig_fn(*a, **kw)
            timers[modname] += time.perf_counter() - t0
            counts[modname] += 1
            return r
        return wrapped

    setattr(m, modname, make_wrapped())

# warm-up (torch/Kratos first-call overhead, file caches)
_ = law.evaluate_with_tangent(e_gp[0], q_prev=np.zeros(law.n_primary), step_index=1)
for k in list(timers):
    timers[k] = 0.0
for k in list(counts):
    counts[k] = 0

N_SAMPLE = 100
rng = np.random.default_rng(0)
idx = rng.choice(e_gp.shape[0], size=min(N_SAMPLE, e_gp.shape[0]), replace=False)

t_total0 = time.perf_counter()
n_iters_list = []
converged_list = []
for i in idx:
    E = e_gp[i]
    _, _, q_p, n_it, converged, _, _, _, _ = law.evaluate_with_tangent(
        E, q_prev=np.zeros(law.n_primary), step_index=1,
    )
    n_iters_list.append(n_it)
    converged_list.append(bool(converged))
t_total = time.perf_counter() - t_total0

print(flush=True)
print(f"[profile] {len(idx)} points (cold start, step_index=1), total {t_total:.3f}s, "
      f"{t_total / len(idx) * 1000:.3f} ms/point", flush=True)
print(f"[profile] n_iters: min={min(n_iters_list)}, max={max(n_iters_list)}, "
      f"mean={np.mean(n_iters_list):.2f}, median={np.median(n_iters_list):.1f}", flush=True)
print(f"[profile] converged: {sum(converged_list)}/{len(converged_list)}", flush=True)
print(flush=True)
print(f"{'piece':42s} {'total_s':>10s} {'calls':>8s} {'ms/call':>10s} {'%total':>8s}")
accounted = 0.0
for name, t in sorted(timers.items(), key=lambda kv: -kv[1]):
    c = counts[name]
    pct = 100.0 * t / t_total
    print(f"{name:42s} {t:10.3f} {c:8d} {1000 * t / max(c, 1):10.4f} {pct:7.1f}%")
    accounted += t
print(f"{'[unaccounted / overlap-adjusted]':42s} {t_total - accounted:10.3f}")
print(flush=True)
print(f"[profile] mean iters/point = {np.mean(n_iters_list):.2f} -> "
      f"loop-body pieces (_eval_qs_and_jac, hessian, assemble_reduced_action, "
      f"_residual_weights_and_jacobian) each run ~{np.mean(n_iters_list):.1f}x per point, "
      f"while _residual_jacobian_at/dqp_dE_at/reaction-force run exactly 1x per point "
      f"(the post-convergence finalization).", flush=True)
print("PROFILE_DONE_MARKER", flush=True)
