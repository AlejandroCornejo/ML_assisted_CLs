#!/usr/bin/env python3
"""Profile LinearHpromIterativeLawFloat64.evaluate_with_tangent with
per-piece timing, on the SAME 600 real macro strains used to profile
HPROM-ANN (e_gp from the n_body=6 continuation run's own final/hardest
step) -- reusing this cruciform-realistic strain sample avoids needing an
expensive full cruciform macro-Newton run just to get profiling data,
since the strain VALUES are representative regardless of which law
originally produced the run they came from (same mesh/geometry)."""
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

import linear_hprom_iterative_law_float64_claude as m  # noqa: E402

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]
print(f"[profile] loaded {e_gp.shape[0]} real macro strains (reused from HPROM-ANN's own run)", flush=True)

law = m.LinearHpromIterativeLawFloat64()
print(f"[profile] n_primary={law.n_primary}, res_assembler n_elems={law.res_assembler.n_elems}, "
      f"n_current_elements={law.n_current_elements}, n_total_dof={law.n_total_dof}", flush=True)

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


for name in ["_evaluate_impl", "_solve_reduced_system", "_reaction_force_hom_sig", "_reaction_force_c_e",
             "_dirichlet_sensitivity"]:
    wrap_bound(law, name)

orig_assemble = law.res_assembler.Assemble


def wrapped_assemble(*a, **kw):
    t0 = time.perf_counter()
    r = orig_assemble(*a, **kw)
    timers["res_assembler.Assemble"] += time.perf_counter() - t0
    counts["res_assembler.Assemble"] += 1
    return r


law.res_assembler.Assemble = wrapped_assemble

orig_cla = law.vec_assembler.ComputeLocalArrays


def wrapped_cla(*a, **kw):
    t0 = time.perf_counter()
    r = orig_cla(*a, **kw)
    timers["vec_assembler.ComputeLocalArrays(full_mesh)"] += time.perf_counter() - t0
    counts["vec_assembler.ComputeLocalArrays(full_mesh)"] += 1
    return r


law.vec_assembler.ComputeLocalArrays = wrapped_cla

for modname in ["InitializeNonLinearIteration", "FinalizeNonLinearIteration",
                "reaction_force_hom_sig_and_jacobian", "maw_hom_weight_and_jacobian_single_model",
                "_evaluate_maw_hom_weights_current",
                "SetDisplacementFromEquationVector", "UpdateCurrentCoordinatesFromDisplacement"]:
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

# warm-up
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
print(f"[profile] {len(idx)} points (cold start), total {t_total:.3f}s, "
      f"{t_total / len(idx) * 1000:.3f} ms/point", flush=True)
print(f"[profile] n_iters: min={min(n_iters_list)}, max={max(n_iters_list)}, "
      f"mean={np.mean(n_iters_list):.2f}, median={np.median(n_iters_list):.1f}", flush=True)
print(f"[profile] converged: {sum(converged_list)}/{len(converged_list)}", flush=True)
print(flush=True)
print(f"{'piece':42s} {'total_s':>10s} {'calls':>8s} {'ms/call':>10s} {'%total':>8s}")
for name, t in sorted(timers.items(), key=lambda kv: -kv[1]):
    c = counts[name]
    pct = 100.0 * t / t_total
    print(f"{name:42s} {t:10.3f} {c:8d} {1000 * t / max(c, 1):10.4f} {pct:7.1f}%")
print("PROFILE_DONE_MARKER", flush=True)
