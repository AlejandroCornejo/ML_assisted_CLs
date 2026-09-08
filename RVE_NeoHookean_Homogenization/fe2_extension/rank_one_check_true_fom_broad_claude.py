#!/usr/bin/env python3
"""Broad rank-one (Legendre-Hadamard) convexity audit of the TRUE,
non-reduced RVE FOM's own homogenized response -- the same question
Kalina et al. 2024 (arXiv:2410.03378) ask of their own RVEs via direct
sampling of the acoustic tensor, and answer "yes, it loses ellipticity"
for nearly all of theirs. This script asks the identical question of
THIS project's own RVE, which has never been checked this way before
(the existing broad audits, rank_one_convexity_check_all6_claude.py /
rank_one_final_all6_claude.py, only ever evaluated TRAINED surrogates,
not the true FOM; the only true-FOM tangent fact on record is the
single-state SPD-loss finding already in the paper, which checks a
different, stronger condition at one state, not rank-one convexity
broadly).

Methodology, deliberately reusing already-validated pieces rather than
re-deriving them:
  - F-sampling distribution: EXACTLY sample_F_a_b's (diag stretches
    exp(uniform(-1,1)), off-diag uniform(-0.4,0.4)), the same
    distribution already used for the published broad-sweep numbers on
    the four PANN tiers and D-HPROM-ANN/HPROM-ANN, so this new number is
    directly comparable to those, not a different experiment.
  - Curvature formula: rank_one_curvature_from_S_CC, already validated
    to machine precision against energy-autograd on the certified PANN.
  - FOM solve + finite-difference tangent: solve_at_strain (the same
    validated cold-start RVE solver used throughout this project) plus
    a manual central difference, h=1e-4, matching
    fom_nested_law_claude.py's own fom_nested_pk2_2d_vectorized exactly.

Cost control: a real FOM solve is ~1-1.1s (confirmed from this
project's own prior runs), and each state needs 1 base + 6 perturbed
solves for the finite-difference tangent (~7-8s/state). Unlike the
free/cheap trained-surrogate audits (2000 fresh (F,a,b) triples), this
script amortizes that cost the way rank_one_check_at_cook_states_claude
.py's audit_at_real_states already does for real Cook states: evaluate
(S,CC) ONCE per sampled F, then check many random (a,b) directions
against that same (S,CC) for free. N_STATES=150, N_DIRECTIONS=20 (3000
total curvature checks) costs ~150*7.5s =~ 19 minutes, not the ~4h a
naive fresh-(F,a,b)-per-sample 2000-count would cost here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE_DIR = ROOT / "core"
for p in (str(CORE_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from rank_one_convexity_check_claude import sample_F_a_b, strain_voigt_from_F, rank_one_curvature_from_S_CC  # noqa: E402
from time_fom_single_query_claude import solve_at_strain  # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402

N_STATES = 150
N_DIRECTIONS = 20
H_FD = 1.0e-4
SEED = 20260828
OUT_PATH = HERE / "rank_one_true_fom_broad_result_claude.npz"


def eval_true_fom_S_and_CC(E_voigt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(E_voigt (3,)) -> (S (3,), CC (3,3)), via one cold-start FOM solve
    plus 6 central-difference-perturbed solves. Exact same pattern as
    fom_nested_law_claude.py's fom_nested_pk2_2d_vectorized, single-state."""
    e0 = np.asarray(E_voigt, dtype=float).reshape(3)
    with true_neo_hookean_active():
        _eps0, sig0 = solve_at_strain(e0)
        CC = np.zeros((3, 3), dtype=float)
        for k in range(3):
            ep = e0.copy(); ep[k] += H_FD
            em = e0.copy(); em[k] -= H_FD
            _epsp, sigp = solve_at_strain(ep)
            _epsm, sigm = solve_at_strain(em)
            CC[:, k] = (sigp - sigm) / (2.0 * H_FD)
    return np.asarray(sig0, dtype=float).reshape(3), CC


def main() -> None:
    generator = torch.Generator().manual_seed(SEED)
    dtype = torch.float64

    states = []
    n_drawn = 0
    while len(states) < N_STATES:
        f, _a_unused, _b_unused = sample_F_a_b(generator, dtype, stretch_log_range=1.0)
        n_drawn += 1
        if torch.det(f).item() <= 0.0:
            continue
        states.append(f.numpy())
    print(f"[true-fom-broad] drew {len(states)} valid F states out of {n_drawn} samples "
          f"(stretch_log_range=1.0, same distribution as the published broad sweep)")

    rng = np.random.default_rng(SEED)
    all_curv = np.full((N_DIRECTIONS, N_STATES), np.nan, dtype=float)
    n_solve_failures = 0
    t_start = time.perf_counter()

    for g, F0 in enumerate(states):
        E0_voigt = strain_voigt_from_F(F0)
        try:
            S0, CC0 = eval_true_fom_S_and_CC(E0_voigt)
        except Exception as exc:  # noqa: BLE001
            n_solve_failures += 1
            print(f"    [state {g + 1}/{N_STATES}] FOM solve failed ({exc!r}), skipping this state")
            continue

        a = rng.standard_normal((N_DIRECTIONS, 2)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = rng.standard_normal((N_DIRECTIONS, 2)); b /= np.linalg.norm(b, axis=1, keepdims=True)
        for d in range(N_DIRECTIONS):
            all_curv[d, g] = rank_one_curvature_from_S_CC(F0, a[d], b[d], S0, CC0)

        elapsed = time.perf_counter() - t_start
        print(f"    [state {g + 1}/{N_STATES}] done "
              f"({elapsed:.1f}s elapsed, {elapsed / (g + 1):.2f}s/state avg)", flush=True)

        # Persist partial results after every state, so a crash/interrupt
        # doesn't lose already-computed (expensive) FOM evaluations.
        np.savez(OUT_PATH, all_curv=all_curv, n_states_done=g + 1,
                 n_solve_failures=n_solve_failures, n_states=N_STATES,
                 n_directions=N_DIRECTIONS, seed=SEED)

    flat = all_curv.reshape(-1)
    valid = flat[~np.isnan(flat)]
    n_violations = int(np.sum(valid < 0.0))
    n_total = valid.size
    print(f"\n[true-fom-broad] FINAL: {N_STATES - n_solve_failures} valid states x "
          f"{N_DIRECTIONS} directions ({n_total} samples, {n_solve_failures} states "
          f"failed to solve): {n_violations} negative "
          f"({100 * n_violations / max(n_total, 1):.3f}%), "
          f"worst curvature = {valid.min() if n_total else float('nan'):.6e}, "
          f"median curvature = {np.median(valid) if n_total else float('nan'):.6e}")
    np.savez(OUT_PATH, all_curv=all_curv, n_states_done=N_STATES,
             n_solve_failures=n_solve_failures, n_states=N_STATES,
             n_directions=N_DIRECTIONS, seed=SEED,
             n_violations=n_violations, n_total=n_total,
             fraction_violations=n_violations / max(n_total, 1),
             worst_curvature=float(valid.min()) if n_total else float("nan"),
             median_curvature=float(np.median(valid)) if n_total else float("nan"))
    print(f"[true-fom-broad] saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
