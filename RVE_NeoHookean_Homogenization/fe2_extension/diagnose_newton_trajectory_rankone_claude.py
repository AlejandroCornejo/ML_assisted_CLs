#!/usr/bin/env python3
"""Tests a specific hypothesis raised in conversation: the earlier
rank-one check (rank_one_check_at_cook_states_claude.py) found ZERO
violations at the BEST (lowest-residual) iterate of every Cook load step
-- but that is, by construction, the single nicest point in that step's
whole Newton trajectory. It says nothing about whether Newton passes
through rank-one-INADMISSIBLE states at the iterates it visits along the
way (and then discards, because run_newton_fe2's stall-tolerant logic
keeps only the best one).

This script does NOT touch run_newton_fe2 (the already-validated driver
behind every reported number) -- it is a standalone copy of just enough
of Cook's own Newton loop (same mesh, same BCs, same material law
monkeypatch, same line search, imported read-only from
Cook.gid/run_cook_pann_claude.py) to record assembler._F (the real
deformation gradient at all 384 Gauss points) after EVERY iteration, not
just the winning one, for D-HPROM-ANN-consistent's own step-1 stall
(the most heavily characterized case this session).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import fom_solver_rve as fom  # noqa: E402
from build_cook_mesh_claude import build_mesh  # noqa: E402
from run_cook_pann_claude import build_model_part, consistent_edge_force, _line_search_alpha  # noqa: E402
import dhprom_ann_direct_law_claude as dhprom_module  # noqa: E402

from rank_one_convexity_check_claude import rank_one_curvature_from_S_CC, strain_voigt_from_F  # noqa: E402

EDGE_LENGTH = 16.0
LINE_LOAD_MODULUS_FINAL = 13_000_000.0
TOTAL_FORCE_FINAL = LINE_LOAD_MODULUS_FINAL * EDGE_LENGTH
N_STEPS = 20
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9


def run_step1_recording_every_iterate(nx=8, ny=8, max_it=15):
    mp, coords, tris, left_nodes, right_nodes = build_model_part(nx, ny)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = dhprom_module.dhprom_ann_pk2_2d_vectorized_consistent
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label="diag-trajectory")

    f_unit = consistent_edge_force(coords, right_nodes)
    f_unit_eq = np.zeros(n_dof)
    np.add.at(f_unit_eq, eq_map[:, 0], f_unit[:, 0])
    np.add.at(f_unit_eq, eq_map[:, 1], f_unit[:, 1])

    free_dofs = np.array([d for d in range(n_dof)
                           if d not in set(eq_map[left_nodes, 0]) | set(eq_map[left_nodes, 1])])

    u = np.zeros(n_dof)
    load_factor = TOTAL_FORCE_FINAL / N_STEPS
    f_ext = f_unit_eq * load_factor

    iterate_F = []
    iterate_res = []
    res_norm0 = None
    for it in range(1, max_it + 1):
        t0 = time.perf_counter()
        K, rhs_int = assembler.Assemble(u)
        F_now = assembler._F.reshape(-1, 2, 2).copy()
        iterate_F.append(F_now)

        residual = rhs_int + f_ext
        res_free = residual[free_dofs]
        res_norm = float(np.linalg.norm(res_free))
        iterate_res.append(res_norm)
        if res_norm0 is None:
            res_norm0 = max(res_norm, 1e-12)
        print(f"  iter {it:2d}: |res|={res_norm:.6e}  |res|/|res0|={res_norm / res_norm0:.6e}  ({time.perf_counter() - t0:.2f}s)")
        if res_norm < RESIDUAL_ABS_TOL or res_norm / res_norm0 < RESIDUAL_REL_TOL:
            print("  converged.")
            break

        K_ff = K[free_dofs, :][:, free_dofs]
        du_free = spsolve(K_ff.tocsc(), res_free)
        if not np.all(np.isfinite(du_free)):
            print("  non-finite update, stopping.")
            break
        alpha = _line_search_alpha(assembler, u, du_free, free_dofs, f_ext)
        u[free_dofs] += alpha * du_free

    return iterate_F, iterate_res


def check_all_iterates(iterate_F, eval_S_and_CC, n_directions=10, seed=0):
    """For each recorded iterate, run the rank-one check (S,CC cached
    once per Gauss point per iterate, reused across n_directions) and
    report violations per iterate."""
    results = []
    for it_idx, F_all in enumerate(iterate_F, start=1):
        n_gp = F_all.shape[0]
        S_all = np.zeros((n_gp, 3), dtype=float)
        CC_all = np.zeros((n_gp, 3, 3), dtype=float)
        for g in range(n_gp):
            E0_voigt = strain_voigt_from_F(F_all[g])
            S0, CC0 = eval_S_and_CC(E0_voigt)
            S_all[g] = np.asarray(S0, dtype=float).reshape(3)
            CC_all[g] = np.asarray(CC0, dtype=float).reshape(3, 3)

        rng = np.random.default_rng(seed)
        n_violations = 0
        n_total = 0
        worst = np.inf
        worst_gp = None
        for d in range(n_directions):
            a = rng.standard_normal((n_gp, 2)); a /= np.linalg.norm(a, axis=1, keepdims=True)
            b = rng.standard_normal((n_gp, 2)); b /= np.linalg.norm(b, axis=1, keepdims=True)
            for g in range(n_gp):
                curv = rank_one_curvature_from_S_CC(F_all[g], a[g], b[g], S_all[g], CC_all[g])
                n_total += 1
                if curv < 0.0:
                    n_violations += 1
                if curv < worst:
                    worst = curv
                    worst_gp = g
        strain_norms = np.linalg.norm(
            np.stack([strain_voigt_from_F(F_all[g]) for g in range(n_gp)], axis=0), axis=1,
        )
        print(f"    iterate {it_idx}: {n_violations}/{n_total} rank-one violations "
              f"({100 * n_violations / n_total:.2f}%), worst_curvature={worst:.4e} at gp={worst_gp}, "
              f"|E| range=[{strain_norms.min():.3e},{strain_norms.max():.3e}]")
        results.append({
            "iterate": it_idx, "n_violations": n_violations, "n_total": n_total,
            "fraction": n_violations / n_total, "worst_curvature": float(worst),
        })
    return results


if __name__ == "__main__":
    print("[diag] running Cook step-1 Newton with D-HPROM-ANN-consistent, recording every iterate's F ...")
    iterate_F, iterate_res = run_step1_recording_every_iterate()

    print(f"\n[diag] recorded {len(iterate_F)} iterates; running rank-one check on EACH (not just the best) ...")
    from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw

    law = DHpromAnnDirectLaw()

    def eval_dhprom(E_voigt):
        _eps, sig, _dEps, dSig = law.evaluate_with_tangent(E_voigt)
        return sig, dSig

    results = check_all_iterates(iterate_F, eval_dhprom, n_directions=10, seed=0)

    print("\n=== SUMMARY: rank-one violations across the Newton trajectory (not just the best iterate) ===")
    any_violation = False
    for r in results:
        tag = "VIOLATION" if r["n_violations"] > 0 else "clean"
        if r["n_violations"] > 0:
            any_violation = True
        print(f"  iterate {r['iterate']:2d}: {r['n_violations']:4d}/{r['n_total']:4d} "
              f"({r['fraction']:.3%}) [{tag}]")
    print(f"\n[diag] {'Found rank-one violations DURING the Newton trajectory.' if any_violation else 'NO rank-one violations found at ANY recorded iterate -- hypothesis not confirmed this way.'}")
