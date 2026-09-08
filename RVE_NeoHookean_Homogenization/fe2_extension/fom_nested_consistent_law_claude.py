#!/usr/bin/env python3
"""True FOM-FE2 material law -- corrected, consistent version.

Two corrections relative to fom_nested_law_claude.py (the original,
exploratory version, kept unmodified for provenance):

1. Homogenized STRESS via the reaction-force-conjugate convention (same
   quantity HPROM-ANN/D-HPROM-ANN/Linear-HPROM already report), not the
   naive volume average fom_nested_law_claude.py used. Formula reused
   verbatim from the already-validated pann/direct_energy/
   reaction_force_direct_stress.py's DirectStressGenerator:

       R = rhs[dir_dofs]                      (rhs == -f_int everywhere,
                                                 this RVE has no external
                                                 force term at all)
       S = -(sens @ R) / (thickness * A0)     (envelope theorem)

   where sens = d(u_dirichlet)/dE, the closed-form affine map's own
   analytic sensitivity (deformation_gradient_and_jacobian_2d +
   _sens_from_dF_dE, both already used by every other law this session --
   NOT the FD-on-the-affine-map DirectStressGenerator itself uses, which
   is FD wrapping a closed form and numerically equivalent, but the
   analytic route avoids that FD layer entirely).

2. An ANALYTIC tangent via the implicit function theorem on the solve's
   own converged tangent stiffness K, replacing fom_nested_law_claude.py's
   6-extra-full-nonlinear-solve central-FD tangent (each of those 6 solves
   is a full independent 400-substep nonlinear ramp, not a cheap
   perturbation -- confirmed by profiling this session, ~0.49s each). K is
   already assembled as a side effect of RunFomBatchSimulation's own
   "force a full LocalSystem evaluation at converged state" step for
   every prior use of this function in this project; return_final_state=
   True (new, purely additive parameter on that shared core function)
   just exposes it instead of discarding it -- no new re-solve, no change
   to the converged VALUE at all, only how the tangent gets computed.

   Derivation: equilibrium is f_int_free(u_free, u_dir(E)) = 0. At the
   converged state, rhs == -f_int everywhere, so differentiating:

       K_FF @ du_free/dE + K_FD @ du_dir/dE = 0
       du_free/dE = -K_FF^{-1} @ K_FD @ du_dir/dE      (sparse solve,
                                                          K already
                                                          assembled)

   with du_dir/dE = sens.T (same closed-form sensitivity as the stress
   formula above). Then, since R = rhs[dir_dofs] = -f_int[dir_dofs] and
   f_int depends on E only through u(E):

       dR/dE = -K[dir_dofs, :] @ du/dE                  (du/dE assembled
                                                           from both free
                                                           and dirichlet
                                                           blocks above)
       dS/dE = -(dsens_dE valued-at-R  +  sens @ dR/dE) / (thickness*A0)

   dsens_dE (sens's own curvature, "Term A") reuses reaction_force_hom_
   tangent_claude.py's own _sens_from_dF_dE pattern verbatim: one central
   finite difference on the already-analytic dF_dE, not a re-solve and
   not FD-of-FD.

Both corrections are verified (see _verify_fom_nested_consistent_claude.py)
against: (a) DirectStressGenerator's own independent value computation,
and (b) a direct finite-difference cross-check of the analytic tangent
against this SAME reaction-force stress formula (not the old naive-
average one) evaluated at E+h/E-h.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE_DIR = ROOT / "core"
for p in (str(CORE_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM  # noqa: E402
import fom_solver_rve as fom  # noqa: E402

from deformation_gradient_jacobian_claude import deformation_gradient_and_jacobian_2d  # noqa: E402
from reaction_force_hom_tangent_claude import _sens_from_dF_dE  # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402

_N_CALLS = 0


def make_parameters(mesh_base=None, materials=None):
    """mesh_base/materials are PURELY ADDITIVE: at their defaults this returns
    exactly the parameters every prior caller in this project got. They exist
    so a second study can point the same validated solver path at a different
    RVE without forking this file -- one implementation, two configurations."""
    with open(CORE_DIR / "ProjectParameters.json", encoding="utf-8") as f:
        config = json.load(f)
    config["output_processes"] = {"gid_output": [], "vtk_output": []}
    config["solver_settings"]["echo_level"] = 0
    parameters = KM.Parameters(json.dumps(config))
    mesh_base = str(CORE_DIR / "rve_geometry") if mesh_base is None else str(mesh_base)
    materials = str(CORE_DIR / "StructuralMaterials.json") if materials is None else str(materials)
    fom.SetInputMeshFilename(parameters, mesh_base)
    material_parts = fom.DetectMaterialSubModelParts(mesh_base + ".mdpa")
    parameters = fom.ConfigureElementModelerForMaterialParts(parameters, material_parts)
    fom.SetMaterialsFilename(parameters, materials)
    return parameters


def evaluate_with_tangent(E, reference_amplitude=2.0, reference_steps=400,
                           out_dir=None, heps=1.0e-6, verbose=False,
                           u_init=None, E_start=None, return_u=False,
                           mesh_base=None, materials=None,
                           hom_reference_measure=None):
    """(E,) -> (S (3,), CC (3,3)): the true FOM-FE2 reaction-force stress
    and its analytic tangent at macro strain E, via ONE full nonlinear RVE
    solve (same ramped-Newton controls as every prior FOM use in this
    project) plus one cheap sparse linear solve -- no finite-difference
    perturbation solves at all.

    u_init/E_start/return_u are PURELY ADDITIVE warm-start support (same
    convention as RunFomBatchSimulation's own return_final_state): with all
    three at their defaults this function is bit-identical to before --
    strain_path=[0,E], initial_displacement=None, returns (S, CC).

    Warm-starting is free of any modelling approximation here because this
    RVE is path-independent (Neo-Hookean hyperelastic, no history variables
    at all): the converged solution is a pure function of E, so ramping
    from zero in N substeps and continuing from a nearby converged state
    reach the SAME state to solver tolerance. The from-zero ramp is a
    Newton-convergence aid, not physics. What it buys: BuildDynamicSegmentSteps
    allocates substeps proportional to the path's own length, so a
    [E_prev -> E] increment between consecutive macro Newton iterations
    needs 1 substep where [0 -> E] needs ~200*|E| of them (13 at |E|~0.065,
    ~55 at this project's own full dogbone load).

    mesh_base/materials/hom_reference_measure are likewise purely additive,
    so a second study can drive this same validated path with its own RVE and
    its own homogenization denominator. That denominator matters: the solver's
    own default is the SOLID area (sum of element areas), whereas macro stress
    is force per unit MACRO area and so must include the void. Passing it here
    also reaches the analytic tangent, since the value travels in final_state
    and is used for both S and dS/dE.

    Caveat, stated rather than assumed: path-independence fails if the RVE
    admits multiple stable branches at the same E (e.g. pore buckling under
    compression), where a warm start could land on a different branch than
    a from-zero ramp. Tension-dominated use is safe; near pore closure it
    is not, and a cold start should be preferred there."""
    global _N_CALLS
    _N_CALLS += 1
    out_dir = out_dir or str(HERE / "fom_query_scratch")
    E = np.asarray(E, dtype=float).reshape(3)
    E0 = np.zeros(3, dtype=float) if E_start is None else np.asarray(E_start, dtype=float).reshape(3)

    with true_neo_hookean_active():
        strain_hist, stress_hist, final_state = fom.RunFomBatchSimulation(
            parameters=make_parameters(mesh_base=mesh_base, materials=materials),
            out_dir=out_dir,
            save_results=False, save_plot=False,
            strain_path=np.vstack((E0, E)), trajectory_index=1,
            reference_amplitude=reference_amplitude, reference_steps=reference_steps,
            initial_displacement=u_init,
            hom_reference_measure=hom_reference_measure,
            return_final_state=True,
        )

    K = final_state["K"].tocsr()
    rhs = np.asarray(final_state["rhs"], dtype=float).reshape(-1)
    free_dofs = np.asarray(final_state["free_dofs"], dtype=np.int64)
    dir_dofs = np.asarray(final_state["dir_dofs"], dtype=np.int64)
    dir_x = np.asarray(final_state["dir_x"], dtype=float)
    dir_y = np.asarray(final_state["dir_y"], dtype=float)
    dir_is_x = np.asarray(final_state["dir_is_x"], dtype=bool)
    n_dof = rhs.size
    denom = float(final_state["thickness"]) * float(final_state["hom_reference_measure"])

    _F_macro, dF_dE = deformation_gradient_and_jacobian_2d(E)
    sens = _sens_from_dF_dE(dF_dE, dir_x, dir_y, dir_is_x)  # (3, n_dir)

    R = rhs[dir_dofs]
    S = -(sens @ R) / denom

    # Term A: sens's own curvature (d(sens)/dE), one central FD on the
    # already-analytic dF_dE -- not a re-solve, not FD-of-FD.
    dsens_dE = np.empty((3, dir_dofs.size, 3), dtype=float)
    for x in range(3):
        step = heps if abs(E[x]) < 1.0 else heps * max(1.0, abs(E[x]))
        Ep, Em = E.copy(), E.copy()
        Ep[x] += step
        Em[x] -= step
        _, dF_dE_p = deformation_gradient_and_jacobian_2d(Ep)
        _, dF_dE_m = deformation_gradient_and_jacobian_2d(Em)
        sens_p = _sens_from_dF_dE(dF_dE_p, dir_x, dir_y, dir_is_x)
        sens_m = _sens_from_dF_dE(dF_dE_m, dir_x, dir_y, dir_is_x)
        dsens_dE[:, :, x] = (sens_p - sens_m) / (2.0 * step)

    # du/dE: dirichlet block is the same closed-form sens; free block via
    # the implicit function theorem on the already-assembled K.
    du_dE = np.zeros((n_dof, 3), dtype=float)
    du_dE[dir_dofs, :] = sens.T

    K_FF = K[free_dofs][:, free_dofs].tocsc()
    K_FD = K[free_dofs][:, dir_dofs]
    rhs_free = -(K_FD @ sens.T)  # (n_free, 3)
    du_free_dE = np.column_stack([
        spsolve(K_FF, rhs_free[:, k]) for k in range(3)
    ])
    du_dE[free_dofs, :] = du_free_dE

    # Term B: dR/dE = -K[dir_dofs, :] @ du/dE (f_int depends on E only
    # through u(E), so d(f_int)/dE = K @ du/dE; rhs = -f_int).
    K_dir_rows = K[dir_dofs, :]
    dR_dE = -(K_dir_rows @ du_dE)  # (n_dir, 3)

    term_A = np.einsum("kjl,j->kl", dsens_dE, R)
    term_B = sens @ dR_dE
    CC = -(term_A + term_B) / denom

    if verbose:
        print(f"    [fom_nested_consistent call #{_N_CALLS}] E={E}, S={S}", flush=True)

    if return_u:
        return S, CC, np.asarray(final_state["u"], dtype=float).reshape(-1).copy()
    return S, CC


def fom_nested_consistent_pk2_2d_vectorized(E_flat, young=None, poisson=None,
                                              reference_amplitude=2.0, reference_steps=400,
                                              out_dir=None, verbose=True):
    """(E_flat (N,3), young, poisson) -> (S (N,3), CC (N,3,3)), same
    contract as fom_nested_law_claude.py's own fom_nested_pk2_2d_vectorized
    (young/poisson accepted but unused, signature compatibility only)."""
    E_flat = np.asarray(E_flat, dtype=float).reshape(-1, 3)
    n = E_flat.shape[0]
    S = np.zeros((n, 3), dtype=float)
    CC = np.zeros((n, 3, 3), dtype=float)
    t0 = time.perf_counter()
    for i in range(n):
        S[i], CC[i] = evaluate_with_tangent(
            E_flat[i], reference_amplitude=reference_amplitude, reference_steps=reference_steps,
            out_dir=out_dir,
        )
        if verbose and (i + 1) % 16 == 0:
            dt = time.perf_counter() - t0
            print(f"    [fom_nested_consistent call #{_N_CALLS}] {i + 1}/{n} Gauss points done "
                  f"({dt:.1f}s elapsed, {dt / (i + 1):.3f}s/GP)", flush=True)
    if verbose:
        dt = time.perf_counter() - t0
        print(f"    [fom_nested_consistent call #{_N_CALLS}] DONE: {n} Gauss points, {dt:.1f}s total "
              f"({dt / max(n, 1):.3f}s/GP)", flush=True)
    return S, CC


if __name__ == "__main__":
    E_query = np.array([0.01173723, 0.00614076, 0.04820225])
    print(f"[fom_nested_consistent] query at E={E_query}", flush=True)
    t0 = time.perf_counter()
    S, CC = evaluate_with_tangent(E_query, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"[fom_nested_consistent] S={S}", flush=True)
    print(f"[fom_nested_consistent] CC=\n{CC}", flush=True)
    print(f"[fom_nested_consistent] ELAPSED = {elapsed:.2f}s for one consistent-tangent query", flush=True)
