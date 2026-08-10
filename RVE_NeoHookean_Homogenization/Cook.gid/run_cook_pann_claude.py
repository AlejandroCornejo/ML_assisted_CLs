"""Cook's membrane, large-deformation, driven entirely by a trained PANN
material law (certified ICNN / free / ICKAN) instead of the base
microscale Neo-Hookean law. Reuses core/fom_solver_rve.py's vectorized
Total-Lagrangian assembler and DOF machinery (Kratos is used only for
mesh/DOF bookkeeping, since this Kratos build cannot dispatch a
Python-overridden ConstitutiveLaw), monkeypatching its material-law call
to evaluate the selected trained PANN via autodiff instead of the
closed-form Neo-Hookean law.

Loading: same ramped total tip shear force as Cook.gid/ProjectParameters.json
(direction +y on RightEdge), calibrated (Cook.gid/check_strain_claude.py)
to stay within this project's own trained strain range at full load.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

sys.path.insert(0, "/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/core")
import fom_solver_rve as fom  # noqa: E402
import KratosMultiphysics as KM  # noqa: E402

from build_cook_mesh_claude import build_mesh  # noqa: E402
import pann_constitutive_law_claude as pann_law  # noqa: E402

EDGE_LENGTH = 16.0  # |C-B|, the (exactly straight) right edge of classic Cook's membrane
LINE_LOAD_MODULUS_FINAL = 13_000_000.0  # matches Cook.gid/ProjectParameters.json's calibrated final modulus (N/m)
TOTAL_FORCE_FINAL = LINE_LOAD_MODULUS_FINAL * EDGE_LENGTH  # total N, matching the validated Kratos run
N_STEPS = 20
MAX_NEWTON_ITER = 30
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9


def _rank_one_check(assembler, which: str, n_directions: int = 20, h: float = 1.0e-3):
    """Samples the rank-one convexity necessary condition,
    d^2/dt^2 W(F + t a(x)b)|_{t=0} >= 0, at every Gauss-point F actually
    visited by the stalled Newton iterate above, for random unit a,b --
    the same falsification attempt as the paper's own Section 6.5, but at
    real structural states instead of a synthetic global sample.
    Not defined for the regression tier: it has no energy potential.
    Returns the full array of sampled curvatures (for later plotting).
    """
    if which == "regression":
        print("      rank-one check: not applicable (no energy potential for this tier)")
        return np.zeros(0)
    law = pann_law.get_law(which)
    F = assembler._F.reshape(-1, 2, 2)  # (n_gp_total, 2, 2), from the last Assemble() call
    n_gp = F.shape[0]
    rng = np.random.default_rng(0)

    worst = np.inf
    n_violations = 0
    n_total = 0
    all_d2 = []
    for _ in range(n_directions):
        a = rng.standard_normal((n_gp, 2)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = rng.standard_normal((n_gp, 2)); b /= np.linalg.norm(b, axis=1, keepdims=True)
        ab = a[:, :, None] * b[:, None, :]  # (n_gp,2,2), a outer b

        def energy_at(t):
            Ft = F + t * ab
            C = np.einsum("gki,gkj->gij", Ft, Ft)
            e_voigt = np.stack([0.5 * (C[:, 0, 0] - 1.0), 0.5 * (C[:, 1, 1] - 1.0), C[:, 0, 1]], axis=1)
            import torch
            model_dtype = law.model.strain_scale.dtype
            raw = torch.as_tensor(e_voigt, dtype=model_dtype)
            normalised = raw / law.strain_scale
            energy_hat = law.model.energy(normalised)
            return energy_hat.detach().numpy().reshape(-1) * law.energy_scale

        w_plus, w0, w_minus = energy_at(h), energy_at(0.0), energy_at(-h)
        d2 = (w_plus - 2.0 * w0 + w_minus) / (h * h)
        worst = min(worst, float(d2.min()))
        n_violations += int(np.sum(d2 < 0.0))
        n_total += d2.size
        all_d2.append(d2)

    all_d2 = np.concatenate(all_d2)
    print(f"      rank-one check @ {n_gp} real Gauss-point states x {n_directions} random directions "
          f"({n_total} samples): {n_violations} negative, worst curvature = {worst:.4e}")
    return all_d2


def _line_search_alpha(assembler, u_base, du_free, free_dofs, f_ext,
                        first_alpha=0.5, second_alpha=1.0, max_it=10,
                        min_alpha=0.1, max_alpha=2.0, tol=0.5):
    """Same secant/interpolation scheme as core/fom_solver_rve.py's
    ComputeLineSearchAlpha (matching Kratos's own LineSearchStrategy),
    adapted to evaluate the directional residual via the vectorized
    assembler instead of Kratos's native per-entity RHS assembly."""
    u_trial = u_base.copy()

    def eval_r(alpha):
        u_trial[:] = u_base
        u_trial[free_dofs] = u_base[free_dofs] + alpha * du_free
        _, rhs_int = assembler.Assemble(u_trial)
        residual_free = (rhs_int + f_ext)[free_dofs]
        return float(alpha * np.dot(du_free, residual_free))

    x1, x2 = float(first_alpha), float(second_alpha)
    r1, r2 = eval_r(x1), eval_r(x2)
    rmax = max(abs(r1), abs(r2))
    x = x2
    for _ in range(int(max_it)):
        rmin = min(abs(r1), abs(r2))
        x = (r1 * x2 - r2 * x1) / (r1 - r2) if abs(r1 - r2) > 1e-10 else 1.0
        x = min(max(x, min_alpha), max_alpha)
        rf = eval_r(x)
        if rmin < tol * rmax or abs(rf) < tol * rmax:
            break
        if abs(r1) > abs(r2):
            r1, x1 = rf, x
        else:
            r2, x2 = r1, x1
            r1, x1 = rf, x
        rmax = max(rmax, abs(rf))
    return float(x)


def build_model_part(nx: int, ny: int):
    coords, tris, left_nodes, right_nodes = build_mesh(nx, ny)

    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.SetBufferSize(1)
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.AddNodalSolutionStepVariable(KM.REACTION)

    for i, (x, y) in enumerate(coords):
        mp.CreateNewNode(i + 1, float(x), float(y), 0.0)

    prop = mp.GetProperties()[1]
    prop.SetValue(KM.YOUNG_MODULUS, 1.0)
    prop.SetValue(KM.POISSON_RATIO, 0.3)
    prop.SetValue(KM.THICKNESS, 1.0)

    for e, conn in enumerate(tris):
        node_ids = [int(c) + 1 for c in conn]
        mp.CreateNewElement("TotalLagrangianElement2D6N", e + 1, node_ids, prop)

    KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, mp)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, mp)

    for nid in left_nodes:
        node = mp.GetNode(nid + 1)
        node.Fix(KM.DISPLACEMENT_X)
        node.Fix(KM.DISPLACEMENT_Y)

    return mp, coords, tris, left_nodes, right_nodes


def consistent_edge_force(coords: np.ndarray, right_nodes: list[int], direction=(0.0, 1.0)) -> np.ndarray:
    """Unit-total-force consistent nodal load along the (exactly straight)
    right edge, quadratic (Simpson 1/6-4/6-1/6) sub-edge weights."""
    n_nodes = coords.shape[0]
    f = np.zeros((n_nodes, 2))
    ny_sub = (len(right_nodes) - 1) // 2
    d = np.array(direction, dtype=float)
    for k in range(ny_sub):
        i_lo, i_mid, i_hi = right_nodes[2 * k], right_nodes[2 * k + 1], right_nodes[2 * k + 2]
        w = 1.0 / ny_sub
        f[i_lo] += (w / 6.0) * d
        f[i_hi] += (w / 6.0) * d
        f[i_mid] += (4.0 * w / 6.0) * d
    return f


def run_newton(which: str, nx: int = 10, ny: int = 10, verbose: bool = True, diagnose: bool = False,
               use_line_search: bool = False, save_npz: bool = True, force_continue: bool = False,
               name_suffix: str = ""):
    mp, coords, tris, left_nodes, right_nodes = build_model_part(nx, ny)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    # Monkeypatch the module-level constitutive law used inside VectorizedAssembler.
    def _pann_material(e_voigt, young, poisson):
        return pann_law.pann_pk2_2d_vectorized(e_voigt, which=which)

    fom._neo_hookean_pk2_2d_vectorized = _pann_material

    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"Cook[{which}]")

    f_unit = consistent_edge_force(coords, right_nodes)
    f_unit_eq = np.zeros(n_dof)
    np.add.at(f_unit_eq, eq_map[:, 0], f_unit[:, 0])
    np.add.at(f_unit_eq, eq_map[:, 1], f_unit[:, 1])

    free_dofs = np.array([d for d in range(n_dof)
                           if d not in set(eq_map[left_nodes, 0]) | set(eq_map[left_nodes, 1])])

    u = np.zeros(n_dof)
    step_log = []
    residual_histories = {}
    u_step1 = None
    converged_step1 = False
    for step in range(1, N_STEPS + 1):
        load_factor = TOTAL_FORCE_FINAL * step / N_STEPS
        f_ext = f_unit_eq * load_factor

        converged = False
        n_iter = 0
        res_norm0 = None
        last_K_ff = None
        res_history = []
        for it in range(1, MAX_NEWTON_ITER + 1):
            n_iter = it
            K, rhs_int = assembler.Assemble(u)  # rhs_int = -f_int
            residual = rhs_int + f_ext
            res_free = residual[free_dofs]
            res_norm = np.linalg.norm(res_free)
            res_history.append(res_norm)
            if res_norm0 is None:
                res_norm0 = max(res_norm, 1e-12)
            if verbose and diagnose:
                print(f"      iter {it:2d}  |res|={res_norm:.6e}  |res|/|res0|={res_norm / res_norm0:.6e}")
            if res_norm < RESIDUAL_ABS_TOL or res_norm / res_norm0 < RESIDUAL_REL_TOL:
                converged = True
                break

            K_ff = K[free_dofs, :][:, free_dofs]
            last_K_ff = K_ff
            try:
                du_free = spsolve(K_ff.tocsc(), res_free)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  [{which}] step {step}: linear solve failed at iter {it}: {exc}")
                break
            if not np.all(np.isfinite(du_free)):
                if verbose:
                    print(f"  [{which}] step {step}: non-finite update at iter {it}")
                break

            alpha = 1.0
            if use_line_search:
                alpha = _line_search_alpha(assembler, u, du_free, free_dofs, f_ext)
                if verbose and diagnose:
                    print(f"        line search alpha = {alpha:.4f}")
            u[free_dofs] += alpha * du_free

        if not converged and last_K_ff is not None:
            from scipy.sparse.linalg import eigsh
            try:
                eigval = eigsh(last_K_ff.tocsc(), k=1, which="SA", return_eigenvectors=False)
                if verbose:
                    print(f"      smallest eigenvalue of K_ff at last iterate: {eigval[0]:.6e}")
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"      eigsh failed: {exc}")

        residual_histories[step] = np.array(res_history)

        if step == 1:
            u_step1 = u.copy()
            converged_step1 = converged

        if not converged and diagnose:
            _rank_one_check(assembler, which)

        step_log.append({"step": step, "load": load_factor, "iters": n_iter, "converged": converged})
        if verbose:
            tag = "OK" if converged else "**DID NOT CONVERGE**"
            print(f"  [{which}] step {step:2d}  load={load_factor:.3e}  iters={n_iter:2d}  {tag}")
        if not converged and not force_continue:
            break

    fom.SetDisplacementFromEquationVector(u, eq_map, ta)
    e_voigt, s_voigt = assembler.ComputeStrainStressOnly(u)
    # ComputeStrainStressOnly writes into the assembler's own reusable buffers and
    # returns views into them, not copies -- the later step-1 call below would
    # otherwise silently overwrite this "final state" data before it is saved.
    e_flat = e_voigt.reshape(-1, 3).copy()
    s_flat = s_voigt.reshape(-1, 3).copy()

    assembler.Assemble(u)  # repopulate assembler._F at the reported (final/stalled) state
    rank_one_samples = _rank_one_check(assembler, which) if diagnose else np.zeros(0)
    if diagnose:
        state = "final converged" if step_log[-1]["converged"] else "last (stalled)"
        print(f"      (rank-one check above was at the {state} state)")

    tip_uy = [mp.GetNode(nid + 1).GetSolutionStepValue(KM.DISPLACEMENT_Y) for nid in right_nodes]
    u_nodal = np.stack([u[eq_map[:, 0]], u[eq_map[:, 1]]], axis=1)

    fully_converged = all(s["converged"] for s in step_log) and len(step_log) == N_STEPS

    # Step-1 (5% of final load) snapshot, captured for every model regardless of
    # whether the run continued past it -- lets non-converged models (which stop
    # here) be compared, at the same load level, against the converged reference.
    e_voigt_1, s_voigt_1 = assembler.ComputeStrainStressOnly(u_step1)
    e_flat_1 = e_voigt_1.reshape(-1, 3).copy()
    s_flat_1 = s_voigt_1.reshape(-1, 3).copy()
    u_nodal_step1 = np.stack([u_step1[eq_map[:, 0]], u_step1[eq_map[:, 1]]], axis=1)
    tip_uy_step1 = u_step1[eq_map[right_nodes, 1]]

    if save_npz:
        np.savez(
            f"cook_results_{which}{name_suffix}_claude.npz",
            coords=coords, tris=tris, u_nodal=u_nodal,
            e_gp=e_flat, s_gp=s_flat,
            iters_per_step=np.array([s["iters"] for s in step_log]),
            converged_per_step=np.array([s["converged"] for s in step_log]),
            load_per_step=np.array([s["load"] for s in step_log]),
            residual_history_step1=residual_histories.get(1, np.zeros(0)),
            residual_history_last=residual_histories.get(step_log[-1]["step"], np.zeros(0)),
            rank_one_samples=rank_one_samples,
            fully_converged=fully_converged,
            u_nodal_step1=u_nodal_step1,
            tip_uy_step1=tip_uy_step1,
            e_gp_step1=e_flat_1,
            s_gp_step1=s_flat_1,
            converged_step1=converged_step1,
        )

    return {
        "which": which,
        "step_log": step_log,
        "e11_range": (float(e_flat[:, 0].min()), float(e_flat[:, 0].max())),
        "e22_range": (float(e_flat[:, 1].min()), float(e_flat[:, 1].max())),
        "gamma12_range": (float(e_flat[:, 2].min()), float(e_flat[:, 2].max())),
        "tip_uy_range": (float(np.min(tip_uy)), float(np.max(tip_uy))),
        "fully_converged": fully_converged,
        "rank_one_samples": rank_one_samples,
    }


if __name__ == "__main__":
    results = {}
    for which in ("certified", "free", "ickan", "regression"):
        print(f"=== {which} ===")
        results[which] = run_newton(which, nx=16, ny=16, diagnose=True, use_line_search=True, save_npz=True)
        print()

    print("=== summary ===")
    for which, res in results.items():
        print(f"{which:10s}: fully_converged={res['fully_converged']}, "
              f"E11={res['e11_range']}, gamma12={res['gamma12_range']}, "
              f"tip_uy={res['tip_uy_range']}")
