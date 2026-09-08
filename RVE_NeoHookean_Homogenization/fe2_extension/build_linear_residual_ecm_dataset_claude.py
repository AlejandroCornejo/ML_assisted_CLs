#!/usr/bin/env python3
"""Linear-HPROM residual-projection ECM dataset builder.

Ports RVE_homogenization_NeoHookean_using_Kratos/stage5_build_ecm_dataset.py's
Q_ecm/b_full construction (q_e = V_e^T @ r_e, the element-local residual
projected onto the linear POD free-dof basis phi_f) into this self-contained
project, replacing its per-element elem.CalculateRightHandSide() Kratos calls
(990 calls/state) with core/fom_solver_rve.py's VectorizedAssembler's own
already-computed, already-verified _f_int array (one vectorized Assemble()
call/state, plus one dense einsum) -- same underlying quantity, much faster.

Sign convention: VectorizedAssembler.Assemble() sets
rhs[g] = -sum_{(e,p)} f_int_flat[e,p] (scatter-sum, confirmed by direct code
reading of fom_solver_rve.py). So the element-local RHS r_e used here is
-f_int_flat[e], matching what the old project's elem.CalculateRightHandSide()
returned per element (Kratos's own residual convention). This is exactly the
quantity core/hprom_solver_rve.py's AssembleHyperReducedSystem later
approximates online: phi_f^T @ rhs_full[free_dofs] = sum_e q_e (over ALL
elements, verified below to floating-point precision before any offline
sampling is trusted).

No homogenization dataset (C_hom/b_hom) built here -- the linear HPROM this
feeds uses homogenization_method="kratos_reference" (an exact, un-reduced
sync assemble already built into RunHpromBatchSimulation), so no separate
eps/sig MAW-ECM rule is needed for this experiment.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CORE_DIR = REPO_ROOT / "core"
POD_DIR = REPO_ROOT / "pod" / "stage_2_pod_rve"
TRAJ_DIR = REPO_ROOT / "trajectories" / "stage_1_training_set_fom"
OUT_NPZ = HERE / "linear_residual_ecm_dataset_claude.npz"

for p in (HERE, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM  # noqa: E402
from fom_solver_rve import (  # noqa: E402
    setup_kratos_parameters,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    VectorizedAssembler,
)
from ecm_sampling_utils import get_param_aware_indices  # noqa: E402

N_TRAJECTORIES = 10
# Matches build_reaction_force_ecm_dataset_claude.py's own convention (5% per
# trajectory, param_aware sampling) for consistency across this session's two
# ECM datasets.
SNAPSHOT_PERCENT_RES = 5.0
PARAM_AWARE_TIME_WEIGHT = 0.2
PARAM_AWARE_SEED = 42


def build_free_map(n_dof: int, free_dofs: np.ndarray) -> np.ndarray:
    map_g2f = -np.ones(n_dof, dtype=np.int64)
    map_g2f[free_dofs] = np.arange(free_dofs.size, dtype=np.int64)
    return map_g2f


def build_dense_projection_operator(assembler, map_g2f: np.ndarray, phi_f: np.ndarray):
    """(n_elems, n_local_dof, nq) dense array Ve_full such that, for the full
    local RHS r_e_full (all local positions, Dirichlet ones included),
    q_e = Ve_full[e].T @ r_e_full -- Dirichlet-local positions contribute
    zero because Ve_full is zero there, so no masking is needed downstream.
    """
    local_pos = map_g2f[assembler.local_eq_ids]  # (n_elems, n_local_dof)
    valid = local_pos >= 0
    n_elems, n_local_dof = local_pos.shape
    nq = phi_f.shape[1]

    rows_safe = np.where(valid, local_pos, 0)
    gathered = phi_f[rows_safe.reshape(-1), :].reshape(n_elems, n_local_dof, nq)
    Ve_full = np.where(valid[:, :, None], gathered, 0.0)
    n_active = int(np.count_nonzero(np.any(valid, axis=1)))
    return Ve_full, n_active


def main() -> None:
    phi_f = np.load(POD_DIR / "pod_basis_free.npy")
    free_dofs = np.load(POD_DIR / "free_dofs.npy").astype(np.int64)
    eq_map_ref = np.load(POD_DIR / "eq_map.npy").astype(np.int64)
    nq = phi_f.shape[1]
    print(f"[build-linear-res] phi_f shape={phi_f.shape} (nq={nq}), free_dofs={free_dofs.size}")

    parameters = setup_kratos_parameters(str(HERE / "rve_geometry"))
    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    n_dof, eq_map, _ = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    elements = list(mp.Elements)
    n_elem = len(elements)
    print(f"[build-linear-res] mesh: {n_elem} elements, {mp.NumberOfNodes()} nodes, n_dof={n_dof}")

    # Verification gate: freshly-built eq_map must match the one phi_f was
    # computed against, or the row-gather below would silently misalign.
    if not np.array_equal(eq_map, eq_map_ref):
        raise RuntimeError(
            "[build-linear-res] freshly-built eq_map does not match "
            "pod/stage_2_pod_rve/eq_map.npy -- refusing to proceed with a "
            "possibly-misaligned basis projection."
        )
    print("[build-linear-res] eq_map verification: PASSED (exact match).")

    assembler = VectorizedAssembler(mp, n_dof, eq_map, log_label="LinearResAssembler")
    map_g2f = build_free_map(n_dof, free_dofs)
    Ve_full, n_active = build_dense_projection_operator(assembler, map_g2f, phi_f)
    print(f"[build-linear-res] residual projection active on {n_active}/{n_elem} elements.")

    # --- sample training states (same param-aware scheme as the reaction-force dataset) ---
    tasks = []
    total_states = 0
    for traj in range(1, N_TRAJECTORIES + 1):
        root = TRAJ_DIR / f"trajectory_{traj}"
        u_file = root / f"trajectory_{traj}_U.npy"
        e_file = root / f"trajectory_{traj}_applied_strain.npy"
        U_meta = np.load(u_file, mmap_mode="r")
        E_meta = np.load(e_file, mmap_mode="r")
        n_steps = min(int(U_meta.shape[0]), int(E_meta.shape[0]))
        n_pick = int(np.ceil((SNAPSHOT_PERCENT_RES / 100.0) * n_steps))
        idx = get_param_aware_indices(
            np.asarray(E_meta[:n_steps, :3], dtype=float), n_pick,
            seed=PARAM_AWARE_SEED + traj, time_weight=PARAM_AWARE_TIME_WEIGHT,
        )
        idx = np.unique(np.asarray(idx, dtype=int).reshape(-1))
        tasks.append((traj, u_file, idx))
        total_states += idx.size
        print(f"  [trajectory {traj}] n_steps={n_steps}, picked {idx.size} states")

    print(f"[build-linear-res] total training states: {total_states} -> Q_ecm rows = {nq * total_states}")

    Q_ecm = np.zeros((nq * total_states, n_elem), dtype=float)
    b_full = np.zeros(nq * total_states, dtype=float)

    s_global = 0
    t0 = time.perf_counter()
    for traj, u_file, idx in tasks:
        U_all = np.load(u_file, mmap_mode="r")
        for k in idx:
            u_snap = np.asarray(U_all[int(k), :], dtype=float)
            assembler.Assemble(u_snap)
            r_full = -assembler._f_int.reshape(assembler.n_elems, -1)  # (n_elems, n_local_dof)
            q_block = np.einsum("epq,ep->qe", Ve_full, r_full)  # (nq, n_elems)

            r0, r1 = nq * s_global, nq * (s_global + 1)
            Q_ecm[r0:r1, :] = q_block
            b_full[r0:r1] = np.sum(q_block, axis=1)
            s_global += 1
        dt = time.perf_counter() - t0
        print(f"  [trajectory {traj}] done ({s_global}/{total_states} states so far, {dt:.1f}s elapsed)", flush=True)

    assert s_global == total_states

    # Verification gate: the per-element sum used to build b_full must match
    # phi_f^T @ rhs_full[free_dofs] computed directly from a full, unweighted
    # Assemble() -- a completely independent computation path -- at a few
    # spot-checked states. IMPORTANT: b_full is the projected residual on
    # FREE dofs at a CONVERGED FOM state -- by static equilibrium, this is
    # necessarily ~0 (to Newton-tolerance precision, verified separately to
    # be ~1e-13 relative to the Dirichlet-side reaction forces for this RVE,
    # which has zero external force everywhere). So the RELATIVE error check
    # used elsewhere in this session is meaningless here (dividing by a
    # near-zero norm) -- this gate instead checks the ABSOLUTE error against
    # the typical per-row magnitude of Q_ecm itself (the individual element
    # contributions, which do NOT cancel and are the actual quantity ECM
    # will select support points from).
    print("[build-linear-res] verification gate: q_block sum vs direct phi_f^T @ rhs[free_dofs]")
    typical_row_scale = float(np.median(np.linalg.norm(Q_ecm, axis=1)))
    max_abs_err = 0.0
    max_rel_to_b = 0.0
    check_states = [(tasks[0][1], tasks[0][2][0]), (tasks[-1][1], tasks[-1][2][-1])]
    for u_file, k in check_states:
        U_all = np.load(u_file, mmap_mode="r")
        u_snap = np.asarray(U_all[int(k), :], dtype=float)
        _, rhs = assembler.Assemble(u_snap)
        direct = phi_f.T @ rhs[free_dofs]
        r_full = -assembler._f_int.reshape(assembler.n_elems, -1)
        via_cache = np.einsum("epq,ep->q", Ve_full, r_full)
        abs_err = float(np.linalg.norm(via_cache - direct))
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_to_b = max(max_rel_to_b, abs_err / max(np.linalg.norm(direct), 1.0e-30))
    print(f"[build-linear-res] max ABSOLUTE error (per-element sum vs direct projection) = {max_abs_err:.3e}")
    print(f"[build-linear-res] typical Q_ecm row norm (individual element contributions) = {typical_row_scale:.3e}")
    print(f"[build-linear-res] b_full norm stats: median={np.median(np.abs(b_full)):.3e}, "
          f"max={np.max(np.abs(b_full)):.3e} (this is the degenerate near-zero equilibrium target)")
    print(f"[build-linear-res] max relative error vs b_full itself (noise-dominated, informational only) = {max_rel_to_b:.3e}")
    assert max_abs_err < 1.0e-6 * typical_row_scale, (
        "per-element residual projection does not reproduce phi_f^T @ rhs at the scale of "
        "individual element contributions -- this WOULD be a real bug (as opposed to the "
        "expected near-zero-target degeneracy from equilibrium, which is a modeling issue, not a bug)."
    )
    print("[build-linear-res] PASSED (absolute-scale check).")

    sim.Finalize()

    np.savez(
        OUT_NPZ,
        Q_ecm=Q_ecm, b_full=b_full, nq=nq, n_elem=n_elem, N_s_res=total_states,
        snapshot_percent_res=SNAPSHOT_PERCENT_RES,
        param_aware_time_weight=PARAM_AWARE_TIME_WEIGHT, seed=PARAM_AWARE_SEED,
    )
    print(f"[build-linear-res] saved Q_ecm {Q_ecm.shape}, b_full {b_full.shape} to {OUT_NPZ}")


if __name__ == "__main__":
    main()
