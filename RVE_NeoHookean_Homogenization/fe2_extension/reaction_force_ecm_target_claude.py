#!/usr/bin/env python3
"""Track B, Stage 1: the per-element reaction-force integrand c_e.

Decomposes pann/direct_energy/reaction_force_direct_stress.py's global
reaction-force macro-stress into an exact per-element sum, so a new
empirical-cubature (MAW-ECM) rule can target it directly instead of the
naive volume average the project's existing Z_sigma rule targets.

Derivation (verified by direct code reading, not new physics): the global
reaction-force formula is

    stress[k] = -(sens[k, :] @ rhs[dir_dofs]) / denom

and rhs is built by core/fom_solver_rve.py's VectorizedAssembler as a pure
scatter-sum of the pre-scatter per-element local internal-force array
_f_int (shape (n_elems, n_nodes, 2), set at the end of Assemble()/
ComputeLocalArrays()):

    rhs[g] = -sum_{(e,p): local_eq_ids[e,p]==g} f_int_flat[e,p]

Substituting and swapping the order of summation (each element contributes
only through its own local Dirichlet-connected positions; elements that
touch no Dirichlet dof contribute exactly zero) gives, for element e:

    c_e[k] = (1/denom) * sum_p sens_local[k,e,p] * f_int_flat[e,p]

with sum_e c_e[k] == stress[k] exactly (linearity of scatter-sum + dot
product -- to floating-point precision, not necessarily bit-identical,
since a re-ordered sum of the same real numbers is a different but equally
valid summation order). This is verified directly below against
DirectStressGenerator's own already-validated output, on real trajectory
data, before being trusted for any offline dataset construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIRECT_ENERGY_DIR = HERE.parent / "pann" / "direct_energy"
if str(DIRECT_ENERGY_DIR) not in sys.path:
    sys.path.insert(0, str(DIRECT_ENERGY_DIR))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from reaction_force_direct_stress import DirectStressGenerator  # noqa: E402


def build_dof_to_dirpos(gen: "DirectStressGenerator") -> np.ndarray:
    """(n_dof,) int64, -1 for non-Dirichlet dofs, else index into gen.dir_dofs.
    Mirrors hprom/ann/hprom_ann_solver_rve.py's _build_free_dof_index_map,
    keyed on dir_dofs instead of free_dofs."""
    dof_to_dirpos = -np.ones(gen.n_dof, dtype=np.int64)
    dof_to_dirpos[gen.dir_dofs] = np.arange(gen.dir_dofs.size, dtype=np.int64)
    return dof_to_dirpos


def per_element_reaction_force_contribution(
    gen: "DirectStressGenerator",
    dof_to_dirpos: np.ndarray,
    u: np.ndarray,
    e: np.ndarray,
    heps: float = 1.0e-6,
) -> np.ndarray:
    """(n_elems, 3) c_e such that c_e.sum(axis=0) == the global reaction-force
    stress at this exact state. Calls gen.assembler.Assemble(u) itself (not
    reused from a prior call), matching direct_stress_history's own per-state
    pattern."""
    assembler = gen.assembler
    _, _rhs = assembler.Assemble(np.asarray(u, dtype=float).reshape(-1))

    local_dirpos = dof_to_dirpos[assembler.local_eq_ids]  # (n_elems, n_local_dof)
    valid = local_dirpos >= 0

    f_int_flat = assembler._f_int.reshape(assembler.n_elems, -1)  # (n_elems, n_local_dof)
    sens = gen.dirichlet_strain_sensitivity(np.asarray(e, dtype=float).reshape(3), heps=heps)  # (3, n_dir)
    denom = gen.thickness * gen.A0

    c_e = np.zeros((assembler.n_elems, 3), dtype=float)
    for k in range(3):
        sk = np.zeros_like(f_int_flat)
        sk[valid] = sens[k, local_dirpos[valid]]
        c_e[:, k] = np.sum(sk * f_int_flat, axis=1) / denom
    return c_e


def verify_against_global(
    gen: "DirectStressGenerator",
    dof_to_dirpos: np.ndarray,
    U: np.ndarray,
    applied_strain: np.ndarray,
    heps: float = 1.0e-6,
) -> np.ndarray:
    """For every state in U/applied_strain, confirm c_e.sum(axis=0) matches
    DirectStressGenerator.direct_stress_history's own already-validated
    output at that same state. Returns the per-state relative L2 error
    array; raises if any exceeds a tight tolerance."""
    n = U.shape[0]
    global_ref = gen.direct_stress_history(U, applied_strain, heps=heps)
    errs = np.zeros(n, dtype=float)
    for i in range(n):
        c_e = per_element_reaction_force_contribution(
            gen, dof_to_dirpos, U[i], applied_strain[i], heps=heps
        )
        summed = c_e.sum(axis=0)
        num = np.linalg.norm(summed - global_ref[i])
        den = max(np.linalg.norm(global_ref[i]), 1.0e-30)
        errs[i] = num / den
    return errs


if __name__ == "__main__":
    import time

    REPO_ROOT = HERE.parent
    TRAJ_DIR = REPO_ROOT / "trajectories" / "stage_1_training_set_fom" / "trajectory_1"
    U = np.load(TRAJ_DIR / "trajectory_1_U.npy")
    applied_strain = np.load(TRAJ_DIR / "trajectory_1_applied_strain.npy")

    # A handful of well-spread real states, not all 16802 -- this is a
    # correctness gate, not a full sweep (Stage 2's dataset build will cover
    # many more states with the same function).
    idx = np.linspace(0, U.shape[0] - 1, 12, dtype=int)

    gen = DirectStressGenerator()
    dof_to_dirpos = build_dof_to_dirpos(gen)

    t0 = time.perf_counter()
    errs = verify_against_global(gen, dof_to_dirpos, U[idx], applied_strain[idx])
    gen.close()
    dt = time.perf_counter() - t0

    print(f"[reaction-force-ecm-target] checked {idx.size} real states from trajectory 1 "
          f"in {dt:.2f}s")
    print(f"[reaction-force-ecm-target] per-state relative L2 error (c_e.sum vs global): "
          f"{errs}")
    print(f"[reaction-force-ecm-target] max error = {errs.max():.3e}")
    assert errs.max() < 1.0e-9, "per-element decomposition does not reproduce the global stress"
    print("[reaction-force-ecm-target] PASSED: sum_e c_e == global reaction-force stress "
          "to floating-point precision.")
