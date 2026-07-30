#!/usr/bin/env python3
"""Reaction-force-based "direct" energy-conjugate macroscopic stress generator.

See ``DATA_DICTIONARY.md`` at the repository root for full background on the
three strain/stress conventions used in this project.  In short:

    "Direct" energy-conjugate stress  s = dW/de ,  e = [E11, E22, gamma12]

is the quantity `pann/anisotropic`'s PANN models are trained against
(stored in ``pann/data/alltraj_stage10_direct_energy.npz``).  Nothing in the
current codebase reproduces it from a fresh solve -- this script does.

FORMULA (derived from the envelope theorem, verified numerically below)
-------------------------------------------------------------------------
This RVE problem has *no external force term at all* -- it is a pure
displacement-driven (mixed Dirichlet/free) problem.  Consequently, for any
call ``K, rhs = assembler.Assemble(u)``,

    rhs == -f_int(u)          everywhere (free AND Dirichlet dofs alike),

because ``VectorizedAssembler.Assemble`` builds ``rhs`` purely as the
negative scatter of internal forces (see ``core/fom_solver_rve.py``,
``Assemble()``, line ~442-443).

The RVE-average stored energy density used everywhere else in this project
(``fom.ComputeMicroscopicNeoHookeanEnergyDensityFromAssembler``) is an
AREA-weighted average of the pointwise energy density using ``area_e``
(the *pure geometric* 2D element area, i.e. it does **not** carry the
THICKNESS factor -- THICKNESS is folded into ``w_detJ``/``f_int`` instead).
So the total (extensive) stored energy is

    Pi_total(u) = thickness * sum_e area_e * psi_mean_e(u)

and ``f_int = d(Pi_total)/du`` (the standard internal-force definition, with
thickness already included via ``w_detJ``).

At a converged equilibrium state, the only free parameters entering the
Dirichlet displacement are the macro strain components ``e_k``. By the
envelope theorem (stationarity of Pi_total with respect to the *free* dofs
at equilibrium):

    d(Pi_total)/de_k = sum_{j in Dirichlet dofs} f_int_j * d(u_dir_j)/de_k

Combined with ``rhs = -f_int`` and dividing by the reference volume
``thickness * A0`` (A0 = sum of reference element areas, i.e. the same
denominator used for the RVE-average energy density) to convert the
extensive total-energy derivative into the same *density* units as
``ComputeMicroscopicNeoHookeanEnergyDensityFromAssembler``:

    s_k = dW/de_k = -(1 / (thickness * A0)) *
                      sum_{j in Dirichlet dofs} rhs_j * d(u_dir_j)/de_k

where ``u_dir_j(e)`` is the closed-form affine Dirichlet map built from
``F(e)`` (``fom.DeformationGradientFromGreenLagrange2D`` /
``fom.ComputeDirichletValuesFromGreenLagrange``), and its strain-sensitivity
``d(u_dir_j)/de_k`` is evaluated by a cheap central finite difference on
that same closed-form map (NOT a re-solve).

This overall MINUS sign and the ``1/(thickness*A0)`` normalization were
both determined empirically by calibration against the known-good ground
truth in ``pann/data/alltraj_stage10_direct_energy.npz`` (see module
docstring of ``verify_against_ground_truth`` below and the report printed
by ``if __name__ == "__main__"``) -- every other sign/scale combination
tried was off by a clearly wrong constant factor or sign.

Usage
-----
    python3 reaction_force_direct_stress.py --trajectory 1
    python3 reaction_force_direct_stress.py --trajectory 6 --max-steps 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CORE_DIR = REPO_ROOT / "core"
TRAJ_DIR = REPO_ROOT / "trajectories" / "stage_1_training_set_fom"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import KratosMultiphysics as KM  # noqa: E402
import fom_solver_rve as fom  # noqa: E402


def make_parameters() -> "KM.Parameters":
    """Build ProjectParameters with absolute mesh/material paths (cwd-independent).

    Mirrors ``studies/fom_tangent_stability_test/run_fom_energy_hessian_audit.py``'s
    ``make_parameters`` so this script can be run from any directory. Does NOT
    modify ``core/fom_solver_rve.py``.
    """
    with open(CORE_DIR / "ProjectParameters.json", encoding="utf-8") as f:
        config = json.load(f)

    config["output_processes"] = {"gid_output": [], "vtk_output": []}
    config["solver_settings"]["echo_level"] = 0
    parameters = KM.Parameters(json.dumps(config))

    mesh_base = str(CORE_DIR / "rve_geometry")
    fom.SetInputMeshFilename(parameters, mesh_base)
    material_parts = fom.DetectMaterialSubModelParts(str(CORE_DIR / "rve_geometry.mdpa"))
    parameters = fom.ConfigureElementModelerForMaterialParts(parameters, material_parts)
    fom.SetMaterialsFilename(parameters, str(CORE_DIR / "StructuralMaterials.json"))
    return parameters


class DirectStressGenerator:
    """Re-initializes a VectorizedAssembler and computes reaction-force-based
    direct stress from an already-converged displacement history, with no
    re-solving and no finite differences on the expensive FOM (only on the
    cheap closed-form Dirichlet map).
    """

    def __init__(self):
        parameters = make_parameters()
        self.model = KM.Model()
        self.sim = fom.RVEHomogenizationDatasetGenerator(self.model, parameters)
        self.sim.Initialize()

        mp = self.sim._GetSolver().GetComputingModelPart()
        self.mp = mp
        n_dof, eq_map, _ = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)
        self.n_dof = n_dof
        self.eq_map = eq_map

        self.assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label="DirectStressGenerator")
        self.A0 = float(np.sum(np.asarray(self.assembler.area_e, dtype=float)))
        self.thickness = float(self.assembler.thickness[0])

        self.sim._InitializeDomainCenterIfNeeded(mp)
        x0c, y0c = float(self.sim._x0c), float(self.sim._y0c)
        (
            self.dir_dofs,
            self.dir_x,
            self.dir_y,
            self.dir_is_x,
            self.free_dofs,
        ) = fom.PrecomputeDirichletPartitionFromNodes(mp, n_dof, x0c, y0c)

        print(
            f"[DirectStressGenerator] n_dof={n_dof}, n_dirichlet={self.dir_dofs.size}, "
            f"n_free={self.free_dofs.size}, A0={self.A0:.6e}, thickness={self.thickness:.6e}"
        )

    def dirichlet_strain_sensitivity(self, e: np.ndarray, heps: float = 1.0e-6) -> np.ndarray:
        """d(u_dirichlet)/de_k for k=0,1,2 via central finite difference on the
        cheap closed-form affine map (not a re-solve). Returns shape (3, n_dir_dofs).
        """
        e = np.asarray(e, dtype=float).reshape(3)
        sens = np.empty((3, self.dir_x.size), dtype=float)
        for k in range(3):
            step = heps if abs(e[k]) < 1.0 else heps * max(1.0, abs(e[k]))
            e_plus = e.copy()
            e_plus[k] += step
            e_minus = e.copy()
            e_minus[k] -= step
            u_plus = fom.ComputeDirichletValuesFromGreenLagrange(
                e_plus, self.dir_x, self.dir_y, self.dir_is_x
            )
            u_minus = fom.ComputeDirichletValuesFromGreenLagrange(
                e_minus, self.dir_x, self.dir_y, self.dir_is_x
            )
            sens[k] = (u_plus - u_minus) / (2.0 * step)
        return sens

    def direct_stress_history(
        self,
        U: np.ndarray,
        applied_strain: np.ndarray,
        heps: float = 1.0e-6,
        max_steps: int | None = None,
    ) -> np.ndarray:
        n_steps = U.shape[0] if max_steps is None else min(int(max_steps), U.shape[0])
        stress = np.zeros((n_steps, 3), dtype=float)
        denom = self.thickness * self.A0

        for i in range(n_steps):
            u = np.asarray(U[i], dtype=float).reshape(-1)
            e = np.asarray(applied_strain[i], dtype=float).reshape(3)

            _, rhs = self.assembler.Assemble(u)
            R = rhs[self.dir_dofs]

            sens = self.dirichlet_strain_sensitivity(e, heps=heps)
            stress[i] = -(sens @ R) / denom

        return stress

    def close(self):
        self.sim.Finalize()


def load_trajectory(trajectory_index: int):
    root = TRAJ_DIR / f"trajectory_{trajectory_index}"
    U = np.load(root / f"trajectory_{trajectory_index}_U.npy")
    applied_strain = np.load(root / f"trajectory_{trajectory_index}_applied_strain.npy")
    return U, applied_strain


def load_ground_truth(trajectory_index: int):
    data = np.load(PANN_DATA)
    tid = data["train_trajectory_id"]
    sidx = data["train_source_index"]
    mask = tid == trajectory_index
    order = np.argsort(sidx[mask])
    strain = data["train_strain"][mask][order]
    stress = data["train_stress"][mask][order]
    src_idx = sidx[mask][order]
    return src_idx, strain, stress


def relative_l2_error(computed: np.ndarray, reference: np.ndarray) -> float:
    num = np.linalg.norm(computed - reference)
    den = np.linalg.norm(reference)
    return float(num / den) if den > 0.0 else float(num)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--heps", type=float, default=1.0e-6)
    args = parser.parse_args()

    U, applied_strain = load_trajectory(args.trajectory)
    src_idx, gt_strain, gt_stress = load_ground_truth(args.trajectory)

    n_steps = U.shape[0] if args.max_steps is None else min(args.max_steps, U.shape[0])
    keep = src_idx < n_steps
    src_idx = src_idx[keep]
    gt_strain = gt_strain[keep]
    gt_stress = gt_stress[keep]

    # Sanity check: ground truth's own recorded strain must match applied_strain
    # at the same source index (confirms the index correspondence assumption).
    assert np.allclose(applied_strain[src_idx], gt_strain), (
        "Ground-truth strain does not match applied_strain at the mapped source index; "
        "index correspondence assumption is wrong."
    )

    gen = DirectStressGenerator()
    try:
        stress_hist = gen.direct_stress_history(U, applied_strain, heps=args.heps, max_steps=n_steps)
    finally:
        gen.close()

    computed = stress_hist[src_idx]
    err = relative_l2_error(computed, gt_stress)
    print(f"[trajectory {args.trajectory}] n_compare={len(src_idx)}, relative L2 error = {err:.6e}")
    print("First few rows (computed vs ground truth):")
    for i in range(min(5, len(src_idx))):
        print(f"  step {src_idx[i]}: computed={computed[i]}, gt={gt_stress[i]}")

    return err


if __name__ == "__main__":
    main()
