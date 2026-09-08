#!/usr/bin/env python3
"""Post-hoc reaction-force correction for the linear HPROM's already-run
test-trajectory result (test_linear_hprom_zres_claude.py's own output,
linear_hprom_zres_test_claude/trajectory_test_traj_q.npy) -- same "oracle"
pattern already used this session for ICNN/D-HPROM-ANN/HPROM-ANN: reconstruct
full displacement vectors from the reduced coordinates and the known affine
Dirichlet map, then run the already-validated DirectStressGenerator on them.
No new HPROM run needed -- reuses the already-computed q history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CORE_DIR = REPO_ROOT / "core"
POD_DIR = REPO_ROOT / "pod" / "stage_2_pod_rve"
Q_DIR = HERE / "linear_hprom_zres_test_claude"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM  # noqa: E402
from fom_solver_rve import (  # noqa: E402
    DeformationGradientFromGreenLagrange2D,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from stage6_test_hprom import generate_safe_test_path  # noqa: E402
from reaction_force_ecm_target_claude import DirectStressGenerator  # noqa: E402


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def main() -> None:
    free_dofs = np.load(POD_DIR / "free_dofs.npy").astype(np.int64)
    dir_dofs = np.load(POD_DIR / "dirichlet_dofs.npy").astype(np.int64)
    eq_map = np.load(POD_DIR / "eq_map.npy").astype(np.int64)
    Xc, Yc = np.load(POD_DIR / "domain_center.npy")
    phi_f = np.load(POD_DIR / "pod_basis_free.npy")
    n_dof = int(eq_map.max()) + 1

    q_hist = np.load(Q_DIR / "trajectory_test_traj_q.npy")
    print(f"[apply-rf] q_hist shape={q_hist.shape}")

    bundle_path = REPO_ROOT / "trajectories" / "stage_0_trajectory" / "stage_0_trajectories.npz"
    data = np.load(bundle_path, allow_pickle=True)
    rel6 = list(data["relative_boundary"])
    emax = float(np.ravel(data["emax"])[0]) if "emax" in data else float(np.ravel(data["reference_amplitude"])[0])
    domain_type = str(data["domain_type"][0]) if "domain_type" in data else "box"
    _, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    E_wp = np.array(waypoints, dtype=float)
    n_seg = len(E_wp) - 1
    seg_steps, _ = BuildDynamicSegmentSteps(
        E_wp, reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT, reference_amplitude=emax,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    assert n_steps_total + 1 == q_hist.shape[0], (n_steps_total, q_hist.shape)

    gen = DirectStressGenerator()
    # Sanity check: gen's own internally-consistent free/dir dof partition
    # must match the POD basis's own (as SETS) -- confirms the eq_map used
    # below to build dof_x/dof_y/is_x_dof is aligned with gen.mp's own model.
    assert set(gen.free_dofs.tolist()) == set(free_dofs.tolist()), "free_dofs mismatch between DirectStressGenerator and POD basis"
    assert set(gen.dir_dofs.tolist()) == set(dir_dofs.tolist()), "dir_dofs mismatch between DirectStressGenerator and POD basis"
    print("[apply-rf] free/dir dof partition verified consistent with DirectStressGenerator's own model.")

    # dof geometry, mirroring RunHpromBatchSimulation's own affine-displacement setup
    x0c, y0c = float(Xc), float(Yc)
    dof_x = np.zeros(n_dof, dtype=float)
    dof_y = np.zeros(n_dof, dtype=float)
    is_x_dof = np.zeros(n_dof, dtype=bool)
    for i, node in enumerate(gen.mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        idx_x, idx_y = int(eq_map[i, 0]), int(eq_map[i, 1])
        dof_x[idx_x], dof_y[idx_x], is_x_dof[idx_x] = xr, yr, True
        dof_x[idx_y], dof_y[idx_y], is_x_dof[idx_y] = xr, yr, False

    E_t_all = np.zeros((n_steps_total + 1, 3), dtype=float)
    for step in range(1, n_steps_total + 1):
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        E_t_all[step] = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]

    print(f"[apply-rf] reconstructed applied-strain path: {E_t_all.shape}")

    # Full displacement reconstruction: u_free = u_aff_free + phi_f @ q,
    # u_dir = u_aff_dir (affine Dirichlet map) -- both via the closed-form
    # DeformationGradientFromGreenLagrange2D map, evaluated at each dof's own
    # reference coordinate relative to the domain center.
    U_full = np.zeros((n_steps_total + 1, n_dof), dtype=float)
    x_free, y_free, is_x_free = dof_x[free_dofs], dof_y[free_dofs], is_x_dof[free_dofs]
    x_dir, y_dir, is_x_dir = dof_x[dir_dofs], dof_y[dir_dofs], is_x_dof[dir_dofs]

    for step in range(n_steps_total + 1):
        F = DeformationGradientFromGreenLagrange2D(E_t_all[step])
        ux_f = (F[0, 0] - 1.0) * x_free + F[0, 1] * y_free
        uy_f = F[1, 0] * x_free + (F[1, 1] - 1.0) * y_free
        U_full[step, free_dofs] = np.where(is_x_free, ux_f, uy_f) + phi_f @ q_hist[step]

        ux_d = (F[0, 0] - 1.0) * x_dir + F[0, 1] * y_dir
        uy_d = F[1, 0] * x_dir + (F[1, 1] - 1.0) * y_dir
        U_full[step, dir_dofs] = np.where(is_x_dir, ux_d, uy_d)

    rf_stress = gen.direct_stress_history(U_full, E_t_all)
    gen.close()

    pann = np.load(PANN_DATA)
    ref_stress = np.asarray(pann["stage10_stress"], dtype=float)
    n = min(rf_stress.shape[0], ref_stress.shape[0])
    err = relative_l2(rf_stress[:n], ref_stress[:n])
    print(f"[apply-rf] relative L2 reaction-force stress error vs ground truth = {err:.4%}")
    for k, comp in enumerate(["xx", "yy", "xy"]):
        e = relative_l2(rf_stress[:n, k], ref_stress[:n, k])
        print(f"  sigma_{comp} relative L2 error = {e:.4%}")

    np.savez(HERE / "linear_hprom_reaction_force_comparison_claude.npz",
             rf_stress=rf_stress[:n], ref_stress=ref_stress[:n], err=err)


if __name__ == "__main__":
    main()
