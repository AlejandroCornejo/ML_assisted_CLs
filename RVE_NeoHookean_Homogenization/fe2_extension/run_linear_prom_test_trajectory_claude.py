#!/usr/bin/env python3
"""Test the pure linear PROM (POD-Galerkin, no ANN, no hyper-reduction) on
the same held-out test trajectory used for every other Table-6 comparison
this session (HPROM-ANN, D-HPROM-ANN, ICNN, etc.).

Per the user's own decision: before tackling the degenerate-residual-target
problem found while building a Z_res ECM rule for a hyper-reduced linear
model (see build_linear_residual_ecm_dataset_claude.py's own findings), first
check whether the linear POD basis phi_f (39 modes, no ANN nonlinear-manifold
correction) reconstructs the RVE response well at all, using the
already-existing, already-working RunPromBatchSimulation (prom/pod/
prom_solver_rve.py) -- full Galerkin projection, no ECM, no hyper-reduction.

Comparison convention: naive volume-average homogenized stress/strain (same
convention on both sides -- PROM's own CalculateHomogenizedFromAssemblerWithElementWeights
with no weights, and hprom/ann/stage_10_results_maw_dynamic/fom_*.npy, which
was generated with the same convention). This isolates "does the linear basis
reconstruct the displacement field well" from the separate reaction-force-
convention question already resolved elsewhere this session.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CORE_DIR = REPO_ROOT / "core"
POD_SOLVER_DIR = REPO_ROOT / "prom" / "pod"
POD_DIR = REPO_ROOT / "pod" / "stage_2_pod_rve"
STAGE10_DIR = REPO_ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
OUT_DIR = HERE / "linear_prom_test_traj_results_claude"

for p in (CORE_DIR, POD_SOLVER_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM  # noqa: E402
from fom_solver_rve import setup_kratos_parameters  # noqa: E402
from stage6_test_hprom import generate_safe_test_path  # noqa: E402
from prom_solver_rve import RunPromBatchSimulation  # noqa: E402


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def main() -> None:
    phi_f = np.load(POD_DIR / "pod_basis_free.npy")
    free_dofs = np.load(POD_DIR / "free_dofs.npy")
    dir_dofs = np.load(POD_DIR / "dirichlet_dofs.npy")
    eq_map = np.load(POD_DIR / "eq_map.npy")
    Xc, Yc = np.load(POD_DIR / "domain_center.npy")
    print(f"[linear-prom-test] phi_f shape={phi_f.shape}")

    # Exactly reproduces stage10_test_hprom_ann.py's own test-path generation
    # (same emax/rel6/domain_type, read from the same stage_0_trajectories.npz),
    # so this is the SAME held-out trajectory already used for every other
    # Table-6 comparison this session.
    bundle_path = REPO_ROOT / "trajectories" / "stage_0_trajectory" / "stage_0_trajectories.npz"
    data = np.load(bundle_path, allow_pickle=True)
    rel6 = list(data["relative_boundary"])
    emax = float(np.ravel(data["emax"])[0]) if "emax" in data else float(np.ravel(data["reference_amplitude"])[0])
    domain_type = str(data["domain_type"][0]) if "domain_type" in data else "box"
    print(f"[linear-prom-test] emax={emax}, rel6={rel6}, domain_type={domain_type}")

    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)
    print(f"[linear-prom-test] waypoints: {len(control_points)}, strain path points: {len(strain_path)}")

    parameters = setup_kratos_parameters(str(HERE / "rve_geometry"))

    strain_hist, stress_hist = RunPromBatchSimulation(
        parameters, phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc,
        out_dir=str(OUT_DIR),
        save_plot=False,
        strain_path=strain_path,
        trajectory_index="test_traj",
        reference_amplitude=emax,
    )
    print(f"[linear-prom-test] strain_hist shape={strain_hist.shape}, stress_hist shape={stress_hist.shape}")

    fom_eps = np.load(STAGE10_DIR / "fom_strain.npy")
    fom_sig = np.load(STAGE10_DIR / "fom_stress.npy")
    print(f"[linear-prom-test] FOM ground truth shape: eps={fom_eps.shape}, sig={fom_sig.shape}")

    n = min(strain_hist.shape[0], fom_eps.shape[0])
    err_eps = relative_l2(strain_hist[:n], fom_eps[:n])
    err_sig = relative_l2(stress_hist[:n], fom_sig[:n])
    print(f"[linear-prom-test] relative L2 strain error (naive-average convention) = {err_eps:.4%}")
    print(f"[linear-prom-test] relative L2 stress error (naive-average convention) = {err_sig:.4%}")

    for k, comp in enumerate(["xx", "yy", "xy"]):
        e = relative_l2(stress_hist[:n, k], fom_sig[:n, k])
        print(f"  sigma_{comp} relative L2 error = {e:.4%}")

    np.savez(
        OUT_DIR / "comparison_vs_fom_claude.npz",
        strain_hist=strain_hist, stress_hist=stress_hist,
        fom_eps=fom_eps[:n], fom_sig=fom_sig[:n],
        err_eps=err_eps, err_sig=err_sig,
    )
    print(f"[linear-prom-test] saved comparison to {OUT_DIR / 'comparison_vs_fom_claude.npz'}")


if __name__ == "__main__":
    main()
