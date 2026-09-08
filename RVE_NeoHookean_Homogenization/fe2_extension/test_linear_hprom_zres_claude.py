#!/usr/bin/env python3
"""Direct empirical test of the Z_res/w_res_full rule built from the
"degenerate-target" residual-projection dataset (linear_residual_z_res_rule_
claude.npz), run through the REAL online solver (RunHpromBatchSimulation) on
the held-out test trajectory -- settles empirically whether the near-zero
ECM weight-fitting target actually breaks the hyper-reduced Newton solve, or
whether it still works because the Newton update itself is invariant to a
common rescaling of the per-element weights (only a fixed-absolute
convergence check would be affected in principle).

No reduced HROM mesh built -- ResolveResidualHyperReductionSelection supports
Z_res/w_res_full directly against the full mesh's own element list, so this
tests the rule as directly as possible before investing in any mesh-building
step. homogenization_method="kratos_reference" sidesteps needing any eps/sig
weights for this test (exact, un-reduced sync assemble each step).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CORE_DIR = REPO_ROOT / "core"
POD_DIR = REPO_ROOT / "pod" / "stage_2_pod_rve"
STAGE10_DIR = REPO_ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
OUT_DIR = HERE / "linear_hprom_zres_test_claude"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM  # noqa: E402
from fom_solver_rve import setup_kratos_parameters  # noqa: E402
from stage6_test_hprom import generate_safe_test_path  # noqa: E402
from hprom_solver_rve import RunHpromBatchSimulation  # noqa: E402


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def main() -> None:
    phi_f = np.load(POD_DIR / "pod_basis_free.npy")
    free_dofs = np.load(POD_DIR / "free_dofs.npy")
    dir_dofs = np.load(POD_DIR / "dirichlet_dofs.npy")
    eq_map = np.load(POD_DIR / "eq_map.npy")
    Xc, Yc = np.load(POD_DIR / "domain_center.npy")

    rule = np.load(HERE / "linear_residual_z_res_rule_claude.npz")
    Z_res = np.asarray(rule["Z_res"], dtype=np.int64)
    w_res_full = np.asarray(rule["w_res_full"], dtype=float)
    print(f"[test-zres] Z_res size={Z_res.size}, w_res_full nonzero={np.count_nonzero(w_res_full)}, "
          f"weight range=[{w_res_full[Z_res].min():.3e}, {w_res_full[Z_res].max():.3e}]")

    ecm_data = {
        "Z_res": Z_res,
        "w_res_full": w_res_full,
        "Z_union": Z_res,
        "n_elem": np.array([int(rule["n_elem"])]),
        "hrom_full_mesh_base": np.array([str(HERE / "rve_geometry")]),
    }

    bundle_path = REPO_ROOT / "trajectories" / "stage_0_trajectory" / "stage_0_trajectories.npz"
    data = np.load(bundle_path, allow_pickle=True)
    rel6 = list(data["relative_boundary"])
    emax = float(np.ravel(data["emax"])[0]) if "emax" in data else float(np.ravel(data["reference_amplitude"])[0])
    domain_type = str(data["domain_type"][0]) if "domain_type" in data else "box"
    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)
    print(f"[test-zres] test trajectory: {len(strain_path)} waypoints, emax={emax}")

    parameters = setup_kratos_parameters(str(HERE / "rve_geometry"))

    strain_hist, stress_hist = RunHpromBatchSimulation(
        parameters, phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc,
        ecm_data=ecm_data,
        out_dir=str(OUT_DIR),
        save_plot=False,
        strain_path=strain_path,
        trajectory_index="test_traj",
        reference_amplitude=emax,
        homogenization_method="kratos_reference",
    )
    print(f"[test-zres] strain_hist shape={strain_hist.shape}, stress_hist shape={stress_hist.shape}")

    fom_eps = np.load(STAGE10_DIR / "fom_strain.npy")
    fom_sig = np.load(STAGE10_DIR / "fom_stress.npy")
    n = min(strain_hist.shape[0], fom_eps.shape[0])
    err_eps = relative_l2(strain_hist[:n], fom_eps[:n])
    err_sig = relative_l2(stress_hist[:n], fom_sig[:n])
    print(f"[test-zres] relative L2 strain error vs FOM (naive-average) = {err_eps:.4%}")
    print(f"[test-zres] relative L2 stress error vs FOM (naive-average) = {err_sig:.4%}")
    for k, comp in enumerate(["xx", "yy", "xy"]):
        e = relative_l2(stress_hist[:n, k], fom_sig[:n, k])
        print(f"  sigma_{comp} relative L2 error = {e:.4%}")


if __name__ == "__main__":
    main()
