#!/usr/bin/env python3
"""Verify the 237-element reduced HROM mesh reproduces the same accuracy as
the full-mesh Z_res/w_res_full test (test_linear_hprom_zres_claude.py) --
same test trajectory, same ecm_data content (now via w_res_hrom on the
actual reduced mesh instead of Z_res/w_res_full on the full one).
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
OUT_DIR = HERE / "linear_hprom_reduced_mesh_test_claude"
MESH_DIR = HERE / "linear_hprom_zres_hrom_mesh"

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

    ecm = np.load(MESH_DIR / "ecm_weights_all.npz", allow_pickle=True)
    ecm_data = {k: ecm[k] for k in ecm.files}
    mesh_base = str(np.ravel(ecm_data["hrom_mesh_base"])[0])
    print(f"[test-reduced-mesh] mesh_base={mesh_base}")
    print(f"[test-reduced-mesh] w_res_hrom size={ecm_data['w_res_hrom'].size}, "
          f"nonzero={np.count_nonzero(ecm_data['w_res_hrom'])}")

    bundle_path = REPO_ROOT / "trajectories" / "stage_0_trajectory" / "stage_0_trajectories.npz"
    data = np.load(bundle_path, allow_pickle=True)
    rel6 = list(data["relative_boundary"])
    emax = float(np.ravel(data["emax"])[0]) if "emax" in data else float(np.ravel(data["reference_amplitude"])[0])
    domain_type = str(data["domain_type"][0]) if "domain_type" in data else "box"
    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)

    parameters = setup_kratos_parameters(mesh_base)

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
    print(f"[test-reduced-mesh] strain_hist shape={strain_hist.shape}, stress_hist shape={stress_hist.shape}")

    fom_eps = np.load(STAGE10_DIR / "fom_strain.npy")
    fom_sig = np.load(STAGE10_DIR / "fom_stress.npy")
    n = min(strain_hist.shape[0], fom_eps.shape[0])
    err_eps = relative_l2(strain_hist[:n], fom_eps[:n])
    err_sig = relative_l2(stress_hist[:n], fom_sig[:n])
    print(f"[test-reduced-mesh] relative L2 strain error vs FOM (naive-average) = {err_eps:.4%}")
    print(f"[test-reduced-mesh] relative L2 stress error vs FOM (naive-average) = {err_sig:.4%}")
    for k, comp in enumerate(["xx", "yy", "xy"]):
        e = relative_l2(stress_hist[:n, k], fom_sig[:n, k])
        print(f"  sigma_{comp} relative L2 error = {e:.4%}")


if __name__ == "__main__":
    main()
