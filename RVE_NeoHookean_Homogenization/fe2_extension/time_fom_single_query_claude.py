#!/usr/bin/env python3
"""Timing probe, not a paper artifact: how long does ONE cold-start,
full-order (no ROM, no ECM/MAW hyper-reduction, plain 990-element RVE
mesh) equilibrium solve at a single imposed macro strain E actually take?
Needed to scope how many query strains an offline PANN/PROM/HPROM-vs-FOM
accuracy comparison could afford.

Mirrors studies/fom_tangent_stability_test/run_fom_energy_hessian_audit.py
's solve_endpoint() pattern (2-waypoint path [[0,0,0], E], so
RunFomBatchSimulation ramps from the trivial zero-strain state up to E
over reference_amplitude/reference_steps-controlled increments) --
core/fom_solver_rve.py itself has no dedicated single-query
SolveAtStrain(E) entry point; RunFomBatchSimulation is a trajectory
solver reused here for one query.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE_DIR = ROOT / "core"
for p in (str(CORE_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM  # noqa: E402
import fom_solver_rve as fom  # noqa: E402


def make_parameters():
    with open(str(CORE_DIR / "ProjectParameters.json")) as f:
        config = json.load(f)
    config["output_processes"] = {"gid_output": [], "vtk_output": []}
    parameters = KM.Parameters(json.dumps(config))
    fom.SetInputMeshFilename(parameters, str(CORE_DIR / "rve_geometry"))
    mats = fom.DetectMaterialSubModelParts(str(CORE_DIR / "rve_geometry.mdpa"))
    parameters = fom.ConfigureElementModelerForMaterialParts(parameters, mats)
    fom.SetMaterialsFilename(parameters, str(CORE_DIR / "StructuralMaterials.json"))
    return parameters


def solve_at_strain(E, reference_amplitude=2.0, reference_steps=400, out_dir=None):
    out_dir = out_dir or str(HERE / "fom_query_scratch")
    strain_hist, stress_hist = fom.RunFomBatchSimulation(
        parameters=make_parameters(), out_dir=out_dir,
        save_results=False, save_plot=False,
        strain_path=np.vstack(([0.0, 0.0, 0.0], E)), trajectory_index=1,
        reference_amplitude=reference_amplitude, reference_steps=reference_steps,
    )
    return np.asarray(strain_hist[-1]), np.asarray(stress_hist[-1])


if __name__ == "__main__":
    # A representative, moderately large Cook nx=8 Gauss-point strain (near
    # the upper end of what was actually observed: |E| ~ 0.05), as a
    # conservative (not best-case) single-query timing estimate.
    E_query = np.array([0.01173723, 0.00614076, 0.04820225])
    print(f"[time-fom] cold-start query at E={E_query}, |E|={np.linalg.norm(E_query):.4e}")
    t0 = time.perf_counter()
    eps_final, sig_final = solve_at_strain(E_query, reference_amplitude=2.0, reference_steps=400)
    elapsed = time.perf_counter() - t0
    print(f"[time-fom] eps_final={eps_final}, sig_final={sig_final}")
    print(f"[time-fom] ELAPSED = {elapsed:.2f}s for one cold-start query "
          f"(reference_amplitude=2.0, reference_steps=400)")
