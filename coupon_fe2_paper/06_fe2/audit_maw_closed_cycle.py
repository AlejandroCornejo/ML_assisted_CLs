"""Closed-cycle check of the frozen iterative HPROM--ANN, not a timing run.

Reuses the published neural-model cycle geometry and Gauss rule. Equilibrium
is re-solved at each point with continuation; no network or support is fitted.
The optional tightened tolerance is process-local and leaves solver files intact.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon-cycle-mpl")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / ".pydeps"))
sys.path.insert(0, str(ROOT / "06_pann"))
import numpy as np
from mechanics_witnesses import rectangle_points
import maw_hprom_ann_fast as fast_module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(law, centre, half_width, order, reverse=False):
    points, increments = rectangle_points(centre, half_width, order)
    if reverse:
        points, increments = points[::-1], -increments[::-1]
    q, previous = None, None
    stresses, counts, residuals = [], [], []
    counter = [0]
    assembler = law.residual.asm
    original = assembler.ComputeLocalArrays

    def counted(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)

    assembler.ComputeLocalArrays = counted
    try:
        for E in points:
            before = counter[0]
            q = law.solve(E, q_init=q, E_start=previous)
            counts.append(counter[0] - before)
            stresses.append(law._stress_from_state(E, q))
            residuals.append(float(np.linalg.norm(law._residual_state(E, q))))
            previous = E.copy()
    finally:
        assembler.ComputeLocalArrays = original
    stress = np.asarray(stresses)
    contributions = np.einsum("ij,ij->i", stress, increments)
    assert np.isfinite(stress).all()
    return {
        "signed_work_J_per_m3": float(contributions.sum()),
        "absolute_accumulated_work_J_per_m3": float(np.abs(contributions).sum()),
        "max_stress_Pa": float(np.max(np.linalg.norm(stress, axis=1))),
        "micro_iterations_total_including_initial_continuation": int(sum(counts)),
        "micro_iterations_first_query": counts[0],
        "micro_iterations_subsequent_queries_min_median_max": [
            int(min(counts[1:])), float(np.median(counts[1:])), int(max(counts[1:]))],
        "max_reduced_residual_norm": max(residuals),
        "gauss_queries": len(points),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", default="8,16,32,64,128")
    parser.add_argument("--output", type=Path, default=HERE / "maw_closed_cycle_audit.json")
    args = parser.parse_args()
    orders = [int(x) for x in args.orders.split(",")]
    assert orders and min(orders) >= 2
    source = ROOT / "06_pann/mechanics_witness_results/current_rve_cycle_audit.json"
    cycle = json.loads(source.read_text())["cycle"]
    centre = np.array(cycle["centre_E"])
    half_width = cycle["half_width_in_E11_and_E22"]
    files = [source, ROOT / "04_training/decoder_basis_B_r39.npz",
             ROOT / "04_training/nslave.npz", ROOT / "05_validation/maw_res_long10.npz",
             ROOT / "05_validation/maw_phase2_sig.npz", ROOT / "03_data/rve_mesh.mdpa",
             HERE / "maw_hprom_ann_fast.py", HERE / "maw_hprom_ann_law.py",
             Path(__file__)]
    result = {
        "scope": "Frozen iterative HPROM--ANN cycle; diagnostic, not FE2 timing or architecture comparison",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "centre_E": centre.tolist(), "half_width_in_E11_and_E22": half_width,
        "orientation": "counter-clockwise; reversed control uses independent continuation from zero",
        "micro_displacement_increment_tolerance": fast_module.NEWTON_TOL,
        "iterations_note": "Counted residual ComputeLocalArrays calls, one per modified-Newton iteration; includes substeps. Not counts from archived FE2 runs.",
        "convergence": {},
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    with tempfile.TemporaryDirectory(prefix="coupon-maw-cycle-") as work:
        with contextlib.redirect_stdout(io.StringIO()):
            law = fast_module.FastMAWHPROMANN(work)
        for order in orders:
            row = evaluate(law, centre, half_width, order)
            result["convergence"][str(order)] = row
            save()
            print(f"order={order} " + json.dumps(row), flush=True)
        result["reverse_control"] = evaluate(law, centre, half_width, max(orders), reverse=True)
        result["reverse_control"]["order"] = max(orders)
        fast_module.NEWTON_TOL = 1e-12
        result["tight_tolerance_control"] = evaluate(law, centre, half_width, max(orders))
        result["tight_tolerance_control"].update(order=max(orders), tolerance=1e-12)
        result["status"] = "completed"
        save()
        print("MAW_CLOSED_CYCLE_COMPLETED " + str(args.output), flush=True)
        print(json.dumps({k:result[k] for k in ("reverse_control", "tight_tolerance_control")}), flush=True)


if __name__ == "__main__":
    main()
