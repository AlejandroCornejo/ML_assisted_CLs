#!/usr/bin/env python3
"""Cross-check B against independent linear assembly and native Kratos elements."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon-b-mpl")
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_rve"))
import periodic_C0 as linear
import periodic_fom as pf
from _material_law_guard_claude import true_neo_hookean_active
import KratosMultiphysics as KM


def sparse_relative(first, second):
    difference = first-second
    return float(np.linalg.norm(difference.data)/np.linalg.norm(second.data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    directory = args.directory.resolve()
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "pilot complete":
        raise RuntimeError("Wait for the full pilot before running this verification")
    rows = []
    six_points = linear.GAUSS.copy()
    # linear.assemble multiplies weights by 1/2: these weights sum to one.
    three_points = np.array([[1/6, 1/6, 1/3], [2/3, 1/6, 1/3], [1/6, 2/3, 1/3]])
    with true_neo_hookean_active():
        for mesh in report["meshes"]:
            name = mesh["name"]
            geometry = np.load(directory / (name + ".npz"))
            xy, triangles = geometry["xy"], geometry["triangles"]
            target = np.array(mesh["reference"]["tangent"])
            row = dict(mesh=name)
            for label, gauss in (("three_point", three_points), ("six_point", six_points)):
                linear.GAUSS = gauss
                c0, _meta = linear.solve_C0(xy, triangles, "periodic", report["spec"]["cell_side"]/2)
                row[label + "_C0"] = c0.tolist()
                row[label + "_relative_difference"] = float(np.linalg.norm(c0-target)/np.linalg.norm(target))
            rve = pf.PeriodicRVE(directory / name, cell_area=report["spec"]["cell_side"]**2)
            a = rve.assembler
            # Reconstruct equation-order displacement from saved node-order data.
            endpoint = np.load(directory / (name + "_combined_x_endpoint.npz"))
            u = np.empty(rve.n_dof)
            u[rve._eq_map] = endpoint["node_displacement"]
            K_vec, R_vec = a.Assemble(u)
            ta = KM.TensorAdaptors.HistoricalVariableTensorAdaptor(rve._mp.Nodes, KM.DISPLACEMENT, [2])
            ta.Check()
            rve._fom.SetDisplacementFromEquationVector(u, rve._eq_map, ta)
            rve._fom.UpdateCurrentCoordinatesFromDisplacement(rve._mp)
            K_native, R_native = rve._fom.AssembleGlobalSystem(rve._mp, rve.n_dof)
            row["native_stiffness_relative_difference"] = sparse_relative(K_vec, K_native)
            row["native_force_relative_difference"] = float(np.linalg.norm(R_vec-R_native)/np.linalg.norm(R_native))
            row["passed"] = bool(row["three_point_relative_difference"] < 1e-8
                and row["six_point_relative_difference"] < .01
                and row["native_stiffness_relative_difference"] < 1e-8
                and row["native_force_relative_difference"] < 1e-8)
            rows.append(row)
            args.out.write_text(json.dumps(dict(rows=rows,
                status="complete" if len(rows) == len(report["meshes"]) else "partial",
                passed=len(rows) == len(report["meshes"]) and all(r["passed"] for r in rows),
                scope="Same-mesh code/quadrature cross-check, not an independent physical experiment. "
                "Three-point linear/native comparisons use 1e-8 relative tolerance; six-point "
                "quadrature sensitivity uses a 1% screening tolerance."), indent=2) + "\n")
            print(name, {k:v for k,v in row.items() if "difference" in k or k == "passed"}, flush=True)
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
