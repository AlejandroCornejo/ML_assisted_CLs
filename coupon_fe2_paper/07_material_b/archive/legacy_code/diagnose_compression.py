#!/usr/bin/env python3
"""Cold-start continuation diagnostic, keeping the original pilot failure."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon-b-mpl")
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_rve"))
import periodic_fom as pf
from _material_law_guard_claude import true_neo_hookean_active

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mesh", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
rows = []
with true_neo_hookean_active():
    rve = pf.PeriodicRVE(args.mesh.resolve(), cell_area=4.0)
    for density in (200.0, 400.0, 800.0):
        pf.SUBSTEPS_PER_UNIT_STRAIN = density
        for e11 in (-0.005, -0.01, -0.02, -0.04):
            row = dict(density=density, strain=[e11, 0.0, 0.0], ok=False)
            start = time.perf_counter()
            try:
                stress, q = rve.solve(row["strain"])
                row.update(ok=True, stress=stress.tolist(), energy=rve.homogenized_energy(),
                           min_micro_J=float(np.linalg.det(rve.assembler._F).min()))
            except Exception as exc:
                row["error"] = repr(exc)
            row["seconds"] = time.perf_counter()-start
            rows.append(row)
            args.out.write_text(json.dumps(dict(mesh=str(args.mesh), rows=rows), indent=2) + "\n")
            print(density, e11, row["ok"], row.get("error", ""), flush=True)
