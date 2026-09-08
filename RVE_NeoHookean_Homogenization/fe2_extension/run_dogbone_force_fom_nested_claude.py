#!/usr/bin/env python3
"""FOM-FE^2 for the final dogbone setup: force-controlled (equal and
opposite tractions at both ends, F=1.0e9), improved mesh (552 elements,
size_far=2.0/size_near_fillet=0.5), gentle shoulder (W_gauge=4.0,
W_grip=4.25, R=2.0). The true, non-reduced 990-element RVE nested at every
one of the 552*3=1656 macro Gauss points -- the actual ground truth the
other 7 rows approximate, needed to settle whether the ICNN/ICKAN/Free
cluster or the D-HPROM/HPROM-iter/Linear-HPROM/Regression cluster is closer
to correct on E22 (the ~12-13% split found this session, traced to the
already-documented, already-applied small-strain tangent-term trade-off,
w_tan0=5). Expensive: with 2.76x the trained-mesh's own Gauss-point count
compared to the cruciform's own FOM-FE^2 run (600 GP, 11.1h), expect on the
order of a day or more with 16 workers."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_dogbone_force_fe2_claude import run_newton_fe2_dogbone_force  # noqa: E402

if __name__ == "__main__":
    print("=== fom_nested (F=1.0e9, W_gauge=4.0, W_grip=4.25, R=2.0, improved mesh) ===", flush=True)
    res = run_newton_fe2_dogbone_force(
        "fom_nested", W_gauge=4.0, W_grip=4.25, R=2.0,
        total_force_final=1.0e9, verbose=True, save_npz=True,
    )
    print(f"\n[fom_nested] fully_converged={res['fully_converged']}  ever_diverged={res['ever_diverged']}  "
          f"E11={res['e11_range']}  E22={res['e22_range']}  g12={res['g12_range']}  "
          f"wall={res['wall_time']:.1f}s", flush=True)
    print("DOGBONE_FORCE_FOM_NESTED_DONE_MARKER", flush=True)
