#!/usr/bin/env python3
"""Isolated diagnostic: before trusting LinearHpromIterativeLawFloat64 in
any driver, call it directly (verbose=True) on a handful of small,
reasonable macro strains to see per-iteration residual behavior, timing,
and whether it converges cleanly -- and inspect the suspicious near-zero
"total area" printed by its own res_assembler construction (237 elements,
total area ~1.1e-10 vs the full 258-element mesh's own ~4.9e-2) to check
whether the residual ECM weights (w_res_local) are sane."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import linear_hprom_iterative_law_float64_claude as m  # noqa: E402

law = m.LinearHpromIterativeLawFloat64(verbose=True)

print(f"\n[diag] n_primary={law.n_primary}, n_current_elements={law.n_current_elements}, "
      f"res_assembler n_elems={law.res_assembler.n_elems}", flush=True)

ecm = np.load(Path(m.DEFAULT_LAW_MESH_DIR) / "ecm_weights_all.npz", allow_pickle=True)
w_res_full = np.asarray(ecm["w_res_full"], dtype=float).reshape(-1)
print(f"[diag] w_res_full: shape={w_res_full.shape}, min={w_res_full.min():.3e}, "
      f"max={w_res_full.max():.3e}, mean={w_res_full.mean():.3e}, "
      f"n_nonzero={np.count_nonzero(w_res_full)}", flush=True)
print(f"[diag] w_res_full sample values: {w_res_full[:10]}", flush=True)

for E in [
    np.array([0.001, 0.0, 0.0]),
    np.array([0.01, 0.0, 0.0]),
    np.array([0.05, 0.0, 0.0]),
]:
    print(f"\n[diag] === evaluate(E={E.tolist()}) ===", flush=True)
    t0 = time.perf_counter()
    hom_eps, hom_sig, q_p, n_it, converged = law.evaluate(E)
    t1 = time.perf_counter() - t0
    print(f"[diag] wall={t1:.3f}s, n_it={n_it}, converged={converged}, "
          f"hom_sig={hom_sig}, |q_p|={np.linalg.norm(q_p):.3e}", flush=True)

print("DIAG_DONE_MARKER", flush=True)
