#!/usr/bin/env python3
"""How much does each slave mode matter for the HOMOGENIZED STRESS?

This decides the loss design for N_slave, and it is not guessable. The slave
modes span four orders of magnitude in amplitude (mode 4 at sigma/sigma_1 =
1.3e-2 down to mode 39 at ~1e-6), so a plain absolute loss will fit mode 4 and
treat mode 39 as noise. Whether that is fine or fatal depends on whether the
small modes move the stress at all.

Measured by truncating the reconstruction

    d_red(k) = Phi_M A_M q_M + Phi_S[:, :k] q_S[:k]

for k = 0 .. 36 slaves, evaluating the homogenized stress at that
(non-equilibrium) displacement exactly as the ROM would, and comparing with
the converged FOM stress. No Newton is involved: the point is what the
reconstruction alone delivers.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(ROOT / "00_rve"),
          str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.append("/home/kratos/Kratos_Eigen_Check/bin/Release")

N_STATES = 30
KS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 22, 28, 36)

d = np.load(ROOT / "03_data" / "data.npz")
b = np.load(HERE / "decoder_basis_B_r39.npz")
E_all, S_all, U_all = d["E_train"], d["S_train"], d["U_train"]
ok = np.isfinite(U_all).all(axis=1) & np.isfinite(S_all).all(axis=1)
E_all, S_all = E_all[ok], S_all[ok]
Phi_M, Phi_S, A_M, q_M, q_S = b["Phi_M"], b["Phi_S"], b["A_M"], b["q_M"], b["q_S"]
sv = b["sv"]

rng = np.random.default_rng(11)
idx = rng.choice(E_all.shape[0], N_STATES, replace=False)

from periodic_fom import PeriodicRVE
from _material_law_guard_claude import true_neo_hookean_active
with true_neo_hookean_active():
    rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"), cell_area=float(d["cell_area"]))
    base = Phi_M @ (A_M @ q_M)          # (n_ind, n_snap)
    print(f"{'slaves k':>9} {'sigma_k/sigma_1':>16} {'median |dS|/|S|':>16} {'p90':>10}")
    for k in KS:
        errs = []
        for i in idx:
            u_ind = base[:, i] + (Phi_S[:, :k] @ q_S[:k, i] if k else 0.0)
            rve.assembler.Assemble(rve.T @ u_ind + rve._g(E_all[i]))
            S = rve.homogenized_stress(E_all[i])
            errs.append(np.linalg.norm(S - S_all[i]) / np.linalg.norm(S_all[i]))
        errs = np.array(errs)
        skk = sv[3 + k - 1] / sv[0] if k else float("nan")
        print(f"{k:9d} {skk:16.3e} {np.median(errs):16.3e} {np.percentile(errs,90):10.3e}",
              flush=True)
