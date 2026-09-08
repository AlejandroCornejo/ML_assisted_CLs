#!/usr/bin/env python3
"""Attribute the ROM's error to its parts, rather than reporting one number.

An end-to-end stress error of 6.4e-04 could come from the POD basis failing to
represent the state, from the identification q_M = mu, or from the network's
regression -- and the three call for different responses. With the test and
probe snapshots now stored, they separate:

  (a) POD projection    ||d - Phi Phi^T d|| / ||d||         the BASIS
  (b) oracle stress     stress from the TRUE q_S            the basis, in stress terms
  (c) identification    stress from true q_S but q_M = mu   the T_m identity map
  (d) network stress    stress from q_S = N_ANN(mu)         the TOTAL

(b) vs (a) converts a displacement error into the quantity of interest.
(c) vs (b) isolates what setting q_M = mu costs, instead of projecting.
(d) vs (c) is what the network itself adds.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
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

import torch  # noqa: E402
from train_nslave import build_net  # noqa: E402

N_EVAL = 120
b = np.load(HERE / "decoder_basis_B_r39.npz")
d = np.load(ROOT / "03_data" / "data.npz")
ev = np.load(ROOT / "03_data" / "eval_snapshots.npz")
m = np.load(HERE / "nslave.npz")
Phi, Phi_M, Phi_S, A_M, T_m = b["Phi_ROM"], b["Phi_M"], b["Phi_S"], b["A_M"], b["T_m"]
mu_m, mu_s = m["mu_mean"], m["mu_std"]

net = build_net(3, Phi_S.shape[1], width=int(m["width"]), depth=int(m["depth"]))
net.load_state_dict({k: torch.from_numpy(m[k]) for k in net.state_dict()})
net.eval()

from periodic_fom import PeriodicRVE  # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402

rng = np.random.default_rng(3)
with true_neo_hookean_active():
    rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"), cell_area=float(d["cell_area"]))

    def stress_at(u_ind, E):
        rve.assembler.Assemble(rve.T @ u_ind + rve._g(E))
        return rve.homogenized_stress(E)

    print(f"{'set':>6} {'n':>4} | {'(a) POD proj':>13} {'(b) oracle S':>13} "
          f"{'(c) q_M=mu':>12} {'(d) network':>12}")
    for nm in ("test", "probe"):
        E_s, S_s, U_s = d[f"E_{nm}"], d[f"S_{nm}"], ev[f"U_{nm}"]
        ok = np.isfinite(S_s).all(1) & np.isfinite(U_s).all(1)
        E_s, S_s, U_s = E_s[ok], S_s[ok], U_s[ok]
        idx = rng.choice(E_s.shape[0], min(N_EVAL, E_s.shape[0]), replace=False)
        E_s, S_s, U_s = E_s[idx], S_s[idx], U_s[idx]
        D = U_s.T
        proj = np.linalg.norm(D - Phi @ (Phi.T @ D), axis=0) / np.linalg.norm(D, axis=0)
        qS_true = Phi_S.T @ D
        qM_proj = T_m @ (Phi.T @ D)
        with torch.no_grad():
            qS_net = net(torch.from_numpy((E_s - mu_m) / mu_s)).numpy().T
        eb, ec, ed = [], [], []
        for i in range(E_s.shape[0]):
            n0 = np.linalg.norm(S_s[i])
            ub = Phi_M @ (A_M @ qM_proj[:, i]) + Phi_S @ qS_true[:, i]
            uc = Phi_M @ (A_M @ E_s[i]) + Phi_S @ qS_true[:, i]
            ud = Phi_M @ (A_M @ E_s[i]) + Phi_S @ qS_net[:, i]
            eb.append(np.linalg.norm(stress_at(ub, E_s[i]) - S_s[i]) / n0)
            ec.append(np.linalg.norm(stress_at(uc, E_s[i]) - S_s[i]) / n0)
            ed.append(np.linalg.norm(stress_at(ud, E_s[i]) - S_s[i]) / n0)
        print(f"{nm:>6} {len(idx):4d} | {np.median(proj):13.4e} "
              f"{np.median(eb):13.4e} {np.median(ec):12.4e} {np.median(ed):12.4e}",
              flush=True)
        qerr = np.linalg.norm(qS_net - qS_true) / np.linalg.norm(qS_true)
        print(f"{'':>11} | q_S regression error {qerr:.4e}   "
              f"identification ||q_M(proj) - mu||/||mu|| "
              f"{np.linalg.norm(qM_proj - E_s.T) / np.linalg.norm(E_s):.4e}")
print("DECOMPOSITION_DONE")
