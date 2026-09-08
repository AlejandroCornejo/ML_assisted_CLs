#!/usr/bin/env python3
"""How many of the network's 36 slave outputs are worth keeping?

Measured, not assumed. Per-mode relative errors on the training pairs show 20
of 36 modes with error above 1.0 -- the network predicts them WORSE than
predicting zero. Those modes do not merely fail to help: they INJECT noise.
Slave mode 36 has amplitude 7.9e-07 of mode 1 and relative error 114, so its
prediction is 114x its true size, i.e. ~9e-05 of injected garbage in mode-1
units; summed over 20 such modes that is of the same order as the observed
end-to-end error of 6.4e-04.

So zeroing the tail may IMPROVE accuracy, not just simplify. This sweeps how
many leading slave outputs to keep and reports the end-to-end homogenized
stress error, on the in-envelope test states and the out-of-envelope probe.
No retraining: the network is fixed and its tail outputs are simply discarded.
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

KEEPS = (0, 3, 6, 10, 13, 16, 20, 26, 36)
N_EVAL = 120

b = np.load(HERE / "decoder_basis_B_r39.npz")
d = np.load(ROOT / "03_data" / "data.npz")
m = np.load(HERE / "nslave.npz")
Phi_M, Phi_S, A_M = b["Phi_M"], b["Phi_S"], b["A_M"]
mu_m, mu_s = m["mu_mean"], m["mu_std"]

net = build_net(3, Phi_S.shape[1], width=int(m["width"]), depth=int(m["depth"]))
net.load_state_dict({k: torch.from_numpy(m[k]) for k in net.state_dict()})
net.eval()

from periodic_fom import PeriodicRVE  # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402

rng = np.random.default_rng(3)
with true_neo_hookean_active():
    rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"), cell_area=float(d["cell_area"]))
    sets = {}
    for nm in ("test", "probe"):
        E_s, S_s = d[f"E_{nm}"], d[f"S_{nm}"]
        ok = np.isfinite(S_s).all(axis=1)
        E_s, S_s = E_s[ok], S_s[ok]
        idx = rng.choice(E_s.shape[0], min(N_EVAL, E_s.shape[0]), replace=False)
        sets[nm] = (E_s[idx], S_s[idx])
    with torch.no_grad():
        preds = {nm: net(torch.from_numpy((sets[nm][0] - mu_m) / mu_s)).numpy()
                 for nm in sets}
    print(f"{'keep':>5} | {'test median':>12} {'test p90':>11} | "
          f"{'probe median':>13} {'probe p90':>11}")
    rows = []
    for k in KEEPS:
        res = {}
        for nm in sets:
            E_s, S_s = sets[nm]
            errs = []
            for i in range(E_s.shape[0]):
                qs = np.zeros(Phi_S.shape[1])
                if k:
                    qs[:k] = preds[nm][i, :k]
                u_ind = Phi_M @ (A_M @ E_s[i]) + Phi_S @ qs
                rve.assembler.Assemble(rve.T @ u_ind + rve._g(E_s[i]))
                S = rve.homogenized_stress(E_s[i])
                errs.append(np.linalg.norm(S - S_s[i]) / np.linalg.norm(S_s[i]))
            res[nm] = np.array(errs)
        rows.append((k, res))
        print(f"{k:5d} | {np.median(res['test']):12.4e} "
              f"{np.percentile(res['test'],90):11.4e} | "
              f"{np.median(res['probe']):13.4e} "
              f"{np.percentile(res['probe'],90):11.4e}", flush=True)
    best = min(rows, key=lambda r: np.median(r[1]["test"]))
    print(f"\nbest in-envelope median at keep = {best[0]}: "
          f"{np.median(best[1]['test']):.4e}")
    print("TRUNCATE_SWEEP_DONE")
