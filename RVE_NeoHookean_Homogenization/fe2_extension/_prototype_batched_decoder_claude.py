#!/usr/bin/env python3
"""Proof of concept: is the decoder's own forward+Jacobian evaluation
(confirmed by profiling to be ~60% of DHpromAnnDirectLawFloat64's own
per-Gauss-point wall time) faster if batched across Gauss points via
torch.func.vmap, instead of one torch.func.jacfwd call per point in a
Python loop? Compares against the CURRENT, deployed code path
(evaluate_with_tangent's own per-point jacfwd), on the SAME real macro
strains from the n_body=6 cruciform run, and checks the batched result
matches the current one to floating-point precision -- vmap changes
only how the computation is scheduled, not what it computes, but this
is checked directly rather than assumed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dhprom_ann_direct_law_float64_claude import DHpromAnnDirectLawFloat64  # noqa: E402

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"

d = np.load(HERE / "cruciform_results_dhprom_f64_consistent_claude.npz")
e_gp = d["e_gp"]
print(f"[proto] loaded {e_gp.shape[0]} real macro strains", flush=True)

law = DHpromAnnDirectLawFloat64(hprom_ann_dir=str(DHPROMANN_DIR))
_ = law.evaluate_with_tangent(e_gp[0])  # warm-up

N_SAMPLE = 300
rng = np.random.default_rng(0)
sample_idx = rng.choice(e_gp.shape[0], size=min(N_SAMPLE, e_gp.shape[0]), replace=False)
E_batch = e_gp[sample_idx]

mu_dim = int(law.qp_aff["mu_dim"])
b_aff = np.asarray(law.qp_aff["b_aff"], dtype=float)


def qp_from_E(E):
    mu = E[:mu_dim]
    return np.concatenate([mu, [1.0]]) @ b_aff


# ---- CURRENT path: one jacfwd call per point, in a Python loop ----
def current_path_one(E):
    q_p = qp_from_E(E)
    q_p_torch = torch.from_numpy(q_p.astype(np.float64)).reshape(1, -1).to(law.device)
    q_in = q_p_torch.reshape(-1).clone().detach()

    def ann_from_qvec(qvec):
        return law.ann_model(qvec.view(1, -1)).reshape(-1)

    with torch.no_grad():
        q_s_final_map = law.ann_model(q_p_torch)
    J_dec_torch = torch.func.jacfwd(ann_from_qvec)(q_in).reshape(law.n_secondary, law.n_primary).detach()
    return q_s_final_map.detach().cpu().numpy().reshape(-1), J_dec_torch.cpu().numpy()


t0 = time.perf_counter()
current_out = [current_path_one(E) for E in E_batch]
t1 = time.perf_counter()
t_current = t1 - t0
print(f"[proto] CURRENT (loop of single jacfwd calls): {t_current:.3f}s for {len(E_batch)} points "
      f"({t_current/len(E_batch)*1000:.3f} ms/point)", flush=True)

# ---- BATCHED path: one vmap(jacfwd(...)) call for the whole batch ----
Q_batch = np.stack([qp_from_E(E) for E in E_batch], axis=0)
Q_batch_t = torch.from_numpy(Q_batch.astype(np.float64)).to(law.device)


def ann_single(qvec):
    return law.ann_model(qvec.view(1, -1)).reshape(-1)


t0 = time.perf_counter()
with torch.no_grad():
    q_s_final_map_batch = law.ann_model(Q_batch_t)
J_dec_batch = torch.func.vmap(torch.func.jacfwd(ann_single))(Q_batch_t).detach().cpu().numpy()
t1 = time.perf_counter()
t_batched = t1 - t0
print(f"[proto] BATCHED (single vmap(jacfwd) call): {t_batched:.3f}s for {len(E_batch)} points "
      f"({t_batched/len(E_batch)*1000:.3f} ms/point)", flush=True)
print(f"[proto] speedup: {t_current/max(t_batched,1e-9):.1f}x", flush=True)

# ---- correctness check ----
q_s_final_map_batch_np = q_s_final_map_batch.detach().cpu().numpy()
max_map_err, max_jac_err = 0.0, 0.0
for i in range(len(E_batch)):
    map_cur, jac_cur = current_out[i]
    max_map_err = max(max_map_err, float(np.max(np.abs(map_cur - q_s_final_map_batch_np[i]))))
    max_jac_err = max(max_jac_err, float(np.max(np.abs(jac_cur - J_dec_batch[i]))))
print(f"[proto] max abs diff, decoder forward output: {max_map_err:.3e}", flush=True)
print(f"[proto] max abs diff, decoder Jacobian:        {max_jac_err:.3e}", flush=True)
print("PROTOTYPE_DONE_MARKER", flush=True)
