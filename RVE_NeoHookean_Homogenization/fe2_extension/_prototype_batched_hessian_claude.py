#!/usr/bin/env python3
"""Isolated prototype: does torch.func.hessian + torch.func.vmap reproduce
HpromAnnIterativeLawFloat64._compute_weighted_decoder_hessian (which uses
the older torch.autograd.functional.hessian, one point at a time) to
floating-point precision, and how much faster is it? Captures REAL
(qp_vec, output_weights) pairs from an actual real-strain sweep (via
monkeypatching), rather than using synthetic/random inputs, per this
project's own established discipline (see _prototype_batched_decoder_claude.py)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import hprom_ann_iterative_law_float64_claude as m  # noqa: E402

HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

law = m.HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR), qp_init_mode="continuation")

captured = []
orig_hessian = law._compute_weighted_decoder_hessian


def capturing_hessian(qp_vec, output_weights):
    captured.append((np.array(qp_vec, dtype=float), np.array(output_weights, dtype=float)))
    return orig_hessian(qp_vec, output_weights)


law._compute_weighted_decoder_hessian = capturing_hessian

N_POINTS = 60
rng = np.random.default_rng(0)
idx = rng.choice(e_gp.shape[0], size=N_POINTS, replace=False)
for i in idx:
    law.evaluate_with_tangent(e_gp[i], q_prev=np.zeros(law.n_primary), step_index=1)

print(f"[proto] captured {len(captured)} real (qp, weights) pairs from {N_POINTS} points "
      f"(mean {len(captured) / N_POINTS:.2f} hessian calls/point)", flush=True)

law._compute_weighted_decoder_hessian = orig_hessian  # restore

qp_batch = np.stack([c[0] for c in captured])
w_batch = np.stack([c[1] for c in captured])
print(f"[proto] qp_batch {qp_batch.shape}, w_batch {w_batch.shape}", flush=True)

# --- OLD: serial, one torch.autograd.functional.hessian call per point ---
t0 = time.perf_counter()
K_old = np.stack([law._compute_weighted_decoder_hessian(qp_batch[i], w_batch[i]) for i in range(len(captured))])
t_old = time.perf_counter() - t0
print(f"[proto] OLD (serial autograd.functional.hessian): {t_old:.4f}s, "
      f"{t_old / len(captured) * 1000:.4f} ms/point", flush=True)

# --- NEW: batched via torch.func.hessian + vmap ---
device = law.device
ann_model = law.ann_model


def weighted_ann_output(qvec, w):
    q_s_raw = ann_model(qvec.view(1, -1)).reshape(-1)
    return torch.dot(w, q_s_raw)


hessian_fn = torch.func.hessian(weighted_ann_output, argnums=0)
batched_hessian_fn = torch.func.vmap(hessian_fn, in_dims=(0, 0))


def compute_hessian_batch(qp_b, w_b):
    qp_t = torch.from_numpy(np.asarray(qp_b, dtype=np.float64)).to(device)
    w_t = torch.from_numpy(np.asarray(w_b, dtype=np.float64)).to(device)
    with torch.enable_grad():
        H = batched_hessian_fn(qp_t, w_t)
    return H.detach().cpu().numpy()


_ = compute_hessian_batch(qp_batch[:2], w_batch[:2])  # warm-up (first-call tracing overhead)

t0 = time.perf_counter()
K_new = compute_hessian_batch(qp_batch, w_batch)
t_new = time.perf_counter() - t0
print(f"[proto] NEW (vmap(torch.func.hessian)): {t_new:.4f}s, "
      f"{t_new / len(captured) * 1000:.4f} ms/point", flush=True)
print(f"[proto] speedup: {t_old / max(t_new, 1e-9):.1f}x", flush=True)

err = np.max(np.abs(K_old - K_new))
rel = err / max(np.max(np.abs(K_old)), 1e-300)
print(f"[proto] max abs diff: {err:.3e} (rel {rel:.3e})", flush=True)
print("PROTO_PASS" if rel < 1e-8 else "PROTO_FAIL", flush=True)
