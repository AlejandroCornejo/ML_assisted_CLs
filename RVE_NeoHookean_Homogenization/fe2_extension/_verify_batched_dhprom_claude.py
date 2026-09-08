#!/usr/bin/env python3
"""Verify the batched dhprom_ann_pk2_2d_vectorized_consistent_float64
(now using DHpromAnnDirectLawFloat64._decoder_batch) reproduces the
UNBATCHED evaluate_with_tangent(E) path (no _decoder_precomputed --
still the exact original per-row jacfwd computation) to floating-point
precision, on real macro strains from the n_body=6 cruciform run, and
reports the real end-to-end speedup."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dhprom_ann_direct_law_float64_claude as m  # noqa: E402

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"

d = np.load(HERE / "cruciform_results_dhprom_f64_consistent_claude.npz")
e_gp = d["e_gp"]
print(f"[verify] loaded {e_gp.shape[0]} real macro strains", flush=True)

law = m.DHpromAnnDirectLawFloat64(hprom_ann_dir=str(DHPROMANN_DIR))
_ = law.evaluate_with_tangent(e_gp[0])  # warm-up

N_SAMPLE = 300
rng = np.random.default_rng(0)
sample_idx = rng.choice(e_gp.shape[0], size=min(N_SAMPLE, e_gp.shape[0]), replace=False)
E_batch = e_gp[sample_idx]

# --- OLD path: unbatched, per-row, no precomputation (still exact original code) ---
t0 = time.perf_counter()
S_old = np.zeros((len(E_batch), 3))
CC_old = np.zeros((len(E_batch), 3, 3))
for i, E in enumerate(E_batch):
    _, S_old[i], _, CC_old[i] = law.evaluate_with_tangent(E)
t_old = time.perf_counter() - t0
print(f"[verify] OLD (unbatched) path: {t_old:.3f}s, {t_old/len(E_batch)*1000:.3f} ms/point", flush=True)

# --- NEW path: the actual deployed, now-batched function ---
m._DEFAULT_LAW_F64 = law  # reuse the same already-built law instance
t0 = time.perf_counter()
S_new, CC_new = m.dhprom_ann_pk2_2d_vectorized_consistent_float64(E_batch)
t_new = time.perf_counter() - t0
print(f"[verify] NEW (batched decoder) path: {t_new:.3f}s, {t_new/len(E_batch)*1000:.3f} ms/point", flush=True)
print(f"[verify] end-to-end speedup: {t_old/max(t_new,1e-9):.2f}x", flush=True)

s_err = np.max(np.abs(S_old - S_new))
cc_err = np.max(np.abs(CC_old - CC_new))
s_rel = s_err / max(np.max(np.abs(S_old)), 1e-300)
cc_rel = cc_err / max(np.max(np.abs(CC_old)), 1e-300)
print(f"[verify] max abs diff S:  {s_err:.3e}  (rel {s_rel:.3e})", flush=True)
print(f"[verify] max abs diff CC: {cc_err:.3e}  (rel {cc_rel:.3e})", flush=True)

ok = s_rel < 1e-8 and cc_rel < 1e-8
print("VERIFY_PASS" if ok else "VERIFY_FAIL", flush=True)
