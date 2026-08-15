#!/usr/bin/env python3
"""Follow-up on check (B) of verify_dhprom_ann_consistent_tangent_subpieces
_claude.py, which found 1-4% mismatches between the autograd decoder
Jacobian and a single-h (1e-3) finite difference at a synthetic q_p. Two
suspects: (i) h=1e-3 genuinely too large for this network's local
curvature (truncation error, should shrink as h shrinks), or (ii) a real
bug in the autograd usage (should NOT shrink as h shrinks). Also uses q_p
values pulled from REAL macro strain states via the actual qp_aff map,
not an arbitrary synthetic point, in case the network behaves unusually
far from its training manifold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann", "hprom/ann"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw


def decoder_jacobian_autograd(law, q_p0):
    q_p0_t = torch.from_numpy(q_p0.astype(np.float32)).reshape(1, -1).to(law.device)
    with torch.enable_grad():
        q_in = q_p0_t.reshape(-1).clone().detach().requires_grad_(True)

        def ann_from_qvec(qvec):
            return law.ann_model(qvec.view(1, -1)).reshape(-1)

        J = torch.autograd.functional.jacobian(ann_from_qvec, q_in).reshape(
            law.n_secondary, law.n_primary
        ).detach().cpu().numpy()
    return J


def decoder_jacobian_fd(law, q_p0, h):
    J_fd = np.zeros((law.n_secondary, law.n_primary), dtype=float)
    with torch.no_grad():
        for k in range(law.n_primary):
            qp = q_p0.copy(); qp[k] += h
            qm = q_p0.copy(); qm[k] -= h
            yp = law.ann_model(torch.from_numpy(qp.astype(np.float32)).reshape(1, -1).to(law.device)).reshape(-1).cpu().numpy()
            ym = law.ann_model(torch.from_numpy(qm.astype(np.float32)).reshape(1, -1).to(law.device)).reshape(-1).cpu().numpy()
            J_fd[:, k] = (yp - ym) / (2 * h)
    return J_fd


def main():
    print("[decoder-jacobian-hscan] building DHpromAnnDirectLaw ...")
    law = DHpromAnnDirectLaw()

    print(f"\nn_primary={law.n_primary}, n_secondary={law.n_secondary}")

    print("\n--- q_p from REAL macro strain states (via qp_aff) ---")
    mu_dim = int(law.qp_aff["mu_dim"])
    b_aff = np.asarray(law.qp_aff["b_aff"], dtype=float)
    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.3, -0.1, 0.05]),
        np.array([0.8, 0.4, -0.05]),
        np.array([1.2, 0.6, 0.06]),
    ]
    for E in test_states:
        mu = E[:mu_dim]
        q_p0 = np.concatenate([mu, [1.0]]) @ b_aff
        print(f"\n  E={E} -> q_p0 (first 5 of {q_p0.size})={q_p0[:5]}, ||q_p0||={np.linalg.norm(q_p0):.4e}")
        J_analytic = decoder_jacobian_autograd(law, q_p0)
        prev = None
        for h in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5):
            J_fd = decoder_jacobian_fd(law, q_p0, h)
            rel = np.linalg.norm(J_analytic - J_fd) / max(np.linalg.norm(J_fd), 1e-30)
            trend = f" (x{rel / prev:.3f})" if prev else ""
            print(f"    h={h:.0e}: ||J_analytic-J_fd||/||J_fd||={rel:.3e}{trend}")
            prev = rel


if __name__ == "__main__":
    main()
