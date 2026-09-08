#!/usr/bin/env python3
"""Does the ICNN's uniform amplitude deficit come from capacity?

WHAT IS BEING TESTED. The trained ICNN at widths (24, 24) reproduces the SHAPE
of the energy surface but not its MAGNITUDE: the predicted-to-target ratio sits
at 0.76-0.85 across the whole range of the energy, and removing a single scale
factor drops the relative error from 0.203 to 0.113. So the deficit is uniform,
not a failure in the tail or near the origin.

Two explanations survive that measurement:

  capacity     1385 parameters against the regression tier's 25475, with widths
               inherited from a different problem.
  saturation   an ICNN with softplus activations and non-negative weights,
               fed features divided by their own maxima, may simply be unable
               to reach the required output magnitude regardless of width.

Width separates them. If the ratio moves toward 1 as the network grows, it was
capacity. If it stays pinned near 0.8, it is structural and the next suspects
are `feature_scale` and the quadratic volumetric term, not the layer sizes.

Two things this reports that the first run did not:

  * losses measured on the FULL fit set rather than on whatever minibatch
    happened to be last. With batch 4096 over 4207 states the final minibatch
    holds 111 samples, and reading its energy loss as if it were the epoch's
    cost me a completely wrong diagnosis -- 5.59e-01 printed against 4.12e-02
    actual, a factor 13.
  * the amplitude ratio itself, since that is the quantity under test.

Polyconvexity is unaffected by width: the certificate rests on each feature
being convex in F, cof F or J and on the ICNN being convex and non-decreasing,
neither of which depends on the layer sizes.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANN = ROOT.parent / "RVE_NeoHookean_Homogenization" / "pann" / "anisotropic"
if str(PANN) not in sys.path:
    sys.path.insert(0, str(PANN))

WIDTHS = ((24, 24), (64, 64), (128, 128), (128, 128, 64), (256, 256, 128))
EPOCHS = 700
BATCH = 4096
LR = 2.0e-3
WEIGHT_DECAY = 1.0e-9
GRAD_CLIP = 50.0
SCHED_PATIENCE = 40
PATIENCE = 250
VAL_FRAC = 0.15
SPLIT_SEED = 5
SEED = 20260904


def main():
    import torch

    import train_anisotropic_pann_claude as T
    from anisotropic_pann_model import AnisotropicPolyconvexEnergy

    d = np.load(ROOT / "03_data" / "data.npz")
    E, S, W = d["E_train"], d["S_train"], d["W_train"]
    fin = np.isfinite(S).all(axis=1) & np.isfinite(W)
    E, S, W = E[fin], S[fin], W[fin]

    ss = float(np.max(np.abs(E)))
    es = float(np.max(np.abs(W)))
    Xn, Wt, St = E / ss, W / es, S * (ss / es)
    s_den = float(max(np.mean(St ** 2), 1e-12))
    w_den = float(max(np.mean(Wt ** 2), 1e-12))

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(E.shape[0])
    nv = int(VAL_FRAC * E.shape[0])
    vi, fi = perm[:nv], perm[nv:]

    Xt = torch.tensor(Xn, dtype=torch.float64)
    Stt = torch.tensor(St, dtype=torch.float64)
    Wtt = torch.tensor(Wt, dtype=torch.float64)

    sets = {}
    for nm in ("test", "probe"):
        Es, Ss = d[f"E_{nm}"], d[f"S_{nm}"]
        m = np.isfinite(Ss).all(axis=1)
        sets[nm] = (Es[m], Ss[m])

    print(f"{'widths':<18} {'params':>7} {'stress fit':>11} {'energy fit':>11} "
          f"{'ratio':>7} {'test':>11} {'probe':>11} {'degr':>6} {'s':>5}")
    print("-" * 96)
    rows = []
    for wid in WIDTHS:
        torch.manual_seed(SEED)
        fs = T.derive_polyconvex_feature_scale(Xt, strain_scale=ss, widths=wid)
        model = AnisotropicPolyconvexEnergy(
            strain_scale=ss, widths=wid, feature_scale=fs).double()
        npar = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=SCHED_PATIENCE, min_lr=1e-7)

        def losses(idx, create_graph):
            # wp comes back as (n, 1) and the target is (n,). Subtracting them
            # BROADCASTS to (n, n) -- the mean of all n^2 CROSS differences, not
            # the elementwise error. That silently pinned the energy loss near
            # 0.5 for any model, good or bad, and its meaningless gradient
            # wrecked the stress fit: with the term dropped entirely the same
            # architecture went from 1.97e-01 to 7.52e-03. reshape(-1) is the
            # whole fix. It also explains a discrepancy I had blamed on
            # minibatch noise -- the reporting path used .ravel() and was right,
            # the training path did not and was wrong.
            x = Xt[idx].clone().requires_grad_(True)
            wp, sp = model.energy_and_stress(x, create_graph=create_graph)
            return (torch.mean((sp - Stt[idx]) ** 2) / s_den,
                    torch.mean((wp.reshape(-1) - Wtt[idx]) ** 2) / w_den)

        best, best_ep, best_state = np.inf, 0, None
        gen = np.random.default_rng(SEED)
        t0 = time.perf_counter()
        for ep in range(1, EPOCHS + 1):
            model.train()
            order = fi[gen.permutation(fi.size)]
            for st in range(0, order.size, BATCH):
                opt.zero_grad(set_to_none=True)
                ls, lw = losses(order[st:st + BATCH], create_graph=True)
                loss = ls + lw
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss, widths {wid}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            model.eval()
            with torch.no_grad():
                pass
            lsv, _ = losses(vi, create_graph=False)
            v = float(lsv)
            sch.step(v)
            if v < best - 1e-14:
                best, best_ep = v, ep
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}
            elif ep - best_ep > PATIENCE:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        dt = time.perf_counter() - t0

        # FULL fit set, not a trailing minibatch
        model.eval()
        xf = Xt[fi].clone().requires_grad_(True)
        wp, sp = model.energy_and_stress(xf, create_graph=False)
        wp = wp.detach().numpy().ravel()
        sp = sp.detach().numpy()
        l_s = float(np.mean((sp - St[fi]) ** 2) / s_den)
        l_w = float(np.mean((wp - Wt[fi]) ** 2) / w_den)
        ratio = float(np.median(wp / np.maximum(Wt[fi], 1e-30)))

        errs = {}
        for nm, (Es, Ss) in sets.items():
            xe = torch.tensor(Es / ss, dtype=torch.float64).requires_grad_(True)
            spe = model.energy_and_stress(xe, create_graph=False)[1]
            spe = spe.detach().numpy() * (es / ss)
            errs[nm] = float(np.linalg.norm(spe - Ss) / np.linalg.norm(Ss))

        print(f"{str(wid):<18} {npar:>7} {l_s:>11.4e} {l_w:>11.4e} "
              f"{ratio:>7.4f} {errs['test']:>11.4e} {errs['probe']:>11.4e} "
              f"{errs['probe'] / errs['test']:>5.1f}x {dt:>5.0f}", flush=True)
        rows.append((wid, npar, l_s, l_w, ratio, errs["test"], errs["probe"]))
        torch.save(dict(state_dict=model.state_dict(), widths=wid,
                        strain_scale=ss, energy_scale=es,
                        err_test=errs["test"], err_probe=errs["probe"],
                        ratio=ratio, best_epoch=best_ep),
                   HERE / f"icnn_w{'x'.join(map(str, wid))}.pt")

    best = min(rows, key=lambda r: r[5])
    print(f"\nbest in-envelope: widths {best[0]}, {best[1]} params, "
          f"test {best[5]:.4e}, amplitude ratio {best[4]:.4f}")
    print("amplitude ratio near 1 -> the deficit was capacity; "
          "pinned near 0.8 -> structural, look at feature_scale next")
    print("\nICNN_WIDTH_SWEEP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
