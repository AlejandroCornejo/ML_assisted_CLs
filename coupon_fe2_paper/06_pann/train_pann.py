#!/usr/bin/env python3
"""Train the four PANN tiers on the coupon RVE's (E -> W, S) data.

WHY A NEW TRAINER RATHER THAN THE EXISTING ONE. The previous project's
trainers are hard-wired to trajectory-structured data and guard it explicitly:

    if set(np.unique(trajectory_ids)) != set(range(1, 11)):
        raise RuntimeError("The final anisotropic models must use all ten
                            Stage-1 trajectories.")

This project's 4950 states come from a sampling BOX, not from ten trajectories.
Feeding them synthetic trajectory ids would satisfy that check while making it
meaningless, and would leave the per-trajectory subsampling logic operating on
a fiction. So the MODEL classes are imported unchanged -- they are the hard
part, with the ICNN's positive weights, the ICKAN core, and the constructive
polyconvexity certificate -- and only the training loop is written here, where
the split and the selection criterion can be controlled.

THE FOUR TIERS, all sharing one interface
(`energy_and_stress(normalised_strain, create_graph=...) -> (W, S)`):

    regression   AnisotropicRegressionStress        no potential at all
    free         AnisotropicFreeEnergy              a potential, unconstrained
    icnn         AnisotropicPolyconvexEnergy        polyconvex by construction
    ickan        AnisotropicPolyconvexEnergyICKAN   polyconvex, KAN core

THE VOIGT CONVENTION IS VERIFIED, NOT ASSUMED. The models take
`[E11, E22, gamma12]` with gamma12 = 2 E12, and since

    S : dE = S11 dE11 + S22 dE22 + S12 dgamma12

the autograd derivative of W with respect to that parametrization is exactly
the stored `[S11, S22, S12]`. That is the algebra; a mismatch here would be
silent, producing only worse numbers, so it is checked numerically against the
data. The training states lie on a structured grid of step GRID_STEP, so
central differences of the stored energy are available at no cost.

THE SCALING IS THE VALIDATED ONE, COPIED RATHER THAN DERIVED. The models
differentiate the energy with respect to the NORMALISED strain, so they emit
normalised quantities, and the targets must be normalised to match:

    x             = E / strain_scale
    energy_target = W / energy_scale
    stress_target = S * (strain_scale / energy_scale)

My first attempt compared the model output against PHYSICAL W and S. The
targets were then wrong by factors of energy_scale and energy_scale /
strain_scale, nothing could fit, and both tiers reported a relative error of
exactly 1.0 -- the signature of a prediction that is effectively zero.

`feature_scale` differs per tier and is derived by the previous project's own
helpers: 4 entries for the free model (the material C-features) and 15 for the
polyconvex ones (1 + 4 * 3 direction systems + 2), obtained by probing the
model's own `structural_features`. Passing the 4-vector to the polyconvex
models raised `feature_scale must contain 15 positive entries`.

SELECTION AND REPORTING ARE SEPARATE SETS. Model selection is a fit to
whatever set it watches, so the best epoch is chosen on a held-out slice of the
TRAINING states, and the reported numbers come from the untouched `test`
(in-envelope) and `probe` (out-of-envelope) sets.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANN = ROOT.parent / "RVE_NeoHookean_Homogenization" / "pann" / "anisotropic"
for _p in (str(PANN),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VAL_FRAC = 0.15
SPLIT_SEED = 5
GRID_STEP = 0.0115
# HYPERPARAMETERS TAKEN FROM THE VALIDATED TRAINER, per tier, rather than
# chosen here. My own first guess was wrong in three ways at once: 20000
# full-batch epochs (a factor ~14 more optimizer steps than needed), a single
# learning rate of 2e-3 for every tier (5x too high for the free model, which
# duly diverged -- early stop at 3699 with its best at 698), and plain Adam
# with no gradient clipping.
BUDGET = {
    "regression": dict(epochs=350, batch=2048, lr=4.0e-4),
    "free":       dict(epochs=350, batch=2048, lr=4.0e-4),
    "icnn":       dict(epochs=700, batch=4096, lr=2.0e-3),
    "ickan":      dict(epochs=700, batch=4096, lr=2.0e-3),
}
WEIGHT_DECAY = 1.0e-9
GRAD_CLIP = 50.0
PATIENCE = 200
SCHED_PATIENCE = 40
ENERGY_WEIGHT = 1.0
STRESS_WEIGHT = 1.0


def check_voigt_convention(E, W, S, tol=5.0e-3):
    """Central differences of the stored energy against the stored stress.

    Uses the grid structure of the sampling box: neighbours differing by
    exactly one GRID_STEP in a single component give dW/dE_i directly. Only
    states with BOTH neighbours present are used, so the estimate is second
    order.
    """
    key = {tuple(np.round(e / GRID_STEP).astype(np.int64)): i
           for i, e in enumerate(E)}
    print("  component   pairs   max rel err   median rel err")
    ok = True
    for c in range(3):
        num, ref = [], []
        step = np.zeros(3, dtype=np.int64)
        step[c] = 1
        for k, i in key.items():
            kp, km = tuple(np.array(k) + step), tuple(np.array(k) - step)
            if kp in key and km in key:
                ip, im = key[kp], key[km]
                h = E[ip, c] - E[im, c]
                if abs(h) < 1e-12:
                    continue
                num.append((W[ip] - W[im]) / h)
                ref.append(S[i, c])
        if not num:
            print(f"  {c:>9}   {'none':>5}   (no interior grid neighbours)")
            continue
        num, ref = np.asarray(num), np.asarray(ref)
        sc = max(np.max(np.abs(ref)), 1e-300)
        rel = np.abs(num - ref) / sc
        print(f"  {c:>9} {len(num):>7}   {rel.max():.3e}   {np.median(rel):.3e}")
        ok = ok and np.median(rel) < tol
    return ok


def build(kind, strain_scale, Xn, torch):
    """Each tier gets the feature scaling its own architecture requires,
    through the previous project's helpers rather than a guess here."""
    import train_anisotropic_pann_claude as T
    from anisotropic_pann_model import (AnisotropicFreeEnergy,
                                        AnisotropicPolyconvexEnergy)
    x = torch.tensor(Xn, dtype=torch.float64)
    if kind == "regression":
        from anisotropic_pann_model_regression_claude import (
            AnisotropicRegressionStress)
        return AnisotropicRegressionStress(strain_scale=strain_scale,
                                           widths=T.FREE_WIDTHS)
    if kind == "free":
        fs = T.derive_free_feature_scale(
            x * strain_scale, strain_scale).to(torch.float64)
        return AnisotropicFreeEnergy(strain_scale=strain_scale,
                                     feature_scale=fs, widths=T.FREE_WIDTHS)
    fs = T.derive_polyconvex_feature_scale(
        x, strain_scale=strain_scale, widths=T.POLYCONVEX_WIDTHS)
    if kind == "icnn":
        return AnisotropicPolyconvexEnergy(strain_scale=strain_scale,
                                           widths=T.POLYCONVEX_WIDTHS,
                                           feature_scale=fs)
    if kind == "ickan":
        from anisotropic_pann_model_ickan_claude import (
            AnisotropicPolyconvexEnergyICKAN)
        return AnisotropicPolyconvexEnergyICKAN(strain_scale=strain_scale,
                                                feature_scale=fs)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=("regression", "free", "icnn", "ickan"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--check-only", action="store_true")
    a_ = ap.parse_args()

    import torch

    d = np.load(ROOT / "03_data" / "data.npz")
    E, S, W = d["E_train"], d["S_train"], d["W_train"]
    fin = np.isfinite(S).all(axis=1) & np.isfinite(W)
    E, S, W = E[fin], S[fin], W[fin]
    print(f"training states {E.shape[0]}  (of {d['E_train'].shape[0]})")

    print("\n=== Voigt convention: dW/dE from grid central differences "
          "vs stored S ===")
    ok = check_voigt_convention(E, W, S)
    print(f"  convention {'CONFIRMED' if ok else 'MISMATCH -- do not train'}")
    if a_.check_only or not ok:
        return 0 if ok else 1

    torch.manual_seed(a_.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))

    strain_scale = float(np.max(np.abs(E)))
    energy_scale = float(np.max(np.abs(W)))
    Xn = E / strain_scale
    W_t = W / energy_scale
    S_t = S * (strain_scale / energy_scale)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(E.shape[0])
    nv = int(VAL_FRAC * E.shape[0])
    vi, fi = perm[:nv], perm[nv:]

    model = build(a_.kind, strain_scale, Xn, torch).double()
    npar = sum(p.numel() for p in model.parameters())
    print(f"\nmodel {a_.kind}: {npar} parameters, "
          f"strain_scale {strain_scale:.4f}, energy_scale {energy_scale:.4e}")

    Xt = torch.tensor(Xn, dtype=torch.float64)
    St = torch.tensor(S_t, dtype=torch.float64)
    Wt = torch.tensor(W_t, dtype=torch.float64)
    # Denominators from the normalised targets, as the validated trainer does,
    # so the two loss terms are comparable without hand-tuned weights.
    s_den = float(max(np.mean(S_t ** 2), 1e-12))
    w_den = float(max(np.mean(W_t ** 2), 1e-12))

    bud = BUDGET[a_.kind]
    n_ep = int(a_.epochs) if a_.epochs else bud["epochs"]
    nb = int(bud["batch"])
    opt = torch.optim.AdamW(model.parameters(), lr=bud["lr"],
                            weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=SCHED_PATIENCE, min_lr=1e-7)
    print(f"  budget {n_ep} epochs, batch {nb}, lr {bud['lr']:.1e}, "
          f"AdamW wd {WEIGHT_DECAY:.0e}, grad clip {GRAD_CLIP:g}", flush=True)

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
        if a_.kind == "regression":
            sp = model.stress(x)
            wp = torch.zeros(len(idx), dtype=torch.float64)
        else:
            wp, sp = model.energy_and_stress(x, create_graph=create_graph)
        ls = torch.mean((sp - St[idx]) ** 2) / s_den
        lw = (torch.zeros((), dtype=torch.float64) if a_.kind == "regression"
              else torch.mean((wp.reshape(-1) - Wt[idx]) ** 2) / w_den)
        return ls, lw

    best, best_ep, best_state = np.inf, 0, None
    t0 = time.perf_counter()
    gen = np.random.default_rng(a_.seed)
    for ep in range(1, n_ep + 1):
        model.train()
        order = fi[gen.permutation(fi.size)]
        for st in range(0, order.size, nb):
            idx = order[st:st + nb]
            opt.zero_grad(set_to_none=True)
            ls, lw = losses(idx, create_graph=True)
            loss = STRESS_WEIGHT * ls + ENERGY_WEIGHT * lw
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        lsv, lwv = losses(vi, create_graph=False)
        v = float(lsv)
        sch.step(v)
        if v < best - 1e-14:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone()
                          for k, t in model.state_dict().items()}
        elif ep - best_ep > PATIENCE:
            print(f"  early stop at epoch {ep} (best {best_ep})")
            break
        if ep % max(1, n_ep // 10) == 0:
            print(f"  ep {ep:6d}  stress {float(ls):.4e}/{v:.4e}  "
                  f"energy {float(lw):.4e}  best {best:.4e}@{best_ep}",
                  flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  trained in {time.perf_counter() - t0:.0f}s, "
          f"best epoch {best_ep}")

    print("\n=== stress error vs FOM, relative Frobenius ===")
    out = {}
    for nm in ("test", "probe"):
        Es, Ss = d[f"E_{nm}"], d[f"S_{nm}"]
        m = np.isfinite(Ss).all(axis=1)
        Es, Ss = Es[m], Ss[m]
        x = torch.tensor(Es / strain_scale,
                         dtype=torch.float64).requires_grad_(True)
        if a_.kind == "regression":
            sp = model.stress(x).detach().numpy()
        else:
            sp = model.energy_and_stress(x, create_graph=False)[1]
            sp = sp.detach().numpy()
        # back to physical units before comparing with the stored stress
        sp = sp * (energy_scale / strain_scale)
        e = float(np.linalg.norm(sp - Ss) / np.linalg.norm(Ss))
        out[nm] = e
        print(f"  {nm:>6}  {e:.4e}   n={Es.shape[0]}")
    print(f"  degradation {out['probe'] / out['test']:.1f}x")

    torch.save(dict(state_dict=model.state_dict(), kind=a_.kind,
                    strain_scale=strain_scale, energy_scale=energy_scale,
                    best_epoch=best_ep, err_test=out["test"],
                    err_probe=out["probe"]),
               HERE / f"pann_{a_.kind}.pt")
    print(f"\nsaved pann_{a_.kind}.pt")
    print("TRAIN_PANN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
