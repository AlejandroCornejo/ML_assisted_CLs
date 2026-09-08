#!/usr/bin/env python3
"""Which part of the energy-based path is broken?

THE DISCRIMINATING EVIDENCE, which was available from the first run and which I
spent an hour and a half not using:

    tier         potential   polyconvex   in-envelope error
    regression      no           no          8.09e-03
    free            yes          no          1.74e-01
    icnn            yes          yes         2.67e-01
    ickan           yes          yes         2.36e-01

`free` is an UNCONSTRAINED MLP potential -- no polyconvexity, no monotonicity,
no ICNN -- and it fails too, 21x worse than the direct regression on identical
data with a network of identical size. So the problem is not polyconvexity, and
it is not capacity (79x more ICNN parameters moved the error 2%), and it is not
the sampling box (restricting to the region the coupon visits made it worse).

What the three failing tiers share, and the regression does not:

    1. stress obtained by AUTOGRAD of the energy, not as a direct output
    2. a subtracted reference term (structural_zero, pressure * log J)
    3. two objectives fitted at once

This isolates 1 and 3 against each other with the same architecture:

    free, stress+energy   the configuration that failed
    free, stress only     energy weight set to zero
    free, energy only     stress weight set to zero

If stress-only reaches the regression's ~8e-03, the energy term or its scaling
is poisoning the fit and the autograd path is sound. If stress-only stays near
1.7e-01, the autograd stress path itself is wrong -- most likely a convention or
scaling error between what the model differentiates and what the target holds.

The FD column is the decisive control: the model's autograd stress against a
central finite difference of its OWN energy. Those must agree to ~1e-6. If they
do, autograd is consistent with the model's energy and any remaining error is a
fitting problem. If they do not, the derivative path is broken and no amount of
training can help.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "12")
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "RVE_NeoHookean_Homogenization" / "pann" / "anisotropic"))

EPOCHS, BATCH, LR, PATIENCE = 350, 2048, 4.0e-4, 200


def main():
    import torch
    import train_anisotropic_pann_claude as T
    from anisotropic_pann_model import AnisotropicFreeEnergy

    d = np.load(ROOT / "03_data" / "data.npz")
    E, S, W = d["E_train"], d["S_train"], d["W_train"]
    fin = np.isfinite(S).all(1) & np.isfinite(W)
    E, S, W = E[fin], S[fin], W[fin]
    ss, es = float(np.abs(E).max()), float(np.abs(W).max())
    Xn, Wt, St = E / ss, W / es, S * (ss / es)
    s_den, w_den = float(np.mean(St ** 2)), float(np.mean(Wt ** 2))
    perm = np.random.default_rng(5).permutation(E.shape[0])
    nv = int(0.15 * E.shape[0]); vi, fi = perm[:nv], perm[nv:]
    Xt = torch.tensor(Xn, dtype=torch.float64)
    Stt = torch.tensor(St, dtype=torch.float64)
    Wtt = torch.tensor(Wt, dtype=torch.float64)
    sets = {nm: (d[f"E_{nm}"][np.isfinite(d[f"S_{nm}"]).all(1)],
                 d[f"S_{nm}"][np.isfinite(d[f"S_{nm}"]).all(1)])
            for nm in ("test", "probe")}
    print(f"strain_scale {ss:.4f}  energy_scale {es:.4e}  "
          f"fit {fi.size}  val {vi.size}\n")

    def run(w_stress, w_energy, lb):
        torch.manual_seed(20260904)
        fscale = T.derive_free_feature_scale(
            Xt * ss, ss).to(torch.float64)
        m = AnisotropicFreeEnergy(strain_scale=ss, feature_scale=fscale,
                                  widths=T.FREE_WIDTHS).double()
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-9)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=40, min_lr=1e-7)

        def L(idx, cg):
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
            wp, sp = m.energy_and_stress(x, create_graph=cg)
            return (torch.mean((sp - Stt[idx]) ** 2) / s_den,
                    torch.mean((wp.reshape(-1) - Wtt[idx]) ** 2) / w_den)

        best, bep, bst = np.inf, 0, None
        gen = np.random.default_rng(20260904)
        t0 = time.perf_counter()
        for ep in range(1, EPOCHS + 1):
            m.train()
            order = fi[gen.permutation(fi.size)]
            for st in range(0, order.size, BATCH):
                opt.zero_grad(set_to_none=True)
                ls, lw = L(order[st:st + BATCH], True)
                (w_stress * ls + w_energy * lw).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 50.0)
                opt.step()
            m.eval()
            lsv, lwv = L(vi, False)
            # select on whatever this configuration is actually optimizing
            v = float(w_stress * lsv + w_energy * lwv)
            sch.step(v)
            if v < best - 1e-14:
                best, bep = v, ep
                bst = {k: t.detach().clone() for k, t in m.state_dict().items()}
            elif ep - bep > PATIENCE:
                break
        m.load_state_dict(bst); m.eval()

        # autograd stress vs a central difference of the model's OWN energy
        xs = Xt[fi[:200]].clone()
        _, sa = m.energy_and_stress(xs.clone().requires_grad_(True),
                                    create_graph=False)
        sa = sa.detach().numpy()
        h = 1.0e-6
        sfd = np.zeros_like(sa)
        for c in range(3):
            xp, xm = xs.clone(), xs.clone()
            xp[:, c] += h; xm[:, c] -= h
            # NOT under torch.no_grad(): `energy()` calls autograd internally
            # to enforce S(I) = 0 through `reference_terms(create_graph=True)`,
            # so suppressing the graph makes it raise rather than return a
            # value. Detach the results instead.
            wp_ = m.energy(xp).detach().numpy().ravel()
            wm_ = m.energy(xm).detach().numpy().ravel()
            sfd[:, c] = (wp_ - wm_) / (2 * h)
        fd = float(np.max(np.abs(sa - sfd)) / max(np.max(np.abs(sfd)), 1e-300))

        errs = {}
        for nm, (Es, Ss) in sets.items():
            xe = torch.tensor(Es / ss, dtype=torch.float64).requires_grad_(True)
            spe = m.energy_and_stress(xe, create_graph=False)[1]
            spe = spe.detach().numpy() * (es / ss)
            errs[nm] = float(np.linalg.norm(spe - Ss) / np.linalg.norm(Ss))
        lsf, lwf = L(fi, False)
        print(f"{lb:<24} stress_fit {float(lsf):.4e}  energy_fit {float(lwf):.4e}  "
              f"test {errs['test']:.4e}  probe {errs['probe']:.4e}  "
              f"autograd-vs-FD {fd:.2e}  [{time.perf_counter() - t0:.0f}s ep{bep}]",
              flush=True)

    print("tier `free` (unconstrained potential), identical architecture:")
    run(1.0, 1.0, "stress + energy")
    run(1.0, 0.0, "stress only")
    run(0.0, 1.0, "energy only")
    print("\nreference: regression tier (direct stress output) = 8.09e-03")
    print("\nDIAG_ENERGY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
