#!/usr/bin/env python3
"""Does restricting training to the region the coupon actually visits fix the
ICNN's amplitude deficit?

WHAT THE SAMPLING BOX GOT WRONG. The box was built as the rectangular hull of
the macro prepass cloud, widened by a 40% envelope margin applied per
component. Measured against the cloud it was meant to cover:

    quantity   coupon Gauss points        sampling box
    E11        +0.0096 .. +0.1505         0.0000 .. +0.1955
    E22        -0.0711 .. -0.0046        -0.0987 .. +0.0163
    J           1.0049 ..  1.0577         0.9037 ..  1.1844
    J < 1       0.0% of states            28.3% of states

The coupon NEVER enters volumetric compression: Poisson contraction in E22 is
always outweighed by extension in E11. Widening a rectangular hull per
component reaches corners the cloud never visits -- low E11 with strongly
negative E22 and large shear -- and those corners cross J = 1, which is not a
wider version of the same regime but a different one, pore closure instead of
pore opening.

That matters specifically for the polyconvex ansatz, because all 15 of its
features are monotone measures of stretch: under compression they all DECREASE
while the energy INCREASES. Counted over 24.5M comparable pairs, the box gives
2,337,075 monotonicity violations with a worst excess of 1094x; restricting to
J >= 1 drops the worst excess to 12.3x, an 89-fold reduction, though 660k
violations remain.

Note the box defect is harmless for the ROM -- extra coverage is wasteful, not
wrong, and the HPROM/MAW models were validated on the test and probe sets, not
on the box. It bites only the energy-based tiers.

Evaluation stays on the FULL test and probe sets in every case, so restricting
the TRAINING region cannot flatter the reported numbers.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "12")
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "RVE_NeoHookean_Homogenization" / "pann" / "anisotropic"))

WID = (128, 128)
EPOCHS, BATCH, LR, PATIENCE = 700, 4096, 2.0e-3, 250


def jac(E):
    return np.sqrt(np.maximum((1 + 2 * E[:, 0]) * (1 + 2 * E[:, 1]) - E[:, 2] ** 2, 0))


def main():
    import torch
    import train_anisotropic_pann_claude as T
    from anisotropic_pann_model import AnisotropicPolyconvexEnergy

    d = np.load(ROOT / "03_data" / "data.npz")
    E0, S0, W0 = d["E_train"], d["S_train"], d["W_train"]
    fin = np.isfinite(S0).all(1) & np.isfinite(W0)
    sets = {nm: (d[f"E_{nm}"][np.isfinite(d[f"S_{nm}"]).all(1)],
                 d[f"S_{nm}"][np.isfinite(d[f"S_{nm}"]).all(1)])
            for nm in ("test", "probe")}

    def run(mask, lb):
        E, S, W = E0[mask], S0[mask], W0[mask]
        ss, es = float(np.abs(E).max()), float(np.abs(W).max())
        Xn, Wt, St = E / ss, W / es, S * (ss / es)
        s_den, w_den = float(np.mean(St ** 2)), float(np.mean(Wt ** 2))
        perm = np.random.default_rng(5).permutation(E.shape[0])
        nv = int(0.15 * E.shape[0]); vi, fi = perm[:nv], perm[nv:]
        Xt = torch.tensor(Xn, dtype=torch.float64)
        Stt = torch.tensor(St, dtype=torch.float64)
        Wtt = torch.tensor(Wt, dtype=torch.float64)
        torch.manual_seed(20260904)
        fs = T.derive_polyconvex_feature_scale(Xt, strain_scale=ss, widths=WID)
        m = AnisotropicPolyconvexEnergy(strain_scale=ss, widths=WID,
                                        feature_scale=fs).double()
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
                (ls + lw).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 50.0)
                opt.step()
            m.eval()
            v = float(L(vi, False)[0]); sch.step(v)
            if v < best - 1e-14:
                best, bep = v, ep
                bst = {k: t.detach().clone() for k, t in m.state_dict().items()}
            elif ep - bep > PATIENCE:
                break
        m.load_state_dict(bst); m.eval()
        xf = Xt[fi].clone().requires_grad_(True)
        wp, sp = m.energy_and_stress(xf, create_graph=False)
        wp = wp.detach().numpy().ravel(); sp = sp.detach().numpy()
        ratio = float(np.median(wp / np.maximum(Wt[fi], 1e-30)))
        errs = {}
        for nm, (Es, Ss) in sets.items():
            xe = torch.tensor(Es / ss, dtype=torch.float64).requires_grad_(True)
            spe = m.energy_and_stress(xe, create_graph=False)[1]
            spe = spe.detach().numpy() * (es / ss)
            errs[nm] = float(np.linalg.norm(spe - Ss) / np.linalg.norm(Ss))
        print(f"{lb:<26} n={E.shape[0]:5d}  ratio {ratio:.4f}  "
              f"fit {float(np.mean((sp - St[fi]) ** 2) / s_den):.4e}  "
              f"test {errs['test']:.4e}  probe {errs['probe']:.4e}  "
              f"[{time.perf_counter() - t0:.0f}s ep{bep}]", flush=True)

    print(f"ICNN widths {WID}, identical budget; evaluation always on the "
          f"FULL test/probe sets\n")
    run(fin, "full box")
    run(fin & (jac(E0) >= 1.0), "J >= 1.0")
    run(fin & (jac(E0) >= 1.0049), "J >= 1.0049 (cloud min)")
    print("\nICNN_J1_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
