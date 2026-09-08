#!/usr/bin/env python3
"""Stage 04: train the decoder's nonlinear closure N_slave: q_M -> q_S.

DESIGN, each choice measured rather than defaulted.

Architecture 3 -> 64 -> 64 -> 36 with tanh. tanh is not cosmetic: the decoder
must be TWICE differentiable in q_M, since its first derivative is the tangent
space of the projected equilibrium equations and its second enters the
consistent Newton linearization. That rules out ReLU. Size is modest against
4950 x 36 = 178k target values for ~6.8k parameters.

Loss is RAW MSE on q_S, not per-mode standardized. Phi_S is orthonormal, so

    || Phi_S q_S || = || q_S ||

which makes raw MSE EXACTLY the displacement reconstruction error, and the
measured stress error tracks it: truncating to k slave modes gave median
|dS|/|S| of 1.02e-02 at k=3, 1.08e-03 at k=8 and 1.15e-06 at k=36, decreasing
monotonically with the retained energy.

Standardizing the outputs would equalize RELATIVE error across modes, forcing
dq_k proportional to sigma_k, which sacrifices accuracy on the large modes to
buy it on the small ones and so INCREASES the total displacement error. The
number that settles it: slave mode 36 has amplitude 7.0e-07 of mode 1, so a 1%
error on mode 1 is already 14000x mode 36's entire amplitude. The small modes
cannot matter unless the large ones are predicted to better than 7e-07, which
no network will do.

Inputs are standardized (their natural scales are 0.060, 0.036, 0.083);
outputs are not, precisely so the loss stays the displacement error.

GENERALIZATION is reported END-TO-END on the 400 test states declared in stage
02, and on the 350 out-of-envelope probe states: the network's q_S is fed
through the decoder, the displacement assembled, and the homogenized stress
compared against the stored FOM value. That measures the quantity of interest
rather than an intermediate, and it needs no extra data -- test and probe
snapshots were deliberately not stored, only their stresses.

The internal 90/10 split is for early stopping ONLY. It is optimistic as a
generalization estimate, since grid neighbours sit on both sides of it, and it
is not used as one; that is what the pre-declared test set is for.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(ROOT / "00_rve"),
          str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

SEED = 20260903
WIDTH = 64
DEPTH = 2
VAL_FRAC = 0.10
EPOCHS = 20000
LR = 2.0e-3
PATIENCE = 1500
N_EVAL = 120          # states to evaluate end-to-end (each needs an assembly)


def build_net(n_in, n_out, width=WIDTH, depth=DEPTH):
    import torch
    import torch.nn as nn
    layers = []
    d = n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers += [nn.Linear(d, n_out)]
    net = nn.Sequential(*layers).double()
    return net


def train(qM, qS, width=WIDTH, depth=DEPTH, verbose=True):
    import torch
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    mu_m, mu_s = qM.mean(0), qM.std(0)
    X = torch.from_numpy((qM - mu_m) / mu_s)
    Y = torch.from_numpy(qS)

    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = int(VAL_FRAC * n)
    vi, ti = perm[:n_val], perm[n_val:]
    Xt, Yt, Xv, Yv = X[ti], Y[ti], X[vi], Y[vi]

    net = build_net(X.shape[1], Y.shape[1], width=width, depth=depth)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=300, min_lr=1e-6)

    # Reported as relative, dividing by ||Y||^2, which does not move the
    # optimum but makes the number readable as a displacement error.
    nrm_t = float(torch.mean(Yt ** 2))
    nrm_v = float(torch.mean(Yv ** 2))

    best, best_state, bad = np.inf, None, 0
    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        net.train()
        opt.zero_grad()
        loss = torch.mean((net(Xt) - Yt) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            net.eval()
            vl = float(torch.mean((net(Xv) - Yv) ** 2))
        sched.step(vl)
        if vl < best * (1 - 1e-7):
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad > PATIENCE:
                break
        if verbose and (ep % 1000 == 0 or ep == 1):
            print(f"  ep {ep:6d}  train {float(loss) / nrm_t:.4e}  "
                  f"val {vl / nrm_v:.4e}  lr {opt.param_groups[0]['lr']:.2e}",
                  flush=True)
    net.load_state_dict(best_state)
    if verbose:
        print(f"  stopped at epoch {ep}, best relative val {best / nrm_v:.4e}, "
              f"{time.perf_counter() - t0:.0f}s")
    return net, (mu_m, mu_s), best / nrm_v


def per_mode_quality(net, qM, qS, norm):
    import torch
    mu_m, mu_s = norm
    with torch.no_grad():
        P = net(torch.from_numpy((qM - mu_m) / mu_s)).numpy()
    num = np.linalg.norm(P - qS, axis=0)
    den = np.linalg.norm(qS, axis=0)
    return num / np.maximum(den, 1e-300), den / den[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    a = ap.parse_args()

    b = np.load(HERE / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    qM = np.ascontiguousarray(b["q_M"].T)       # (n, 3)
    qS = np.ascontiguousarray(b["q_S"].T)       # (n, 36)
    print(f"training pairs {qM.shape[0]}, inputs {qM.shape[1]}, "
          f"outputs {qS.shape[1]}, net {a.width}x{a.depth} tanh")

    net, norm, relval = train(qM, qS, width=a.width, depth=a.depth)

    rel, amp = per_mode_quality(net, qM, qS, norm)
    print("\nper-mode relative error on the training pairs "
          "(amplitude relative to mode 1):")
    for k in range(qS.shape[1]):
        mark = "  <-- below its own amplitude" if rel[k] > 1.0 else ""
        print(f"  slave {k + 1:2d}: amp {amp[k]:.3e}   rel err {rel[k]:.3e}{mark}")

    # END-TO-END: decoder -> displacement -> homogenized stress
    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    import torch
    Phi_M, Phi_S, A_M = b["Phi_M"], b["Phi_S"], b["A_M"]
    mu_m, mu_s = norm

    def rom_stress(rve, E):
        with torch.no_grad():
            qs = net(torch.from_numpy(((E - mu_m) / mu_s)[None, :])).numpy()[0]
        u_ind = Phi_M @ (A_M @ E) + Phi_S @ qs
        rve.assembler.Assemble(rve.T @ u_ind + rve._g(E))
        return rve.homogenized_stress(E)

    rng = np.random.default_rng(3)
    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        print("\nend-to-end homogenized-stress error (ROM vs FOM):")
        out = {}
        for nm in ("test", "probe"):
            E_s, S_s = d[f"E_{nm}"], d[f"S_{nm}"]
            ok = np.isfinite(S_s).all(axis=1)
            E_s, S_s = E_s[ok], S_s[ok]
            idx = rng.choice(E_s.shape[0], min(N_EVAL, E_s.shape[0]), replace=False)
            errs = []
            for i in idx:
                S_rom = rom_stress(rve, E_s[i])
                errs.append(np.linalg.norm(S_rom - S_s[i]) / np.linalg.norm(S_s[i]))
            errs = np.array(errs)
            out[nm] = errs
            print(f"  {nm:6s} n={len(idx):4d}   median {np.median(errs):.4e}   "
                  f"p90 {np.percentile(errs, 90):.4e}   max {errs.max():.4e}")

    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    np.savez_compressed(HERE / "nslave.npz", mu_mean=mu_m, mu_std=mu_s,
                        width=a.width, depth=a.depth, rel_val=relval,
                        per_mode_rel=rel, per_mode_amp=amp,
                        err_test=out["test"], err_probe=out["probe"], **sd)
    print("\nNSLAVE_TRAINED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
