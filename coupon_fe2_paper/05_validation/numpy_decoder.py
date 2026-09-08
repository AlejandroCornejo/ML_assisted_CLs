#!/usr/bin/env python3
"""The decoder and its Jacobian in closed form, in NumPy, with no torch in the
hot loop.

WHY. The first HPROM-ANN called torch.autograd.functional.jacobian once per
Newton iteration, ~146 times per solve, and came out 3x SLOWER than the linear
HPROM despite solving 3 unknowns instead of 39. Two things were wrong with
that:

  * it is REVERSE mode, which for a 3 -> 36 map costs 36 backward passes where
    forward mode would cost 3;
  * even in forward mode, the per-call torch dispatch dominates a network this
    small. The previous project measured exactly this and reported a 61.6x gain
    from batching the same computation.

But the network is a plain MLP with tanh, so its Jacobian is closed form:

    y  = W3 h2 + b3,  h2 = tanh(W2 h1 + b2),  h1 = tanh(W1 x + b1)
    J  = W3 diag(1 - h2^2) W2 diag(1 - h1^2) W1

which is three products of tiny matrices. Vectorized over a batch of q, so the
same routine serves a single Newton iteration and an entire FE^2 Gauss-point
sweep.

Verified against the torch Jacobian in the self-test below rather than trusted:
a closed form derived by hand is exactly the kind of thing that is 99% right.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE), str(ROOT / "04_training")):
    if p not in sys.path:
        sys.path.insert(0, p)


class NumpyDecoder:
    """d_red(q) = Phi_M A_M q + Phi_S N(q), and Phi_D = d(d_red)/dq."""

    def __init__(self, basis_npz, net_npz):
        b, m = np.load(basis_npz), np.load(net_npz)
        self.Phi_M = np.ascontiguousarray(b["Phi_M"])
        self.Phi_S = np.ascontiguousarray(b["Phi_S"])
        self.A_M = np.ascontiguousarray(b["A_M"])
        self.mu_m = np.asarray(m["mu_mean"], dtype=float)
        self.mu_s = np.asarray(m["mu_std"], dtype=float)

        # nn.Sequential(Linear, Tanh, Linear, Tanh, ..., Linear) stores its
        # parameters as "<layer>.weight" / "<layer>.bias" with the layer index
        # counting Tanh modules too, so sorting numerically recovers the order.
        keys = [k for k in m.files if k.endswith(".weight")]
        order = sorted(keys, key=lambda k: int(k.split(".")[0]))
        self.Ws = [np.ascontiguousarray(m[k]) for k in order]
        self.bs = [np.ascontiguousarray(m[k.replace(".weight", ".bias")])
                   for k in order]
        self.PhiMA = self.Phi_M @ self.A_M

    def net_and_jac(self, Q):
        """Q (n,3) -> (N (n,n_s), J (n,n_s,3)). Batched."""
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        X = (Q - self.mu_m) / self.mu_s
        h = X
        acts = []
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = np.tanh(h @ W.T + b)
            acts.append(h)
        N = h @ self.Ws[-1].T + self.bs[-1]

        # J = W_last diag(1-h^2) W ... , accumulated from the output side.
        J = np.broadcast_to(self.Ws[-1], (Q.shape[0],) + self.Ws[-1].shape)
        J = np.array(J, dtype=float)
        for W, hk in zip(reversed(self.Ws[:-1]), reversed(acts)):
            J = (J * (1.0 - hk ** 2)[:, None, :]) @ W
        J = J / self.mu_s[None, None, :]        # chain rule of the input scaling
        return N, J

    def restrict(self, rows):
        """A decoder that returns only the ROWS the caller needs.

        The 140 us/call of the full version is dominated not by the network's
        Jacobian but by Phi_S @ N and Phi_S @ J, which are (6320 x 36) products.
        A hyperreduced solve only ever needs d_red and Phi_D at the dofs its own
        reduced mesh touches -- about 640 of 6320 -- so slicing the bases once,
        outside the Newton loop, removes an order of magnitude of arithmetic
        that was being thrown away every iteration.
        """
        out = object.__new__(NumpyDecoder)
        out.Phi_M, out.A_M = self.Phi_M, self.A_M
        out.Phi_S = np.ascontiguousarray(self.Phi_S[rows])
        out.PhiMA = np.ascontiguousarray(self.PhiMA[rows])
        out.mu_m, out.mu_s = self.mu_m, self.mu_s
        out.Ws, out.bs = self.Ws, self.bs
        return out

    def value_and_jac(self, q):
        """Single q -> (d_red (n_rows,), Phi_D (n_rows,3))."""
        q = np.asarray(q, dtype=float).reshape(3)
        N, J = self.net_and_jac(q[None, :])
        d = self.PhiMA @ q + self.Phi_S @ N[0]
        Phi_D = self.PhiMA + self.Phi_S @ J[0]
        return d, Phi_D


def _self_test():
    import time
    import torch
    from train_nslave import build_net

    basis = ROOT / "04_training" / "decoder_basis_B_r39.npz"
    netf = ROOT / "04_training" / "nslave.npz"
    dec = NumpyDecoder(basis, netf)
    m = np.load(netf)
    tnet = build_net(3, dec.Phi_S.shape[1], width=int(m["width"]),
                     depth=int(m["depth"]))
    tnet.load_state_dict({k: torch.from_numpy(m[k]) for k in tnet.state_dict()})
    tnet.eval()

    rng = np.random.default_rng(0)
    Q = rng.uniform(-0.05, 0.2, size=(8, 3))
    N_np, J_np = dec.net_and_jac(Q)

    errs_n, errs_j = [], []
    for i in range(Q.shape[0]):
        x = torch.from_numpy((Q[i] - dec.mu_m) / dec.mu_s)
        with torch.no_grad():
            n_t = tnet(x[None, :]).numpy()[0]
        j_t = torch.autograd.functional.jacobian(
            lambda v: tnet(v[None, :]).reshape(-1), x).numpy() / dec.mu_s[None, :]
        errs_n.append(np.max(np.abs(N_np[i] - n_t)) / max(np.max(np.abs(n_t)), 1e-300))
        errs_j.append(np.max(np.abs(J_np[i] - j_t)) / max(np.max(np.abs(j_t)), 1e-300))

    # timing, single-point, the way the Newton loop uses it
    n_rep = 300
    t0 = time.perf_counter()
    for _ in range(n_rep):
        dec.value_and_jac(Q[0])
    t_np = (time.perf_counter() - t0) / n_rep
    x0 = torch.from_numpy((Q[0] - dec.mu_m) / dec.mu_s)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        torch.autograd.functional.jacobian(
            lambda v: tnet(v[None, :]).reshape(-1), x0)
    t_t = (time.perf_counter() - t0) / n_rep

    print(f"network output  max rel err {max(errs_n):.3e}")
    print(f"jacobian        max rel err {max(errs_j):.3e}")
    rows = rng.choice(dec.Phi_S.shape[0], 640, replace=False)
    dr = dec.restrict(rows)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        dr.value_and_jac(Q[0])
    t_r = (time.perf_counter() - t0) / n_rep
    d_full, P_full = dec.value_and_jac(Q[0])
    d_res, P_res = dr.value_and_jac(Q[0])
    e_r = max(np.max(np.abs(d_res - d_full[rows])) / np.max(np.abs(d_full)),
              np.max(np.abs(P_res - P_full[rows])) / np.max(np.abs(P_full)))
    print(f"restricted matches full rows: {e_r:.3e}")
    print(f"numpy value+jac {t_np * 1e6:8.1f} us/call")
    print(f"numpy restricted{t_r * 1e6:8.1f} us/call   -> {t_np / t_r:.1f}x")
    print(f"torch jacobian  {t_t * 1e6:8.1f} us/call   -> {t_t / t_np:.1f}x")
    ok = max(errs_n) < 1e-12 and max(errs_j) < 1e-10
    print("NUMPY_DECODER_PASS" if ok else "NUMPY_DECODER_FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _self_test() else 1)
