#!/usr/bin/env python3
"""MAW-ECM laboratory: supports, weight fields, and the comparisons that matter.

Three things were wrong with how MAW-ECM was being evaluated, and this module
fixes all three.

1. THE BASELINE WAS UNFAIR TO THE WRONG SIDE. MAW-ECM at 10 points was compared
   against the classic fixed-weight rule at 40 (residual) and 73 (stress)
   points -- 4x and 7x the cost. The question worth asking is which rule wins
   AT EQUAL POINT COUNT, and what each one's accuracy-vs-cost curve looks like.

2. THE FIELD WAS FIT TO THE WRONG TARGET. `fit_mawecm_ann` minimizes

       loss_kl  +  10 * loss_mse  +  physics_weight * loss_phys

   where the first two terms match the PRUNED WEIGHT VALUES and only the third
   is constraint satisfaction. At the previous project's physics_weight = 1.0
   the objective we actually care about carries ~1% of the loss. Raising it
   helped a lot (residual 8.4e-01 -> 1.0e-01), but it is still a weighted
   compromise between an intermediate and the objective.

   The intermediate is not needed at all. Constraint satisfaction requires only
   A(q) and b(q), and b is the full-mesh integral, which `full_integrand.npz`
   now stores at every state. So the field can be trained on the objective
   DIRECTLY, with no target weights anywhere in the loss. The pruned weights
   are still what SELECTS the support -- that is what MAW-ECM is for -- they
   are just not used as a regression target afterwards.

3. THE FIT WAS STARVED OF DATA. The pipeline used 495 states (421 to fit)
   because that is what the pruning subsampled to. There are 4950. For a
   3-input regression asked to generalize, 421 samples is ~7.5 per dimension.
   The pruning still runs on a subsample -- its cost scales with states -- but
   the FIELD is trained on all of them, which is free now that b is stored.

BOTH STRUCTURAL GUARANTEES SURVIVE, which is the whole reason for the softmax
parametrization and the property the classic rule's 1.2x out-of-envelope
degradation rested on:

    w(q) = n_elements * softmax(logits(q))    =>    w >= 0 and sum(w) = n_elements

exactly, for any q, however wrong the logits. Note sum(w) = n_elements is
precisely the volume row of the constraint system, so that row is satisfied by
construction and only the remaining 3 (residual) or 4 (stress) rows have to be
learned.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(PROJ / "mawecm"), str(PROJ / "fe2_extension"),
          str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

VAL_FRAC = 0.15
SPLIT_SEED = 5


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load():
    f = np.load(HERE / "full_integrand.npz")
    return dict(C_res=f["C_res"], C_sig=f["C_sig"], q=f["q_train"],
                ne=int(f["n_elements"]))


def targets(C):
    """b[k] = full-mesh sum, with the volume row appended."""
    ne, n, m = C.shape
    b = np.empty((n, m + 1))
    b[:, :m] = C.sum(axis=0)
    b[:, m] = float(ne)
    return b


def blocks(C, z):
    """A[k] (m+1, |z|): integrand rows on the support, plus the volume row."""
    ne, n, m = C.shape
    k = np.asarray(z, dtype=np.int64).size
    A = np.empty((n, m + 1, k))
    A[:, :m, :] = np.transpose(C[np.asarray(z, dtype=np.int64)], (1, 2, 0))
    A[:, m, :] = 1.0
    return A


def split(n):
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    nv = int(VAL_FRAC * n)
    return perm[nv:], perm[:nv]          # fit, held out


def const_err(A, b, W):
    """Per-state relative constraint error. W is (k,) fixed or (k, n) adaptive."""
    if W.ndim == 1:
        r = np.einsum("kmj,j->km", A, W) - b
    else:
        r = np.einsum("kmj,jk->km", A, W) - b
    return np.linalg.norm(r, axis=1) / np.maximum(
        np.linalg.norm(b, axis=1), 1e-300)


# --------------------------------------------------------------------------
# classic ECM at a prescribed rank -- the equal-cost baseline
# --------------------------------------------------------------------------
def integrand_svd(C, states=None):
    """Left singular vectors of the element-wise integrand.

    A direct SVD of the (ne, n_states * m) matrix rather than the
    eigendecomposition of C C^T: the method of snapshots squares the condition
    number, which already cost this project an orthogonality failure in the
    displacement POD.

    Cached deliberately by the caller. This is a (1546 x 12624) SVD, ~1 min,
    and calling classic_ecm once per point count recomputed it every time --
    which was the actual cost of the first sweep, not the network training it
    was easy to blame.
    """
    M = C if states is None else C[:, states, :]
    M = np.ascontiguousarray(M.reshape(M.shape[0], -1))
    # PINNED TO ONE THREAD, DELIBERATELY. Measured: the same call under
    # OMP_NUM_THREADS=8 and =6 returns different left singular vectors, the
    # greedy ECM then selects a different support, and the resulting baseline
    # error moves 35% (3.20e-01 vs 4.32e-01 for the 20-point residual rule).
    # Each answer is individually valid -- threaded LAPACK is free to return a
    # different basis for a near-degenerate singular subspace -- but it makes
    # every ECM number in the project depend on an environment variable, and it
    # already produced two different values for the same baseline in two of
    # these scripts. Reproducibility is worth more here than the seconds.
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            U, sv, _ = np.linalg.svd(M, full_matrices=False)
    except ImportError:
        U, sv, _ = np.linalg.svd(M, full_matrices=False)
    return U, sv


def classic_ecm(C, rank, states=None, tol=1e-6, U=None):
    """Fixed-weight ECM on the rank-truncated integrand. Support size ~ rank."""
    import contextlib
    import io

    from hprom_full import run_ecm
    sv = None
    if U is None:
        U, sv = integrand_svd(C, states)
    # The ECM implementation prints one line per greedy iteration, which for a
    # sweep over ranks buries the results table.
    with contextlib.redirect_stdout(io.StringIO()):
        z, w = run_ecm(np.ascontiguousarray(U[:, :int(rank)]), tol=tol)
    o = np.argsort(z)
    return z[o], w[o], sv


# --------------------------------------------------------------------------
# per-state optimal weights: a NON-ARBITRARY regression target
# --------------------------------------------------------------------------
def optimal_weights(A, b, target_sum, w0=None, floor=1e-9):
    """Per state, the weights closest to a smooth reference that satisfy the
    constraints exactly, subject to w >= 0.

        min ||w - w0||^2   s.t.   A w = b,  w >= 0

    WHY THIS EXISTS. Training the field directly on constraint satisfaction is
    the right objective but a badly conditioned one: measured, the loss fell
    from 8.2e+01 to 8.5e-01 over 21000 epochs and was still decreasing. And
    fitting the PRUNED weights instead is worse than arbitrary -- with 10
    weights against 4 constraints the pruning's answer is one point in a
    7-dimensional solution set, chosen by the elimination order, so
    neighbouring states can be handed wildly different weight vectors that
    are equally valid. A field cannot regress that.

    The minimum-deviation solution removes exactly that freedom. It is unique,
    and it varies continuously with (A, b), hence smoothly with q away from the
    non-negativity boundary -- which is what makes it regressable. Used as a
    warm-start target, after which the physics loss takes over and the
    intermediate is dropped entirely.

    w0 defaults to the uniform rule target_sum / k, so the reference carries no
    information beyond volume conservation.
    """
    n, m, k = A.shape
    if w0 is None:
        w0 = np.full(k, float(target_sum) / k)
    W = np.empty((k, n))
    n_clip = 0
    for j in range(n):
        Aj, bj = A[j], b[j]
        # KKT solution of the equality-constrained problem
        G = Aj @ Aj.T
        try:
            lam = np.linalg.solve(G, bj - Aj @ w0)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(G, bj - Aj @ w0, rcond=None)[0]
        w = w0 + Aj.T @ lam
        if w.min() < floor:
            # Fall back to a bounded solve, with the equality rows weighted up
            # so they are met as closely as non-negativity allows.
            from scipy.optimize import lsq_linear
            sc = np.maximum(np.linalg.norm(Aj, axis=1), 1e-300)
            Aw = np.vstack([Aj / sc[:, None] * 1.0e3, np.eye(k) * 1.0e-3])
            bw = np.concatenate([bj / sc * 1.0e3, w0 * 1.0e-3])
            w = lsq_linear(Aw, bw, bounds=(floor, np.inf),
                           tol=1e-12, max_iter=200).x
            n_clip += 1
        W[:, j] = w
    return W, n_clip


# --------------------------------------------------------------------------
# the weight field, trained on the objective itself
# --------------------------------------------------------------------------
def fit_field(q, A, b, target_sum, fit_idx, val_idx, hidden=(128, 128, 128),
              act="gelu", epochs=12000, lr=2e-3, patience=1200, seed=11,
              batch=None, verbose=False, label="", warm_target=None,
              warm_epochs=4000, warm_lr=3e-3, log_every=1000,
              sched_patience=1000, select_by="median"):
    """w(q) = target_sum * softmax(net(q)), trained to satisfy A w = b.

    Stage 2's loss IS the reported metric (mean of the squared relative
    constraint error), so there is no gap between what is minimized and what is
    measured. If `warm_target` is given, stage 1 first regresses those weights
    to condition the optimization; the target is then dropped and plays no part
    in the reported result. See optimal_weights for why a warm start is needed
    and why the pruned weights are the wrong thing to warm-start from.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    dev = "cpu"

    mu, sd = q.mean(axis=0), q.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    X = torch.tensor((q - mu) / sd, dtype=torch.float64)
    At = torch.tensor(A, dtype=torch.float64)
    bt = torch.tensor(b, dtype=torch.float64)
    bn = torch.clamp(torch.sum(bt ** 2, dim=1), min=1e-300)

    acts = dict(gelu=nn.GELU, silu=nn.SiLU, tanh=nn.Tanh, elu=nn.ELU)
    layers, d0 = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(d0, h), acts[act]()]
        d0 = h
    layers += [nn.Linear(d0, A.shape[2])]
    net = nn.Sequential(*layers).to(dev).double()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # SCHEDULER PATIENCE IS INDEPENDENT OF EARLY-STOPPING PATIENCE, and it has
    # to be. It used to be `patience // 4`, so disabling early stopping by
    # passing patience = 100000 silently gave the scheduler a patience of 25000
    # epochs -- longer than the worst stall in a 100k run (17449 epochs), which
    # meant the learning rate was NEVER reduced across the whole run. The
    # symptoms were then read as difficulty of the problem: a loss oscillating
    # at 32x its own best, and three apparent plateaus that were really a fixed
    # step size bouncing around the optimum.
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(1, int(sched_patience)), min_lr=1e-7)

    fi = torch.tensor(fit_idx, dtype=torch.long)
    vi = torch.tensor(val_idx, dtype=torch.long)
    # SELECT AND REPORT ON DISJOINT HALVES of the held-out states. Model
    # selection is itself a fit to whatever set it watches, so choosing the
    # best epoch on the same states the result is quoted from makes that result
    # optimistic. The FIT set is left untouched so every earlier number stays
    # comparable; only the held-out block is halved.
    _v = np.asarray(val_idx)
    _h = _v.size // 2
    si = torch.tensor(_v[:_h], dtype=torch.long)      # selection
    ri = torch.tensor(_v[_h:], dtype=torch.long)      # reporting

    def loss_on(idx):
        w = float(target_sum) * torch.softmax(net(X[idx]), dim=1)
        r = torch.bmm(At[idx], w.unsqueeze(2)).squeeze(2) - bt[idx]
        return torch.mean(torch.sum(r ** 2, dim=1) / bn[idx])

    def median_on(idx):
        """The metric every table quotes: MEDIAN relative constraint error.

        Tracked separately from the loss because the two genuinely disagree.
        Measured on the 10-point residual rule: the loss froze at epoch 16160
        and rose monotonically from epoch 45000 onward, while the median fell
        from 5e-02 to 1.97e-02 over the same 55000 epochs. Selecting the model
        by loss therefore kept a model 2.2x worse in the reported metric than
        one the run had already visited.
        """
        w = float(target_sum) * torch.softmax(net(X[idx]), dim=1)
        r = torch.bmm(At[idx], w.unsqueeze(2)).squeeze(2) - bt[idx]
        return torch.median(torch.linalg.norm(r, dim=1) / torch.sqrt(bn[idx]))

    t0 = time.perf_counter()
    warm_hist = None
    if warm_target is not None and warm_epochs > 0:
        Wt = torch.tensor(np.asarray(warm_target).T / float(target_sum),
                          dtype=torch.float64)
        wopt = torch.optim.Adam(net.parameters(), lr=warm_lr)
        for ep in range(1, int(warm_epochs) + 1):
            wopt.zero_grad()
            lp = torch.log_softmax(net(X[fi]), dim=1)
            # KL(target || pred): scale-free in the weights and better behaved
            # than MSE when the target distribution is peaked, which it is --
            # a 10-point rule reproducing a 1546-element integral concentrates
            # most of the volume on a few elements.
            wl = torch.sum(Wt[fi] * (torch.log(torch.clamp(Wt[fi], min=1e-300))
                                     - lp), dim=1).mean()
            wl.backward()
            wopt.step()
            if verbose and ep % int(log_every) == 0:
                print(f"    {label} warm ep {ep:6d}  KL {float(wl):.4e}",
                      flush=True)
        warm_hist = float(wl)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=max(1, int(sched_patience)),
            min_lr=1e-7)

    best, best_state, best_ep = np.inf, None, 0
    nb = fi.numel() if batch is None else int(batch)
    for ep in range(1, int(epochs) + 1):
        net.train()
        if nb >= fi.numel():
            opt.zero_grad()
            loss_on(fi).backward()
            opt.step()
        else:
            perm = fi[torch.randperm(fi.numel())]
            for s in range(0, perm.numel(), nb):
                opt.zero_grad()
                loss_on(perm[s:s + nb]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            v = (float(median_on(si)) if select_by == "median"
                 else float(loss_on(si)))
        sch.step(v)
        if v < best - 1e-14:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        elif ep - best_ep > patience:
            break
        if verbose and ep % int(log_every) == 0:
            # Report the MEDIAN RELATIVE CONSTRAINT ERROR alongside the loss:
            # the loss is a MEAN of SQUARED relative errors, so it is dominated
            # by the worst states and is not the number any table quotes.
            with torch.no_grad():
                ls = float(loss_on(si))
                med = float(median_on(ri))
            print(f"    {label} ep {ep:6d}  sel {v:.4e}  best {best:.4e}"
                  f"@{best_ep}  loss {ls:.4e}  report {med:.4e}", flush=True)
    if best_state is not None:
        net.load_state_dict(best_state)

    with torch.no_grad():
        W = (float(target_sum) * torch.softmax(net(X), dim=1)).numpy().T
    with torch.no_grad():
        med_rep = float(median_on(ri))
        med_sel = float(median_on(si))
    return dict(W=W, mu=mu, sd=sd, best_val=best, best_epoch=best_ep,
                warm_kl=warm_hist, select_by=select_by,
                median_report=med_rep, median_select=med_sel,
                report_idx=np.asarray(val_idx)[_h:],
                select_idx=np.asarray(val_idx)[:_h],
                seconds=time.perf_counter() - t0,
                state={k: v.numpy() for k, v in net.state_dict().items()},
                hidden=np.asarray(hidden), act=act, target_sum=float(target_sum))


def field_weights(model, q):
    """Evaluate a fitted field at arbitrary q, in numpy."""
    x = (np.atleast_2d(q) - model["mu"]) / model["sd"]
    st, h = model["state"], x
    ks = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
    # torch.nn.GELU defaults to the EXACT erf form, not the tanh
    # approximation; using the approximation here left a 1.0e-04 mismatch
    # against torch, which is 4 orders above what a numpy re-implementation of
    # the same network should show.
    from math import sqrt as _sqrt
    from scipy.special import erf as _erf
    fn = dict(gelu=lambda v: 0.5 * v * (1.0 + _erf(v / _sqrt(2.0))),
        silu=lambda v: v / (1.0 + np.exp(-v)),
        tanh=np.tanh, elu=lambda v: np.where(v > 0, v, np.expm1(v)))[model["act"]]
    for i, k in enumerate(ks):
        h = h @ st[f"{k}.weight"].T + st[f"{k}.bias"]
        if i < len(ks) - 1:
            h = fn(h)
    e = np.exp(h - h.max(axis=1, keepdims=True))
    return (float(model["target_sum"]) * e / e.sum(axis=1, keepdims=True)).T


def report(tag, A, b, W, vi, extra=""):
    e = const_err(A[vi], b[vi], W[:, vi] if W.ndim == 2 else W)
    s = W.sum(axis=0)
    print(f"  {tag:<34} pts {W.shape[0]:3d}  median {np.median(e):.4e}  "
          f"p90 {np.percentile(e, 90):.4e}  max {e.max():.4e}  "
          f"w>=0 {bool(np.all(W >= -1e-9))}  "
          f"sum {np.min(s):.2f}..{np.max(s):.2f} {extra}", flush=True)
    return float(np.median(e))
