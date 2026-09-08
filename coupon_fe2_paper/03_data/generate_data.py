#!/usr/bin/env python3
"""Stage 03: solve the RVE at every sampled state and record the data.

Produces, for the 4951 TRAINING states:
    E (3,)          the macro Green-Lagrange strain
    S (3,)          the homogenized 2nd PK stress, per unit CELL volume
    W (scalar)      the homogenized strain energy density, same normalization
    u_ind (n_ind,)  the periodic snapshot, i.e. Hernandez's d_red

and for the 400 test and 350 probe states, E, S and W only -- snapshots exist
to build the POD basis, which uses training states alone.

WARM-STARTING AND PARALLELISM PULL AGAINST EACH OTHER, and the resolution
matters for the cost. Warm-starting is inherently sequential: each state
continues from the previous one. Substeps are allocated proportional to the
strain increment, so a grid step of 0.0115 needs ~3 substeps against ~55 for a
from-zero ramp. The grid is therefore traversed as a BOUSTROPHEDON (snake) so
consecutive states are always one grid step apart, and parallelism comes from
splitting along the E11 axis: one slice per worker, each slice cold-started
once at its head and snaked warm thereafter. 18 cold starts instead of 4951.

Test and probe states are scattered, so they are solved cold; at ~7 s each
that is minutes across workers, and not worth the bookkeeping of finding a
warm neighbour for each.

NORMALIZATION AND HOMOGENIZATION are both handled inside PeriodicRVE, and both
had to be corrected once. S comes from Fbar^-1 <P>, the first Piola-Kirchhoff
average, not from averaging the microscopic 2nd PK stress -- the latter is
wrong by O(fluctuation^2) and violated S = dW/dE by 0.5-0.7%. W uses the same
w_detJ quadrature as the stress. Both are per unit CELL volume, void included.
With these, the finite-strain gate reports S = dW/dE to 1.2e-06 and dS/dE
symmetric to 4.1e-10.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(ROOT / "00_rve"), str(ROOT / "02_sampling"),
          str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

_FORK = multiprocessing.get_context("fork")
MESH_BASE = str(HERE / "rve_mesh")
N_WORKERS = 16


def snake_order(n22, n12):
    """Boustrophedon over a 2D slice, so consecutive entries are neighbours."""
    out = []
    for i in range(n22):
        rng = range(n12) if i % 2 == 0 else range(n12 - 1, -1, -1)
        out += [(i, k) for k in rng]
    return out


def _solve_sequence(args):
    """Solve a list of states in order, warm-starting each from the previous.

    Returns (S, W, U or None, n_fail, seconds). U is (n, n_ind) when snapshots
    are requested.
    """
    E_list, keep_u, cell_area, label = args
    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active

    E_list = np.asarray(E_list, dtype=float).reshape(-1, 3)
    t0 = time.perf_counter()
    with true_neo_hookean_active():
        rve = PeriodicRVE(MESH_BASE, cell_area=cell_area)
        n = E_list.shape[0]
        S = np.full((n, 3), np.nan)
        W = np.full(n, np.nan)
        U = np.full((n, rve.T.shape[1]), np.nan) if keep_u else None
        prev_u, prev_E, n_fail = None, None, 0

        for i in range(n):
            try:
                s, u = rve.solve(E_list[i], u_ind_init=prev_u, E_start=prev_E)
            except RuntimeError:
                # Cold retry: a warm start can fail where a from-zero ramp
                # succeeds, since the ramp is the Newton aid.
                try:
                    s, u = rve.solve(E_list[i])
                except RuntimeError:
                    n_fail += 1
                    prev_u, prev_E = None, None
                    continue
            S[i] = s
            W[i] = rve.homogenized_energy()
            if keep_u:
                U[i] = u
            prev_u, prev_E = u, E_list[i]
    return S, W, U, n_fail, time.perf_counter() - t0, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--limit-slices", type=int, default=0,
                    help="solve only the first N E11 slices (smoke test)")
    a = ap.parse_args()

    # Pool BEFORE the parent touches Kratos: same fork-after-threading
    # discipline every other parallel driver in this project follows.
    ex = ProcessPoolExecutor(max_workers=a.workers, mp_context=_FORK)

    from gen_rve_mesh import build_mesh, write_mdpa
    coords, tris, outer, geom = build_mesh(periodic=True)
    write_mdpa(MESH_BASE + ".mdpa", coords, tris, outer)
    cell_area = geom["block_area"]
    print(f"mesh: {geom['n_elements']} elements, cell area {cell_area}", flush=True)

    g = np.load(ROOT / "02_sampling" / "train_grid.npz")
    ev = np.load(ROOT / "02_sampling" / "eval_sets.npz")
    E_tr, shape = g["E"], tuple(int(v) for v in g["shape"])
    n_s, n_22, n_12 = shape

    # The zero state sits at row 0 and is not part of the tensor grid.
    has_zero = E_tr.shape[0] == int(np.prod(shape)) + 1
    E_grid = E_tr[1:] if has_zero else E_tr
    G = E_grid.reshape(n_s, n_22, n_12, 3)

    order = snake_order(n_22, n_12)
    slices = list(range(n_s))
    if a.limit_slices:
        slices = slices[:a.limit_slices]
    jobs = []
    idx_map = []
    for si in slices:
        seq = np.array([G[si, i, k] for (i, k) in order])
        jobs.append((seq, True, cell_area, f"slice{si}"))
        idx_map.append([(si, i, k) for (i, k) in order])

    if not a.limit_slices:
        for nm, arr in (("test", ev["test"]), ("probe", ev["probe"]),
                        ("zero", E_tr[:1] if has_zero else np.zeros((1, 3)))):
            chunks = np.array_split(arr, max(1, min(a.workers, len(arr))))
            for c, ch in enumerate(chunks):
                if len(ch):
                    jobs.append((ch, False, cell_area, f"{nm}:{c}"))

    print(f"{len(jobs)} jobs on {a.workers} workers "
          f"({len(slices)} training slices of {len(order)} states each)", flush=True)

    t0 = time.perf_counter()
    results = []
    for r in ex.map(_solve_sequence, jobs):
        results.append(r)
        print(f"  [{r[5]}] {r[0].shape[0]} states, {r[3]} failed, {r[4]:.1f}s",
              flush=True)
    ex.shutdown()
    wall = time.perf_counter() - t0

    # Reassemble the training arrays in grid order
    n_ind = next(r[2].shape[1] for r in results if r[2] is not None)
    S_tr = np.full((len(slices), n_22, n_12, 3), np.nan)
    W_tr = np.full((len(slices), n_22, n_12), np.nan)
    U_tr = np.full((len(slices), n_22, n_12, n_ind), np.nan, dtype=np.float64)
    n_fail = 0
    ptr = 0
    for si_local, si in enumerate(slices):
        r = results[ptr]
        ptr += 1
        n_fail += r[3]
        for j, (_s, i, k) in enumerate(idx_map[si_local]):
            S_tr[si_local, i, k] = r[0][j]
            W_tr[si_local, i, k] = r[1][j]
            U_tr[si_local, i, k] = r[2][j]

    out = dict(E_train=G[slices].reshape(-1, 3),
               S_train=S_tr.reshape(-1, 3), W_train=W_tr.reshape(-1),
               U_train=U_tr.reshape(-1, n_ind), shape=np.array(shape),
               cell_area=cell_area, n_elements=geom["n_elements"])
    if not a.limit_slices:
        rest = results[ptr:]
        for nm in ("test", "probe", "zero"):
            sel = [r for r in rest if r[5].startswith(nm)]
            sel.sort(key=lambda r: int(r[5].split(":")[1]))
            out[f"S_{nm}"] = np.concatenate([r[0] for r in sel], axis=0)
            out[f"W_{nm}"] = np.concatenate([r[1] for r in sel], axis=0)
            n_fail += sum(r[3] for r in sel)
        out["E_test"], out["E_probe"] = ev["test"], ev["probe"]
        out["labels_probe"] = ev["labels"]

    np.savez_compressed(HERE / "data.npz", **out)
    n_total = sum(r[0].shape[0] for r in results)
    print(f"\n{n_total} states, {n_fail} failed, {wall / 60:.1f} min wall "
          f"({wall / max(n_total, 1):.2f} s/state effective)")
    print(f"snapshots: {U_tr.reshape(-1, n_ind).shape} "
          f"({U_tr.nbytes / 1e6:.0f} MB uncompressed)")
    print("DATA_ALL_CONVERGED" if n_fail == 0 else f"DATA_FAILURES={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
