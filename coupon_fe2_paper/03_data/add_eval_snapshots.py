#!/usr/bin/env python3
"""Solve the test and probe states again, KEEPING their snapshots.

The first pass stored only S and W for these, to save space. That was enough
to measure end-to-end stress error, but not enough to ATTRIBUTE it: an
end-to-end error of, say, 1e-3 could come from the network's regression or
from the POD basis failing to represent those states, and the two call for
opposite responses. With the snapshots, the error decomposes into

    POD projection error at the eval states   -> the basis
    q_S regression error at the eval states   -> the network
    end-to-end homogenized stress error       -> the total

Written to a separate file so the 240 MB data.npz is not rewritten.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import multiprocessing, sys, time
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
sys.path.append("/home/kratos/Kratos_Eigen_Check/bin/Release")

_FORK = multiprocessing.get_context("fork")
MESH_BASE = str(HERE / "rve_mesh")
N_WORKERS = 16


def _worker(args):
    E_list, cell_area, label = args
    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    E_list = np.asarray(E_list, dtype=float).reshape(-1, 3)
    t0 = time.perf_counter()
    with true_neo_hookean_active():
        rve = PeriodicRVE(MESH_BASE, cell_area=cell_area)
        n = E_list.shape[0]
        S = np.full((n, 3), np.nan)
        W = np.full(n, np.nan)
        U = np.full((n, rve.T.shape[1]), np.nan)
        nf = 0
        for i in range(n):
            try:
                s, u = rve.solve(E_list[i])
            except RuntimeError:
                nf += 1
                continue
            S[i], U[i] = s, u
            W[i] = rve.homogenized_energy()
    return S, W, U, nf, time.perf_counter() - t0, label


if __name__ == "__main__":
    ex = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=_FORK)
    d = np.load(HERE / "data.npz")
    cell_area = float(d["cell_area"])
    out = {}
    jobs, meta = [], []
    for nm in ("test", "probe"):
        arr = d[f"E_{nm}"]
        chunks = np.array_split(arr, N_WORKERS)
        for c, ch in enumerate(chunks):
            if len(ch):
                jobs.append((ch, cell_area, f"{nm}:{c}"))
                meta.append(nm)
    print(f"{len(jobs)} jobs, {sum(len(j[0]) for j in jobs)} states", flush=True)
    t0 = time.perf_counter()
    res = list(ex.map(_worker, jobs))
    ex.shutdown()
    nf = 0
    for nm in ("test", "probe"):
        sel = [(m, r) for m, r in zip(meta, res) if m == nm]
        sel.sort(key=lambda mr: int(mr[1][5].split(":")[1]))
        out[f"S_{nm}"] = np.concatenate([r[0] for _m, r in sel], axis=0)
        out[f"W_{nm}"] = np.concatenate([r[1] for _m, r in sel], axis=0)
        out[f"U_{nm}"] = np.concatenate([r[2] for _m, r in sel], axis=0)
        nf += sum(r[3] for _m, r in sel)
    wall = time.perf_counter() - t0
    # Consistency: the freshly solved stresses must match the stored ones.
    for nm in ("test", "probe"):
        a, b = out[f"S_{nm}"], d[f"S_{nm}"]
        m = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        r = np.max(np.linalg.norm(a[m] - b[m], axis=1) / np.linalg.norm(b[m], axis=1))
        print(f"  {nm}: S matches the stored pass to {r:.3e}", flush=True)
    np.savez_compressed(HERE / "eval_snapshots.npz", **out)
    print(f"{nf} failed, {wall/60:.1f} min", flush=True)
    print("EVAL_SNAPSHOTS_DONE")
