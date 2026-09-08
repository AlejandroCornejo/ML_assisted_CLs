#!/usr/bin/env python3
"""Clean force-controlled Linear-HPROM--FE2 coupon run.

This is the pure POD/ECM tier: 39 linear reduced coordinates, a fixed
135-element residual ECM mesh, and a distinct fixed 73-element stress ECM
mesh.  It intentionally contains no ANN, KAN, learned weights, or manifold.
The macro setup is identical to ``run_fom_fe2.py`` so its wall time and fields
can be compared directly to the clean FOM--FE2 reference.

Default: local element projection and batched stress differentiation.
--implementation baseline retains the original sparse implementation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve


# A worker owns exactly one Kratos/HPROM bundle.  The pool must be forked
# before the parent creates its macro Kratos model part.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon_fe2_mpl")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for _p in (str(ROOT), str(ROOT / "00_rve"), str(ROOT / "01_macro_prepass"),
           str(ROOT / "04_training"), str(ROOT / "05_validation"),
           str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg  # noqa: E402
from run_fom_fe2 import (  # noqa: E402
    _canonical_meshes,
    _coverage,
    _force_vector,
    _minimal_rbm_constraints,
    _sha256,
    _worker_bootstrap,
)


_FORK = multiprocessing.get_context("fork")
_WORKER_LAW = None
_WORKER_IMPLEMENTATION = "baseline"


def _worker_chunk(job):
    """Evaluate one batch on persistent, actual ECM MDPA model parts."""
    global _WORKER_LAW
    E, E_start, q_start = job
    if _WORKER_LAW is None:
        # Import only in a child after the pool was forked.  Its constructor
        # writes worker-private 135/73 element MDPA meshes below /tmp.
        if _WORKER_IMPLEMENTATION == "optimized":
            from linear_hprom_fast import FastLinearHPROMECM as LinearHPROMECM
        else:
            from linear_hprom_ecm import LinearHPROMECM
        _WORKER_LAW = LinearHPROMECM()

    E = np.asarray(E, dtype=float).reshape(-1, 3)
    S = np.empty_like(E)
    C = np.empty((E.shape[0], 3, 3))
    Q = np.empty((E.shape[0], _WORKER_LAW.n_modes))
    t0 = time.perf_counter()
    for i, e in enumerate(E):
        if E_start is None:
            S[i], C[i], Q[i] = _WORKER_LAW.stress_and_tangent(e, return_state=True)
        else:
            S[i], C[i], Q[i] = _WORKER_LAW.stress_and_tangent(
                e, q_init=q_start[i], E_start=E_start[i], return_state=True
            )
    return S, C, Q, time.perf_counter() - t0, _WORKER_LAW.ecm_metadata


class ParallelLinearHPROM:
    """Macro-law adapter with continuation per macro Gauss point."""

    def __init__(self, executor, workers):
        self.executor = executor
        self.workers = int(workers)
        self.calls = 0
        self.unique_queries = 0
        self.hprom_seconds = 0.0
        self._E_previous = None
        self._q_previous = None
        # One entry per worker-private pair of *actual* reduced MDPA meshes.
        # The first zero-strain broadcast reaches only one worker, so collect
        # metadata throughout the run rather than preserving only that entry.
        self.worker_metadata = {}

    def __call__(self, E_flat, young=None, poisson=None):
        del young, poisson
        E = np.asarray(E_flat, dtype=float).reshape(-1, 3)
        if self._E_previous is None or self._E_previous.shape != E.shape:
            # The very first macro assembly is exactly E=0 at every point;
            # solve that state only once and broadcast it, just as FOM--FE2.
            E_unique, inverse = np.unique(E, axis=0, return_inverse=True)
            chunks = [c for c in np.array_split(E_unique, min(self.workers, len(E_unique))) if c.size]
            results = list(self.executor.map(_worker_chunk, [(c, None, None) for c in chunks]))
            Su = np.concatenate([r[0] for r in results], axis=0)
            Cu = np.concatenate([r[1] for r in results], axis=0)
            Qu = np.concatenate([r[2] for r in results], axis=0)
            S, C, Q = Su[inverse], Cu[inverse], Qu[inverse]
            n_solved = int(len(E_unique))
        else:
            indices = np.arange(E.shape[0])
            chunks = [c for c in np.array_split(indices, min(self.workers, E.shape[0])) if c.size]
            results = list(self.executor.map(
                _worker_chunk,
                [(E[c], self._E_previous[c], self._q_previous[c]) for c in chunks],
            ))
            S = np.empty_like(E)
            C = np.empty((E.shape[0], 3, 3))
            Q = np.empty((E.shape[0], self._q_previous.shape[1]))
            for c, (s, cc, q, _elapsed, _meta) in zip(chunks, results):
                S[c], C[c], Q[c] = s, cc, q
            n_solved = int(E.shape[0])

        for meta in (r[4] for r in results):
            self.worker_metadata[meta["residual_mdpa"]] = meta
        self.calls += 1
        self.unique_queries += n_solved
        self.hprom_seconds += float(sum(r[3] for r in results))
        self._E_previous, self._q_previous = E.copy(), Q
        return S, C


def run(args) -> int:
    global _WORKER_IMPLEMENTATION
    _WORKER_IMPLEMENTATION = args.implementation
    macro_base, rve_base = _canonical_meshes(args.macro_divisor)
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=_FORK)
    pids = sorted(set(executor.map(_worker_bootstrap, range(args.workers))))
    if len(pids) != args.workers:
        executor.shutdown(cancel_futures=True)
        raise RuntimeError(f"requested {args.workers} workers but started {len(pids)}")

    import fom_solver_rve as fom
    from macro_prepass import MacroCoupon

    law = ParallelLinearHPROM(executor, args.workers)
    original_law = fom._neo_hookean_pk2_2d_vectorized
    macro = MacroCoupon(macro_base, verbose=True)
    f_final = _force_vector(macro, args.force)
    fixed = _minimal_rbm_constraints(macro)
    free_mask = np.ones(macro.n_dof, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    grid = np.load(ROOT / "02_sampling" / "train_grid.npz")
    blo, bhi = np.asarray(grid["blo"]), np.asarray(grid["bhi"])
    u = np.zeros(macro.n_dof)
    records, cloud = [], []
    cached_assembly = None
    t0 = time.perf_counter()  # clean online timing: setup intentionally excluded

    try:
        fom._neo_hookean_pk2_2d_vectorized = law
        for step in range(1, args.max_steps + 1):
            f_ext = f_final * (step / args.n_steps)
            res0, converged = None, False
            best = (np.inf, u.copy())
            for it in range(1, args.max_newton + 1):
                tic = time.perf_counter()
                if cached_assembly is None:
                    K, rhs = macro.assembler.Assemble(u)
                else:
                    K, rhs = cached_assembly
                    cached_assembly = None
                residual = rhs + f_ext
                res = float(np.linalg.norm(residual[free]))
                res0 = max(res, 1.0e-30) if res0 is None else res0
                rel = res / res0
                best = min(best, (res, u.copy()), key=lambda x: x[0])
                print(
                    f"  step {step:2d}/{args.n_steps} iter {it:2d}: |R|={res:.5e}, "
                    f"rel={rel:.3e}, wall={time.perf_counter() - tic:.1f}s",
                    flush=True,
                )
                if res < args.abs_tol or rel < args.rel_tol:
                    converged = True
                    break
                du = spsolve(K[free, :][:, free].tocsc(), residual[free])
                if not np.all(np.isfinite(du)):
                    raise RuntimeError(f"non-finite macro update at step {step}, iter {it}")
                u[free] += du
                cached_assembly = None
            if not converged:
                u = best[1]
                raise RuntimeError(
                    f"macro Newton did not converge at step {step}; best relative residual {best[0] / res0:.3e}"
                )

            E = macro.assembler._E_voigt.reshape(-1, 3).copy()
            cov = _coverage(E, blo, bhi)
            cloud.append(E)
            records.append(
                dict(step=step, iterations=it, residual=res, relative_residual=rel, coverage=cov)
            )
            cached_assembly = (K, rhs)
            print(
                f"    converged: in-box {cov['inside']}/{cov['n']}; "
                f"E11=[{cov['minimum'][0]:+.4f},{cov['maximum'][0]:+.4f}], "
                f"E22=[{cov['minimum'][1]:+.4f},{cov['maximum'][1]:+.4f}], "
                f"g12=[{cov['minimum'][2]:+.4f},{cov['maximum'][2]:+.4f}]",
                flush=True,
            )

        wall = time.perf_counter() - t0
        E_final = macro.assembler._E_voigt.reshape(-1, 3).copy()
        S_final = macro.assembler._S_voigt.reshape(-1, 3).copy()
        u_nodes = np.stack((u[macro.eq_map[:, 0]], u[macro.eq_map[:, 1]]), axis=1)
        tag = args.tag or f"hprom_ecm_w{args.workers}_w{args.macro_divisor:g}_f{args.force / 1e3:g}kn"
        if not args.tag and args.implementation == "optimized":
            tag += "_optimized"
        out = HERE / f"hprom_fe2_{tag}.npz"
        summary = dict(
            status="converged", model="linear_hprom_fixed_ecm", tag=tag,
            implementation=args.implementation,
            output=str(out), wall_seconds=wall, workers=args.workers, worker_pids=pids,
            macro_elements=macro.assembler.n_elems,
            macro_gauss_points=macro.assembler.n_elems * macro.assembler.n_gauss,
            rve_elements_full=int(np.load(ROOT / "03_data" / "data.npz")["n_elements"]),
            hprom_modes=39, residual_ecm_elements=135, stress_ecm_elements=73,
            # The two fixed rules share 25 selected elements: 135 + 73 - 25.
            total_distinct_ecm_elements=183,
            force_per_end=args.force, n_steps_requested=args.n_steps,
            n_steps_completed=len(records), macro_newton=records,
            material_calls=law.calls, unique_hprom_queries=law.unique_queries,
            summed_hprom_seconds=law.hprom_seconds,
            worker_ecm_mdpa=list(law.worker_metadata.values()),
            macro_relative_tolerance=args.rel_tol, macro_absolute_tolerance=args.abs_tol,
            timing_scope="macro solve including constitutive tangents, IPC and lazy worker model loading; excludes parent setup and result export",
            source_sha256={name: _sha256(HERE / name) for name in
                           ("run_hprom_fe2.py", "linear_hprom_ecm.py", "linear_hprom_fast.py", "reduced_stress_batch.py")},
            macro_mesh_sha256=_sha256(str(macro_base) + ".mdpa"),
            rve_mesh_sha256=_sha256(str(rve_base) + ".mdpa"),
        )
        if not args.no_output:
            np.savez_compressed(
                out, coords=macro.xy, connectivity=macro.assembler.connectivity,
                u_nodal=u_nodes, E_final=E_final, S_final=S_final,
                E_path=np.concatenate(cloud, axis=0), force_per_end=args.force,
                n_steps_requested=args.n_steps, n_steps_completed=len(records),
                workers=args.workers, macro_divisor=args.macro_divisor,
            )
            out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    finally:
        fom._neo_hookean_pk2_2d_vectorized = original_law
        executor.shutdown(wait=True, cancel_futures=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    p.add_argument("--macro-divisor", type=float, default=4.0)
    p.add_argument("--force", type=float, default=1.0e5)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=20,
                   help="run the first N load steps; use 1 only as a smoke test")
    p.add_argument("--max-newton", type=int, default=12)
    p.add_argument("--rel-tol", type=float, default=1.0e-7)
    p.add_argument("--abs-tol", type=float, default=1.0e-5)
    p.add_argument("--tag", default="")
    p.add_argument("--implementation", choices=("baseline", "optimized"), default="optimized",
                   help="equivalent sparse baseline or local/batched ECM implementation")
    p.add_argument("--no-output", action="store_true", help="do not save a smoke-test result")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    if args.n_steps < 1 or not 1 <= args.max_steps <= args.n_steps:
        p.error("require 1 <= --max-steps <= --n-steps")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
