#!/usr/bin/env python3
"""Clean MAW-HPROM-ANN--FE2 coupon run on actual 10 + 10 MDPA meshes.

This is the iterative nonlinear-manifold tier: a three-coordinate ANN decoder
solves a reduced RVE equilibrium on a 10-element residual mesh and evaluates
homogenized stress on a separate 10-element mesh.  Both MAW-ECM weight fields
are adaptive, non-negative, and volume conserving.  It is intentionally not
the linear 39-mode 135 + 73 ECM HPROM.

The default iterative implementation projects local element arrays and batches
the IFT stencil.  --implementation baseline retains the original sparse
assembly for reproducibility; the equations and tolerances are identical.
With --direct the optimized implementation batches closure/stress evaluations
over macro points and the same seven-point finite-difference stencil.
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


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon_fe2_mpl")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for candidate in (ROOT, ROOT / "00_rve", ROOT / "01_macro_prepass",
                  ROOT / "04_training", ROOT / "05_validation",
                  PROJ / "fe2_extension", PROJ / "core"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
_kratos_candidates = (
    Path("/home/sares/Kratos_Eigen_Check/bin/Release"),
    Path("/home/kratos/Kratos_Eigen_Check/bin/Release"),
)
KRATOS_PATH = next((path for path in _kratos_candidates if path.is_dir()), _kratos_candidates[0])
if str(KRATOS_PATH) not in sys.path:
    sys.path.append(str(KRATOS_PATH))

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
_WORKER_MODEL = "iterative"
_WORKER_IMPLEMENTATION = "baseline"


def _worker_chunk(job):
    """Evaluate one batch on persistent worker-private MAW reduced meshes."""
    global _WORKER_LAW
    E, E_start, q_start = job
    if _WORKER_LAW is None:
        if _WORKER_MODEL == "iterative":
            if _WORKER_IMPLEMENTATION == "optimized":
                from maw_hprom_ann_fast import FastMAWHPROMANN as MAWHPROMANN
            else:
                from maw_hprom_ann_law import MAWHPROMANN
            _WORKER_LAW = MAWHPROMANN()
        elif _WORKER_MODEL == "direct":
            if _WORKER_IMPLEMENTATION == "optimized":
                from direct_hprom_ann_fast import FastMAWDHPROMANN as MAWDHPROMANN
            else:
                from direct_hprom_ann_law import MAWDHPROMANN
            _WORKER_LAW = MAWDHPROMANN()
        else:
            raise RuntimeError(f"unknown manifold model {_WORKER_MODEL!r}")

    E = np.asarray(E, dtype=float).reshape(-1, 3)
    stress = np.empty_like(E)
    tangent = np.empty((E.shape[0], 3, 3))
    latent = np.empty((E.shape[0], 3))
    tic = time.perf_counter()  # worker/model construction is excluded
    if _WORKER_MODEL == "direct" and _WORKER_IMPLEMENTATION == "optimized":
        stress, tangent, latent = _WORKER_LAW.stress_and_tangent_batch(E)
        return stress, tangent, latent, time.perf_counter() - tic, _WORKER_LAW.ecm_metadata
    for index, strain in enumerate(E):
        if E_start is None:
            stress[index], tangent[index], latent[index] = _WORKER_LAW.stress_and_tangent(
                strain, return_state=True
            )
        else:
            stress[index], tangent[index], latent[index] = _WORKER_LAW.stress_and_tangent(
                strain, q_init=q_start[index], E_start=E_start[index], return_state=True
            )
    return stress, tangent, latent, time.perf_counter() - tic, _WORKER_LAW.ecm_metadata


class ParallelMAWHPROMANN:
    """Macro-law adapter retaining continuation state per Gauss point."""

    def __init__(self, executor, workers: int):
        self.executor = executor
        self.workers = int(workers)
        self.calls = 0
        self.unique_queries = 0
        self.hprom_ann_seconds = 0.0
        self._E_previous = None
        self._q_previous = None
        self.worker_metadata = {}

    def __call__(self, E_flat, young=None, poisson=None):
        del young, poisson
        E = np.asarray(E_flat, dtype=float).reshape(-1, 3)
        if self._E_previous is None or self._E_previous.shape != E.shape:
            # All first assemblies are at E=0: solve this once then broadcast.
            E_unique, inverse = np.unique(E, axis=0, return_inverse=True)
            chunks = [part for part in np.array_split(E_unique, min(self.workers, len(E_unique)))
                      if part.size]
            result = list(self.executor.map(_worker_chunk, [(part, None, None) for part in chunks]))
            unique_stress = np.concatenate([item[0] for item in result], axis=0)
            unique_tangent = np.concatenate([item[1] for item in result], axis=0)
            unique_latent = np.concatenate([item[2] for item in result], axis=0)
            stress, tangent, latent = (unique_stress[inverse], unique_tangent[inverse],
                                       unique_latent[inverse])
            solved = int(E_unique.shape[0])
        else:
            indices = np.arange(E.shape[0])
            chunks = [part for part in np.array_split(indices, min(self.workers, E.shape[0])) if part.size]
            result = list(self.executor.map(
                _worker_chunk,
                [(E[part], self._E_previous[part], self._q_previous[part]) for part in chunks],
            ))
            stress = np.empty_like(E)
            tangent = np.empty((E.shape[0], 3, 3))
            latent = np.empty((E.shape[0], 3))
            for part, (s, c, q, _elapsed, _meta) in zip(chunks, result):
                stress[part], tangent[part], latent[part] = s, c, q
            solved = int(E.shape[0])

        for metadata in (item[4] for item in result):
            # Direct MAW-D-HPROM-ANN deliberately has no residual MDPA; its
            # worker identity is therefore the stress mesh path.
            mesh_key = metadata.get("residual_mdpa", metadata["stress_mdpa"])
            self.worker_metadata[mesh_key] = metadata
        self.calls += 1
        self.unique_queries += solved
        self.hprom_ann_seconds += float(sum(item[3] for item in result))
        self._E_previous, self._q_previous = E.copy(), latent
        return stress, tangent


def run(args) -> int:
    global _WORKER_MODEL, _WORKER_IMPLEMENTATION
    _WORKER_MODEL = "direct" if args.direct else "iterative"
    _WORKER_IMPLEMENTATION = args.implementation
    macro_base, rve_base = _canonical_meshes(args.macro_divisor)
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=_FORK)
    pids = sorted(set(executor.map(_worker_bootstrap, range(args.workers))))
    if len(pids) != args.workers:
        executor.shutdown(cancel_futures=True)
        raise RuntimeError(f"requested {args.workers} workers but started {len(pids)}")

    import fom_solver_rve as fom
    from macro_prepass import MacroCoupon

    law = ParallelMAWHPROMANN(executor, args.workers)
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
    t0 = time.perf_counter()  # clean online timing; offline training/setup excluded

    try:
        fom._neo_hookean_pk2_2d_vectorized = law
        for step in range(1, args.max_steps + 1):
            external = f_final * (step / args.n_steps)
            residual_initial, converged = None, False
            best = (np.inf, u.copy())
            for iteration in range(1, args.max_newton + 1):
                tic = time.perf_counter()
                if cached_assembly is None:
                    K, rhs = macro.assembler.Assemble(u)
                else:
                    K, rhs = cached_assembly
                    cached_assembly = None
                residual_vector = rhs + external
                residual = float(np.linalg.norm(residual_vector[free]))
                residual_initial = max(residual, 1.0e-30) if residual_initial is None else residual_initial
                relative = residual / residual_initial
                best = min(best, (residual, u.copy()), key=lambda item: item[0])
                print(f"  step {step:2d}/{args.n_steps} iter {iteration:2d}: |R|={residual:.5e}, "
                      f"rel={relative:.3e}, wall={time.perf_counter() - tic:.2f}s", flush=True)
                if residual < args.abs_tol or relative < args.rel_tol:
                    converged = True
                    break
                increment = spsolve(K[free, :][:, free].tocsc(), residual_vector[free])
                if not np.all(np.isfinite(increment)):
                    raise RuntimeError(f"non-finite macro update at step {step}, iteration {iteration}")
                u[free] += increment
            if not converged:
                u = best[1]
                raise RuntimeError(f"macro Newton did not converge at step {step}; "
                                   f"best relative residual={best[0] / residual_initial:.3e}")

            strain = macro.assembler._E_voigt.reshape(-1, 3).copy()
            coverage = _coverage(strain, blo, bhi)
            cloud.append(strain)
            records.append(dict(step=step, iterations=iteration, residual=residual,
                                relative_residual=relative, coverage=coverage))
            cached_assembly = (K, rhs)
            print(f"    converged: in-box {coverage['inside']}/{coverage['n']}; "
                  f"E11=[{coverage['minimum'][0]:+.4f},{coverage['maximum'][0]:+.4f}], "
                  f"E22=[{coverage['minimum'][1]:+.4f},{coverage['maximum'][1]:+.4f}], "
                  f"g12=[{coverage['minimum'][2]:+.4f},{coverage['maximum'][2]:+.4f}]", flush=True)

        wall = time.perf_counter() - t0
        E_final = macro.assembler._E_voigt.reshape(-1, 3).copy()
        S_final = macro.assembler._S_voigt.reshape(-1, 3).copy()
        u_nodes = np.stack((u[macro.eq_map[:, 0]], u[macro.eq_map[:, 1]]), axis=1)
        prefix = "dhprom_ann_fe2" if args.direct else "hprom_ann_fe2"
        default_tag = (f"maw10_direct_w{args.workers}_f{args.force / 1e3:g}kn"
                       if args.direct else f"maw10_w{args.workers}_f{args.force / 1e3:g}kn")
        if args.implementation == "optimized":
            default_tag += "_optimized"
        tag = args.tag or default_tag
        out = HERE / f"{prefix}_{tag}.npz"
        summary = dict(
            status="converged",
            model=("maw_dhprom_ann_direct" if args.direct else "maw_hprom_ann_iterative"),
            implementation=args.implementation,
            output=str(out), wall_seconds=wall, workers=args.workers, worker_pids=pids,
            macro_elements=macro.assembler.n_elems,
            macro_gauss_points=macro.assembler.n_elems * macro.assembler.n_gauss,
            rve_elements_full=int(np.load(ROOT / "03_data" / "data.npz")["n_elements"]),
            hprom_ann_latent_dimension=3,
            residual_ecm_elements=(0 if args.direct else 10), stress_ecm_elements=10,
            force_per_end=args.force, n_steps_requested=args.n_steps,
            n_steps_completed=len(records), macro_newton=records,
            material_calls=law.calls, unique_hprom_ann_queries=law.unique_queries,
            summed_hprom_ann_seconds=law.hprom_ann_seconds,
            worker_ecm_mdpa=list(law.worker_metadata.values()),
            macro_relative_tolerance=args.rel_tol, macro_absolute_tolerance=args.abs_tol,
            timing_scope="macro solve including constitutive tangents, IPC and lazy worker model loading; excludes parent setup and result export",
            source_sha256={name: _sha256(HERE / name) for name in
                           ("run_hprom_ann_fe2.py", "maw_hprom_ann_law.py", "maw_hprom_ann_fast.py",
                            "direct_hprom_ann_law.py", "direct_hprom_ann_fast.py", "reduced_stress_batch.py")},
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument("--macro-divisor", type=float, default=4.0)
    parser.add_argument("--force", type=float, default=1.0e5)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-newton", type=int, default=16)
    parser.add_argument("--rel-tol", type=float, default=1.0e-7)
    parser.add_argument("--abs-tol", type=float, default=1.0e-5)
    parser.add_argument("--tag", default="")
    parser.add_argument("--implementation", choices=("baseline", "optimized"), default="optimized",
                        help="baseline or local/batched implementation for either tier; equations/tolerances are identical")
    parser.add_argument("--direct", action="store_true",
                        help="run MAW-D-HPROM-ANN: direct closure and the 10-element stress MDPA only")
    parser.add_argument("--no-output", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.n_steps < 1 or not 1 <= args.max_steps <= args.n_steps:
        parser.error("require 1 <= --max-steps <= --n-steps")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
