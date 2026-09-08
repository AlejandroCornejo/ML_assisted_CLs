#!/usr/bin/env python3
"""Nested periodic FOM--FE2 reference for the coupon's one primary case.

The macro problem is the ASTM D638 coupon under equal-and-opposite axial
tractions.  The material call at every macro Gauss point is the *periodic*
RVE FOM from stage 00, not a PANN or a reduced model.  This is deliberately a
new stage-06 adapter: it reuses the already verified macro assembler and RVE
solver, without modifying either one's constitutive physics.

Run this file in a fresh Python process.  The worker pool is started before
the parent imports Kratos, then every worker owns one persistent RVE assembler.
That avoids both fork-after-Kratos hazards and rebuilding a 1546-element RVE
for every Gauss-point query.
"""
from __future__ import annotations

import os

# One microscopic RVE per process; do not let BLAS create another pool inside
# every one of the outer FE2 workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon_fe2_mpl")

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for _p in (str(ROOT), str(ROOT / "00_rve"), str(ROOT / "01_macro_prepass"),
           str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_kratos_candidates = (
    Path("/home/sares/Kratos_Eigen_Check/bin/Release"),
    Path("/home/kratos/Kratos_Eigen_Check/bin/Release"),
)
KRATOS_PATH = next((p for p in _kratos_candidates if p.is_dir()), _kratos_candidates[0])
if str(KRATOS_PATH) not in sys.path:
    sys.path.append(str(KRATOS_PATH))

import config as cfg  # noqa: E402

_FORK = multiprocessing.get_context("fork")
_WORKER_RVE = None
_WORKER_KEY = None


def _worker_bootstrap(_):
    """Force the pool to fork before this process imports Kratos."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # A brief overlap makes ProcessPoolExecutor instantiate *every* requested
    # worker rather than letting a few fast bootstrap jobs be reused.  That
    # detail matters: a worker forked only later would inherit a parent that
    # has already imported Kratos.
    time.sleep(0.25)
    return os.getpid()


def _rve_chunk(job):
    """Evaluate a chunk in a persistent child process.

    ``q_start``/``E_start`` are the last converged state of the *same macro
    Gauss point*.  They alter only the nonlinear-solver initial guess and its
    ramp, never the periodic boundary-value problem being homogenized.
    """
    global _WORKER_RVE, _WORKER_KEY
    E, E_start, q_start, mesh_base, cell_area = job
    key = (str(mesh_base), float(cell_area))
    if _WORKER_RVE is None or _WORKER_KEY != key:
        from periodic_fom import PeriodicRVE
        _WORKER_RVE = PeriodicRVE(mesh_base, cell_area=cell_area)
        _WORKER_KEY = key

    E = np.asarray(E, dtype=float).reshape(-1, 3)
    S = np.empty_like(E)
    CC = np.empty((E.shape[0], 3, 3), dtype=float)
    Q = np.empty((E.shape[0], _WORKER_RVE.T.shape[1]), dtype=float)
    t0 = time.perf_counter()
    for i, e in enumerate(E):
        if E_start is None:
            S[i], CC[i], Q[i] = _WORKER_RVE.stress_and_tangent_consistent(
                e, return_state=True)
        else:
            S[i], CC[i], Q[i] = _WORKER_RVE.stress_and_tangent_consistent(
                e, u_ind_init=q_start[i], E_start=E_start[i], return_state=True)
    return S, CC, Q, time.perf_counter() - t0


class ParallelPeriodicFOM:
    """Vectorized-assembler material-law adapter backed by persistent RVEs."""

    def __init__(self, executor, workers, mesh_base, cell_area):
        self.executor = executor
        self.workers = int(workers)
        self.mesh_base = str(mesh_base)
        self.cell_area = float(cell_area)
        self.calls = 0
        self.rve_seconds = 0.0
        self.unique_queries = 0
        self._E_previous = None
        self._q_previous = None

    def __call__(self, E_flat, young=None, poisson=None):
        del young, poisson
        E = np.asarray(E_flat, dtype=float).reshape(-1, 3)
        # First assembly: thousands of exactly-zero Gauss points require one
        # RVE solve, then its identical state can be broadcast.  Thereafter
        # each macro point carries its own prior q and is continued exactly.
        if self._E_previous is None or self._E_previous.shape != E.shape:
            E_unique, inverse = np.unique(E, axis=0, return_inverse=True)
            n_jobs = min(self.workers, E_unique.shape[0])
            chunks = [c for c in np.array_split(E_unique, n_jobs) if c.size]
            result = list(self.executor.map(
                _rve_chunk,
                [(c, None, None, self.mesh_base, self.cell_area) for c in chunks],
            ))
            Su = np.concatenate([r[0] for r in result], axis=0)
            Cu = np.concatenate([r[1] for r in result], axis=0)
            Qu = np.concatenate([r[2] for r in result], axis=0)
            S, CC, Q = Su[inverse], Cu[inverse], Qu[inverse]
            n_solved = int(E_unique.shape[0])
        else:
            indices = np.arange(E.shape[0])
            chunks = [c for c in np.array_split(indices, min(self.workers, E.shape[0]))
                      if c.size]
            result = list(self.executor.map(
                _rve_chunk,
                [(E[c], self._E_previous[c], self._q_previous[c],
                  self.mesh_base, self.cell_area) for c in chunks],
            ))
            S, CC = np.empty_like(E), np.empty((E.shape[0], 3, 3), dtype=float)
            Q = np.empty((E.shape[0], self._q_previous.shape[1]), dtype=float)
            for c, (s, cc, q, _elapsed) in zip(chunks, result):
                S[c], CC[c], Q[c] = s, cc, q
            n_solved = int(E.shape[0])
        self.calls += 1
        self.unique_queries += n_solved
        self.rve_seconds += float(sum(r[3] for r in result))
        self._E_previous = E.copy()
        self._q_previous = Q
        return S, CC


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _canonical_meshes(divisor):
    """Return the immutable stage-01/03 meshes used to design this study.

    Re-meshing in stage 06 would make a timing test depend on the installed
    Gmsh version and, worse, could silently make the nested FOM use a different
    RVE than the one that generated ``data.npz``.  The canonical W/3, W/4,
    W/6, W/8 and W/12 macro meshes and the deployed periodic RVE mesh were
    retained precisely to avoid that ambiguity.
    """
    suffix = f"{float(divisor):g}"
    macro_base = ROOT / "01_macro_prepass" / f"coupon_conv_div{suffix}"
    rve_base = ROOT / "03_data" / "rve_mesh"
    for base in (macro_base, rve_base):
        if not Path(str(base) + ".mdpa").is_file():
            raise FileNotFoundError(f"required canonical mesh is absent: {base}.mdpa")
    return macro_base, rve_base


def _edge_unit_force(coords, nodes, direction):
    """Unit-total traction on a straight quadratic T6 end face."""
    nodes = np.asarray(nodes, dtype=np.int64)
    nodes = nodes[np.argsort(coords[nodes, 1])]
    if nodes.size < 3 or (nodes.size - 1) % 2:
        raise RuntimeError("end face does not carry consecutive quadratic nodes")
    f = np.zeros((coords.shape[0], 2), dtype=float)
    d = np.asarray(direction, dtype=float)
    n_sub = (nodes.size - 1) // 2
    for k in range(n_sub):
        lo, mid, hi = nodes[2 * k:2 * k + 3]
        f[lo] += d / (6.0 * n_sub)
        f[mid] += 4.0 * d / (6.0 * n_sub)
        f[hi] += d / (6.0 * n_sub)
    return f


def _force_vector(macro, total_force):
    """Equal-and-opposite axial forces; ``total_force`` is per end face."""
    f_node = (_edge_unit_force(macro.xy, macro.right, (1.0, 0.0))
              + _edge_unit_force(macro.xy, macro.left, (-1.0, 0.0)))
    f = np.zeros(macro.n_dof, dtype=float)
    np.add.at(f, macro.eq_map[:, 0], f_node[:, 0])
    np.add.at(f, macro.eq_map[:, 1], f_node[:, 1])
    return float(total_force) * f


def _minimal_rbm_constraints(macro):
    """Three point constraints remove rigid modes without clamping a face."""
    left_mid = int(macro.left[np.argmin(np.abs(macro.xy[macro.left, 1]))])
    right_mid = int(macro.right[np.argmin(np.abs(macro.xy[macro.right, 1]))])
    return np.array((macro.eq_map[left_mid, 0], macro.eq_map[left_mid, 1],
                     macro.eq_map[right_mid, 1]), dtype=np.int64)


def _coverage(E, blo, bhi):
    E = np.asarray(E, dtype=float).reshape(-1, 3)
    inside = np.all((E >= blo[None, :] - 1e-12)
                    & (E <= bhi[None, :] + 1e-12), axis=1)
    return dict(
        n=int(E.shape[0]), inside=int(np.sum(inside)), outside=int(np.sum(~inside)),
        minimum=np.min(E, axis=0).tolist(), maximum=np.max(E, axis=0).tolist(),
    )


def _save_checkpoint(path, *, args, u, cloud, records, law, macro_base, rve_base):
    """Write a restartable state after a *converged* macro load step.

    A FOM--FE2 load path costs hours, so the useful unit of persistence is a
    converged load step, never an unconverged Newton iterate.  The checkpoint
    has the macro displacement and the complete strain history needed for the
    final audit; microscopic states deliberately are not serialized because
    the periodic solver is path independent and they can safely restart cold.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    payload = dict(
        u=np.asarray(u), E_path=np.concatenate(cloud, axis=0),
        n_steps_completed=len(records), force_per_end=args.force,
        n_steps_requested=args.n_steps, macro_divisor=args.macro_divisor,
        material_calls=law.calls, unique_rve_queries=law.unique_queries,
        summed_rve_seconds=law.rve_seconds,
        macro_mesh_sha256=_sha256(str(macro_base) + ".mdpa"),
        rve_mesh_sha256=_sha256(str(rve_base) + ".mdpa"),
    )
    # q is not physical history (the energy has none), but persisting it
    # prevents a restart from unnecessarily re-ramping every RVE from zero.
    # It is intentionally optional so checkpoints made before this feature
    # remain restartable.
    if law._E_previous is not None and law._q_previous is not None:
        payload["rve_E_previous"] = law._E_previous
        payload["rve_q_previous"] = law._q_previous
    np.savez_compressed(tmp, **payload)
    # np.savez appends .npz when needed.  Use replace so an interruption
    # cannot corrupt the last good point of the load path.
    os.replace(str(tmp) + ".npz", path)
    path.with_suffix(".json").write_text(
        json.dumps(dict(status="checkpoint", records=records), indent=2),
        encoding="utf-8",
    )


def _load_checkpoint(path, args, macro_base, rve_base):
    """Validate and return a compatible converged-step checkpoint."""
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"resume requires {path} and {meta_path}")
    with np.load(path) as z:
        checks = dict(
            force_per_end=float(z["force_per_end"]),
            n_steps_requested=int(z["n_steps_requested"]),
            macro_divisor=float(z["macro_divisor"]),
            macro_mesh_sha256=str(z["macro_mesh_sha256"]),
            rve_mesh_sha256=str(z["rve_mesh_sha256"]),
        )
        expected = dict(
            force_per_end=float(args.force), n_steps_requested=int(args.n_steps),
            macro_divisor=float(args.macro_divisor),
            macro_mesh_sha256=_sha256(str(macro_base) + ".mdpa"),
            rve_mesh_sha256=_sha256(str(rve_base) + ".mdpa"),
        )
        if checks != expected:
            raise RuntimeError("checkpoint does not belong to this FOM--FE2 case: "
                               f"found {checks}, expected {expected}")
        state = dict(
            u=np.asarray(z["u"], dtype=float),
            E_path=np.asarray(z["E_path"], dtype=float),
            n_steps_completed=int(z["n_steps_completed"]),
            material_calls=int(z["material_calls"]),
            unique_rve_queries=int(z["unique_rve_queries"]),
            summed_rve_seconds=float(z["summed_rve_seconds"]),
            rve_E_previous=(np.asarray(z["rve_E_previous"], dtype=float)
                            if "rve_E_previous" in z.files else None),
            rve_q_previous=(np.asarray(z["rve_q_previous"], dtype=float)
                            if "rve_q_previous" in z.files else None),
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    records = meta["records"]
    if len(records) != state["n_steps_completed"]:
        raise RuntimeError("checkpoint metadata and array state disagree")
    if not 0 <= state["n_steps_completed"] <= args.n_steps:
        raise RuntimeError("invalid checkpoint step count")
    return state, records


def run(args):
    # Mesh generation has no Kratos dependency.  Start and explicitly warm the
    # children before MacroCoupon imports Kratos in this parent process.
    macro_base, rve_base = _canonical_meshes(args.macro_divisor)
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=_FORK)
    pids = sorted(set(executor.map(_worker_bootstrap, range(args.workers))))
    if len(pids) != args.workers:
        executor.shutdown(cancel_futures=True)
        raise RuntimeError(f"requested {args.workers} workers but started {len(pids)}")

    import fom_solver_rve as fom
    from macro_prepass import MacroCoupon

    law = ParallelPeriodicFOM(executor, args.workers, rve_base, cfg.CELL_AREA)
    original_law = fom._neo_hookean_pk2_2d_vectorized
    m = MacroCoupon(macro_base, verbose=True)
    f_final = _force_vector(m, args.force)
    fixed = _minimal_rbm_constraints(m)
    free_mask = np.ones(m.n_dof, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    tag = args.tag or f"w{args.workers}_w{args.macro_divisor:g}_s{args.max_steps}"
    checkpoint = HERE / f"fom_fe2_{tag}.checkpoint.npz"
    u = np.zeros(m.n_dof, dtype=float)
    grid = np.load(ROOT / "02_sampling" / "train_grid.npz")
    blo, bhi = np.asarray(grid["blo"]), np.asarray(grid["bhi"])
    records, cloud = [], []
    cached_assembly = None
    first_step = 1
    if args.resume:
        state, records = _load_checkpoint(checkpoint, args, macro_base, rve_base)
        if state["n_steps_completed"] > args.max_steps:
            raise RuntimeError("checkpoint is beyond --max-steps")
        u = state["u"]
        cloud = [state["E_path"]]
        first_step = state["n_steps_completed"] + 1
        law.calls = state["material_calls"]
        law.unique_queries = state["unique_rve_queries"]
        law.rve_seconds = state["summed_rve_seconds"]
        if state["rve_E_previous"] is not None:
            law._E_previous = state["rve_E_previous"]
            law._q_previous = state["rve_q_previous"]
        print(f"[resume] continuing after converged step {first_step - 1}", flush=True)
    t0 = time.perf_counter()

    try:
        fom._neo_hookean_pk2_2d_vectorized = law
        for step in range(first_step, min(args.n_steps, args.max_steps) + 1):
            f_ext = f_final * (step / args.n_steps)
            res0 = None
            best = (np.inf, u.copy())
            converged = False
            for it in range(1, args.max_newton + 1):
                tic = time.perf_counter()
                if cached_assembly is None:
                    K, rhs = m.assembler.Assemble(u)
                else:
                    # The preceding step ended at this unchanged u.  Its
                    # material response and tangent are therefore exactly
                    # this first iterate; avoid repeating 2070 RVE solves.
                    K, rhs = cached_assembly
                    cached_assembly = None
                residual = rhs + f_ext
                res = float(np.linalg.norm(residual[free]))
                res0 = max(res, 1.0e-30) if res0 is None else res0
                best = min(best, (res, u.copy()), key=lambda x: x[0])
                rel = res / res0
                print(f"  step {step:2d}/{args.n_steps} iter {it:2d}: "
                      f"|R|={res:.5e}, rel={rel:.3e}, "
                      f"wall={time.perf_counter() - tic:.1f}s", flush=True)
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
                raise RuntimeError(f"macro Newton did not converge at step {step}; best rel "
                                   f"{best[0] / res0:.3e}")

            E = m.assembler._E_voigt.reshape(-1, 3).copy()
            S = m.assembler._S_voigt.reshape(-1, 3).copy()
            cov = _coverage(E, blo, bhi)
            cloud.append(E)
            records.append(dict(step=step, iterations=it, residual=res,
                                relative_residual=rel, coverage=cov))
            cached_assembly = (K, rhs)
            if args.checkpoint:
                _save_checkpoint(checkpoint, args=args, u=u, cloud=cloud,
                                 records=records, law=law, macro_base=macro_base,
                                 rve_base=rve_base)
            print(f"    converged: in-box {cov['inside']}/{cov['n']}; "
                  f"E11=[{cov['minimum'][0]:+.4f},{cov['maximum'][0]:+.4f}], "
                  f"E22=[{cov['minimum'][1]:+.4f},{cov['maximum'][1]:+.4f}], "
                  f"g12=[{cov['minimum'][2]:+.4f},{cov['maximum'][2]:+.4f}]", flush=True)

        # The assembler already holds the final converged material response.
        E_final = m.assembler._E_voigt.reshape(-1, 3).copy()
        S_final = m.assembler._S_voigt.reshape(-1, 3).copy()
        u_nodes = np.stack((u[m.eq_map[:, 0]], u[m.eq_map[:, 1]]), axis=1)
        wall = time.perf_counter() - t0
        out = HERE / f"fom_fe2_{tag}.npz"
        np.savez_compressed(out, coords=m.xy, connectivity=m.assembler.connectivity,
                            u_nodal=u_nodes, E_final=E_final, S_final=S_final,
                            E_path=np.concatenate(cloud, axis=0),
                            force_per_end=args.force, n_steps_requested=args.n_steps,
                            n_steps_completed=len(records), workers=args.workers,
                            macro_divisor=args.macro_divisor)
        summary = dict(
            status="converged", tag=tag, output=str(out), wall_seconds=wall,
            workers=args.workers, worker_pids=pids, macro_elements=m.assembler.n_elems,
            macro_gauss_points=m.assembler.n_elems * m.assembler.n_gauss,
            rve_elements=int(np.load(ROOT / "03_data" / "data.npz")["n_elements"]),
            force_per_end=args.force,
            n_steps_requested=args.n_steps, n_steps_completed=len(records),
            macro_newton=records, material_calls=law.calls,
            unique_rve_queries=law.unique_queries, summed_rve_seconds=law.rve_seconds,
            macro_mesh_sha256=_sha256(str(macro_base) + ".mdpa"),
            rve_mesh_sha256=_sha256(str(rve_base) + ".mdpa"),
        )
        (HERE / f"fom_fe2_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    finally:
        fom._neo_hookean_pk2_2d_vectorized = original_law
        executor.shutdown(wait=True, cancel_futures=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--macro-divisor", type=float, default=4.0,
                   help="macro h = gauge width / divisor; W/4 is the planned first reference mesh")
    p.add_argument("--force", type=float, default=1.0e5,
                   help="total force in N on EACH end face; initial calibrated estimate")
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=20,
                   help="run only the first N load steps; use 1 for the worker benchmark")
    p.add_argument("--max-newton", type=int, default=12)
    p.add_argument("--rel-tol", type=float, default=1.0e-7)
    p.add_argument("--abs-tol", type=float, default=1.0e-5)
    p.add_argument("--tag", default="")
    p.add_argument("--resume", action="store_true",
                   help="continue from this tag's converged-step checkpoint")
    p.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True,
                   help="write restart data after each converged step; disable for clean timing")
    a = p.parse_args()
    if a.workers < 1:
        p.error("--workers must be positive")
    if a.max_steps < 1 or a.n_steps < 1 or a.max_steps > a.n_steps:
        p.error("require 1 <= --max-steps <= --n-steps")
    if a.resume and not a.checkpoint:
        p.error("--resume requires --checkpoint")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
