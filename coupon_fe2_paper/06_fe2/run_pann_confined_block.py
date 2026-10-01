#!/usr/bin/env python3
"""Confined compression of a block (oedometer) or of a bonded pad, with a direct PANN law.

A plane-strain block is compressed by a uniform force on one face while rigid,
frictionless walls fix the normal displacement of the opposite face and of the
two lateral faces.  Away from its corners the block therefore follows the
rank-one path F = I + (lambda - 1) n (x) n, along which a polyconvex energy has a
monotone load.  The macro assembly, the Newton iteration and the PANN laws are
those of the coupon runs in run_pann_fe2.py; only the mesh and the boundary
conditions change.  No microscopic model is solved, so the test concerns the
constitutive surrogates alone.

``--problem pad`` instead bonds a wide block to two rigid plates: the bottom plate is
fixed, the top plate moves down under displacement control, and the lateral faces are
free.  Bonding confines the interior, as in an anti-vibration mount.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

import run_pann_fe2 as coupon


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def block_mesh(length, height, nx, ny):
    """Structured quadratic triangles (Kratos Triangle2D6), alternating diagonals."""
    xs = np.linspace(0.0, length, 2 * nx + 1)
    ys = np.linspace(-0.5 * height, 0.5 * height, 2 * ny + 1)
    index = np.arange(xs.size * ys.size).reshape(ys.size, xs.size)   # index[row(y), column(x)]
    coords = np.array([(x, y) for y in ys for x in xs])
    node = lambda i, j: int(index[j, i])
    tris = []
    for j in range(ny):
        for i in range(nx):
            a, b = 2 * i, 2 * j
            c00, c20, c22, c02 = node(a, b), node(a + 2, b), node(a + 2, b + 2), node(a, b + 2)
            if (i + j) % 2 == 0:
                tris.append((c00, c20, c22, node(a + 1, b), node(a + 2, b + 1), node(a + 1, b + 1)))
                tris.append((c00, c22, c02, node(a + 1, b + 1), node(a + 1, b + 2), node(a, b + 1)))
            else:
                tris.append((c00, c20, c02, node(a + 1, b), node(a + 1, b + 1), node(a, b + 1)))
                tris.append((c20, c22, c02, node(a + 2, b + 1), node(a + 1, b + 2), node(a + 1, b + 1)))
    return coords, np.asarray(tris, dtype=np.int64)


def face_unit_force(coords, nodes, along, direction):
    """Consistent nodal forces of a unit total uniform traction on a straight T6 face."""
    nodes = nodes[np.argsort(coords[nodes, along])]
    s = coords[nodes, along]
    if (nodes.size - 1) % 2:
        raise ValueError("a quadratic face needs an odd number of nodes")
    force = np.zeros((coords.shape[0], 2))
    for k in range(0, nodes.size - 2, 2):
        weight = (s[k + 2] - s[k]) / (s[-1] - s[0])
        for n, w in zip(nodes[k:k + 3], (1 / 6, 4 / 6, 1 / 6)):
            force[n] += w * weight * np.asarray(direction, dtype=float)
    return force


class MacroBlock:
    """The coupon's Kratos model part and vectorized assembler on a block mesh."""

    def __init__(self, mesh_base):
        import KratosMultiphysics as KM
        import fom_solver_rve as fom
        from fom_nested_consistent_law_claude import make_parameters

        model = KM.Model()
        sim = fom.RVEHomogenizationDatasetGenerator(model, make_parameters(mesh_base=str(mesh_base)))
        sim.Initialize()
        mp = sim._GetSolver().GetComputingModelPart()
        self._sim, self._mp = sim, mp
        n_dof, eq_map, _ = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)
        self.n_dof = int(n_dof)
        self.assembler = fom.VectorizedAssembler(mp, n_dof, eq_map)
        self.xy = np.array([[n.X0, n.Y0] for n in mp.Nodes], dtype=float)
        self.eq_map = np.asarray(eq_map, dtype=np.int64)
        thickness = np.unique(np.asarray(self.assembler.thickness, dtype=float))
        if thickness.size != 1:
            raise RuntimeError(f"non-uniform thickness {thickness}")
        self.thickness = float(thickness[0])


def run(args) -> int:
    coupon._configure_torch(args.threads)
    checkpoint = Path(args.checkpoint).resolve()
    law = coupon.CouponFreePANNLaw(checkpoint) if args.tier == "free" else coupon.FlexiblePANNLaw(checkpoint)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    length, height = args.size, args.size
    coords, tris = block_mesh(length, height, args.cells, args.cells)
    tol = 1.0e-9 * length
    left = np.flatnonzero(np.abs(coords[:, 0]) < tol)
    right = np.flatnonzero(np.abs(coords[:, 0] - length) < tol)
    bottom = np.flatnonzero(np.abs(coords[:, 1] + 0.5 * height) < tol)
    top = np.flatnonzero(np.abs(coords[:, 1] - 0.5 * height) < tol)
    mesh_base = output_dir / f"block_{args.cells}x{args.cells}"
    from macro_prepass import write_coupon_mdpa
    write_coupon_mdpa(str(mesh_base) + ".mdpa", coords, tris, left)

    import fom_solver_rve as fom
    macro = MacroBlock(mesh_base)
    if not np.allclose(macro.xy, coords):
        raise RuntimeError("Kratos node order differs from the generated mesh")
    # Loaded face and walls: x-compression loads the right face; y-compression the top face.
    if args.direction == "x":
        loaded, direction = right, (-1.0, 0.0)
        fixed = np.concatenate((macro.eq_map[left, 0], macro.eq_map[bottom, 1], macro.eq_map[top, 1]))
        face_length = height
    else:
        loaded, direction = top, (0.0, -1.0)
        fixed = np.concatenate((macro.eq_map[bottom, 1], macro.eq_map[left, 0], macro.eq_map[right, 0]))
        face_length = length
    f_node = face_unit_force(coords, loaded, 1 if args.direction == "x" else 0, direction)
    f_unit = np.zeros(macro.n_dof)
    np.add.at(f_unit, macro.eq_map[:, 0], f_node[:, 0])
    np.add.at(f_unit, macro.eq_map[:, 1], f_node[:, 1])
    total_force = args.pressure * face_length * macro.thickness
    free = np.setdiff1d(np.arange(macro.n_dof), np.unique(fixed))

    u = np.zeros(macro.n_dof)
    records, failure, iterates, saved_u = [], None, [], []
    cached_assembly = None
    original_law = fom._neo_hookean_pk2_2d_vectorized
    t0 = time.perf_counter()
    try:
        fom._neo_hookean_pk2_2d_vectorized = law
        for step in range(1, args.n_steps + 1):
            f_ext = total_force * f_unit * (step / args.n_steps)
            res0, converged, iterates = None, False, []
            for it in range(1, args.max_newton + 1):
                if cached_assembly is None:
                    try:
                        K, rhs = macro.assembler.Assemble(u)
                    except (ValueError, FloatingPointError, RuntimeError) as error:
                        failure = f"material evaluation failed at step {step}, iteration {it}: {error}"
                        break
                else:
                    K, rhs = cached_assembly
                    cached_assembly = None
                residual = rhs + f_ext
                res = float(np.linalg.norm(residual[free]))
                res0 = max(res, 1.0e-30) if res0 is None else res0
                rel = res / res0
                E_it = macro.assembler._E_voigt.reshape(-1, 3)
                det_c = (1.0 + 2.0 * E_it[:, 0]) * (1.0 + 2.0 * E_it[:, 1]) - E_it[:, 2] ** 2
                iterates.append(dict(iteration=it, residual=res, relative_residual=rel,
                                     min_det_C=float(np.min(det_c)) if np.all(np.isfinite(det_c)) else None))
                if not np.isfinite(res):
                    failure = f"non-finite residual at step {step}, iteration {it}"
                    break
                if res < args.abs_tol or rel < args.rel_tol:
                    converged = True
                    break
                du = spsolve(K[free, :][:, free].tocsc(), residual[free])
                if not np.all(np.isfinite(du)):
                    failure = f"non-finite macro update at step {step}, iteration {it}"
                    break
                u[free] += du
            if failure is None and not converged:
                failure = f"macro Newton did not converge at step {step} within {args.max_newton} iterations"
            if failure is not None:
                break
            E = macro.assembler._E_voigt.reshape(-1, 3)
            det_c = (1.0 + 2.0 * E[:, 0]) * (1.0 + 2.0 * E[:, 1]) - E[:, 2] ** 2
            dof = macro.eq_map[loaded, 0 if args.direction == "x" else 1]
            stretch = 1.0 + float(np.mean(u[dof])) / (length if args.direction == "x" else height)
            records.append(dict(step=step, pressure_Pa=args.pressure * step / args.n_steps,
                                iterations=it, mean_stretch=stretch, min_J=float(np.sqrt(np.min(det_c))),
                                iterates=iterates))
            cached_assembly = (K, rhs)
            if args.save_fields:
                saved_u.append(u.copy())
            print(f"  step {step:2d}/{args.n_steps}: p={args.pressure * step / args.n_steps / 1e6:7.1f} MPa, "
                  f"iterations={it}, stretch={stretch:.4f}, min J={records[-1]['min_J']:.4f}", flush=True)
    finally:
        fom._neo_hookean_pk2_2d_vectorized = original_law
    summary = dict(status="not_converged" if failure else "converged", error=failure, tier=args.tier,
                   tag=args.tag, direction=args.direction, pressure_Pa=args.pressure, n_steps=args.n_steps,
                   max_newton=args.max_newton, n_steps_completed=len(records), size_m=args.size,
                   cells=args.cells, thickness_m=macro.thickness, checkpoint=law.metadata,
                   wall_seconds=time.perf_counter() - t0, steps=records,
                   failed_step=dict(iterates=iterates) if failure else None)
    (output_dir / f"{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_fields:
        np.savez_compressed(output_dir / f"{args.tag}.npz", coords=coords, triangles=tris,
                            eq_map=macro.eq_map, u=np.asarray(saved_u))
    print(json.dumps(dict(status=summary["status"], error=failure, completed=len(records))), flush=True)
    return 0 if failure is None else 1


def run_pad(args) -> int:
    coupon._configure_torch(args.threads)
    checkpoint = Path(args.checkpoint).resolve()
    law = coupon.CouponFreePANNLaw(checkpoint) if args.tier == "free" else coupon.FlexiblePANNLaw(checkpoint)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = args.size, args.size * args.aspect
    coords, tris = block_mesh(width, height, args.cells * args.aspect, args.cells)
    tol = 1.0e-9 * width
    bottom = np.flatnonzero(np.abs(coords[:, 1] + 0.5 * height) < tol)
    top = np.flatnonzero(np.abs(coords[:, 1] - 0.5 * height) < tol)
    mesh_base = output_dir / f"pad_{args.cells * args.aspect}x{args.cells}"
    from macro_prepass import write_coupon_mdpa
    write_coupon_mdpa(str(mesh_base) + ".mdpa", coords, tris, bottom)
    import fom_solver_rve as fom
    macro = MacroBlock(mesh_base)
    if not np.allclose(macro.xy, coords):
        raise RuntimeError("Kratos node order differs from the generated mesh")
    zero = np.concatenate((macro.eq_map[bottom, 0], macro.eq_map[bottom, 1], macro.eq_map[top, 0]))
    plate = macro.eq_map[top, 1]
    free = np.setdiff1d(np.arange(macro.n_dof), np.concatenate((zero, plate)))
    u = np.zeros(macro.n_dof)
    records, failure, iterates, fields = [], None, [], dict(u=[], J=[])
    original_law = fom._neo_hookean_pk2_2d_vectorized
    t0 = time.perf_counter()
    try:
        fom._neo_hookean_pk2_2d_vectorized = law
        for step in range(1, args.n_steps + 1):
            u[plate] = -args.compression * height * step / args.n_steps
            res0, converged, iterates = None, False, []
            for it in range(1, args.max_newton + 1):
                try:
                    K, rhs = macro.assembler.Assemble(u)
                except (ValueError, FloatingPointError, RuntimeError) as error:
                    failure = f"material evaluation failed at step {step}, iteration {it}: {error}"
                    break
                res = float(np.linalg.norm(rhs[free]))
                res0 = max(res, 1.0e-30) if res0 is None else res0
                rel = res / res0
                iterates.append(dict(iteration=it, residual=res, relative_residual=rel))
                if not np.isfinite(res):
                    failure = f"non-finite residual at step {step}, iteration {it}"
                    break
                if it > 1 and (res < args.abs_tol or rel < args.rel_tol):
                    converged = True
                    break
                du = spsolve(K[free, :][:, free].tocsc(), rhs[free])
                if not np.all(np.isfinite(du)):
                    failure = f"non-finite macro update at step {step}, iteration {it}"
                    break
                u[free] += du
            if failure is None and not converged:
                failure = f"macro Newton did not converge at step {step} within {args.max_newton} iterations"
            if failure is not None:
                break
            E = macro.assembler._E_voigt.reshape(-1, 3)
            J = np.sqrt((1.0 + 2.0 * E[:, 0]) * (1.0 + 2.0 * E[:, 1]) - E[:, 2] ** 2)
            force = float(-rhs[plate].sum())
            records.append(dict(step=step, compression=args.compression * step / args.n_steps,
                                plate_force_N=force, nominal_pressure_Pa=abs(force) / (width * macro.thickness),
                                iterations=it, min_J=float(J.min())))
            fields["u"].append(u.copy()); fields["J"].append(J.reshape(-1, macro.assembler.n_gauss).mean(axis=1))
            print(f"  step {step:2d}/{args.n_steps}: compression={records[-1]['compression']:.3f}, "
                  f"p={records[-1]['nominal_pressure_Pa'] / 1e6:8.2f} MPa, iterations={it}, "
                  f"min J={records[-1]['min_J']:.4f}", flush=True)
    finally:
        fom._neo_hookean_pk2_2d_vectorized = original_law
    summary = dict(status="not_converged" if failure else "converged", error=failure, tier=args.tier,
                   tag=args.tag, problem="pad", compression=args.compression, n_steps=args.n_steps,
                   max_newton=args.max_newton, n_steps_completed=len(records), size_m=args.size,
                   aspect=args.aspect, cells=args.cells, thickness_m=macro.thickness,
                   checkpoint=law.metadata, wall_seconds=time.perf_counter() - t0, steps=records,
                   failed_step=dict(iterates=iterates) if failure else None)
    (output_dir / f"{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / f"{args.tag}.npz", coords=coords, triangles=tris, eq_map=macro.eq_map,
                        u=np.asarray(fields["u"]), J_element=np.asarray(fields["J"]))
    print(json.dumps(dict(status=summary["status"], error=failure, completed=len(records))), flush=True)
    return 0 if failure is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("confined", "pad"), default="confined")
    parser.add_argument("--tier", choices=("flexible", "free"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--direction", choices=("x", "y"), default="x")
    parser.add_argument("--pressure", type=float, default=0.0, help="Final nominal pressure [Pa] (confined).")
    parser.add_argument("--compression", type=float, default=0.3,
                        help="Final plate displacement over pad height (pad).")
    parser.add_argument("--aspect", type=int, default=4, help="Pad width over height (pad).")
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--max-newton", type=int, default=50)
    parser.add_argument("--rel-tol", type=float, default=1.0e-7)
    parser.add_argument("--abs-tol", type=float, default=1.0e-5)
    parser.add_argument("--size", type=float, default=0.01, help="Block side [m].")
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-fields", action="store_true",
                        help="Also store the displacement of every converged step (confined).")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    return run_pad(args) if args.problem == "pad" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
