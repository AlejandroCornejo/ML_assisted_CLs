#!/usr/bin/env python3
"""Generate the unit cell (or an n x n block of cells) with a rotated
elliptical hole, written in the same .mdpa format the validated solver
already consumes.

Deliberately a drop-in replacement for core/rve_geometry.mdpa: same
Triangle2D6 elements, same `material` / `dirichlet` sub-model-part structure,
`dirichlet` carrying the OUTER boundary only so the hole stays traction-free
as a pore must. Nothing in the solver, the homogenization or the analytic
tangent has to change.

The n x n block support exists for one specific measurement: affine Dirichlet
BCs over-stiffen through a boundary layer (C_Neumann <= C_periodic <=
C_affine), so the effective response of a single cell is not the periodic
answer. Comparing C0 across 1x1, 3x3 and 5x5 quantifies that with a number
instead of leaving it as an open objection. Those are linear solves, so the
study is cheap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import config as cfg  # noqa: E402

GMSH_TRI6 = 9  # gmsh element type id for the 6-node second-order triangle


def build_mesh(n_cells=1, porosity=None, aspect=None, angle_deg=None,
               size_far=None, size_hole=None, verbose=False, periodic=False):
    """(coords (N,2), tris (M,6) 0-based, outer_nodes (K,) 0-based, geom dict).

    n_cells: cells per side. The block spans n_cells*CELL_SIDE and carries one
    hole per cell, so porosity is preserved as n_cells grows and the ONLY
    thing that changes is how far the affine boundary layer reaches into the
    interior -- which is exactly the quantity being measured.
    """
    import gmsh

    porosity = cfg.POROSITY if porosity is None else porosity
    aspect = cfg.ELLIPSE_ASPECT if aspect is None else aspect
    angle_deg = cfg.ELLIPSE_ANGLE_DEG if angle_deg is None else angle_deg
    # Deployed mesh sizes live in config, not duplicated in signatures here
    # and in every caller's argparse default -- that duplication is precisely
    # what let the previous codebase mix two problems into one results table.
    size_far = cfg.MESH_SIZE_FAR if size_far is None else size_far
    size_hole = cfg.MESH_SIZE_HOLE if size_hole is None else size_hole

    a, b = cfg.ellipse_semi_axes(porosity, aspect, cfg.CELL_AREA)
    hw, hh = cfg.ellipse_bounding_half_extents(porosity, aspect, angle_deg, cfg.CELL_AREA)
    if max(hw, hh) >= cfg.CELL_SIDE / 2.0:
        raise ValueError(f"ellipse bounding box {hw:.4f}x{hh:.4f} breaches the "
                         f"cell half-side {cfg.CELL_SIDE / 2.0}")

    cell = cfg.CELL_SIDE
    block = n_cells * cell
    half = block / 2.0
    theta = np.radians(angle_deg)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("rve")

        rect = gmsh.model.occ.addRectangle(-half, -half, 0.0, block, block)
        holes = []
        centres = []
        for i in range(n_cells):
            for j in range(n_cells):
                xc = -half + (i + 0.5) * cell
                yc = -half + (j + 0.5) * cell
                centres.append((xc, yc))
                d = gmsh.model.occ.addDisk(xc, yc, 0.0, a, b)
                # OCC addDisk builds the ellipse axis-aligned (a along x);
                # rotate about the hole's own centre so the rotation does not
                # translate it.
                gmsh.model.occ.rotate([(2, d)], xc, yc, 0.0, 0.0, 0.0, 1.0, theta)
                holes.append((2, d))

        gmsh.model.occ.cut([(2, rect)], holes)
        gmsh.model.occ.synchronize()

        # Refine towards the hole boundaries: that is where the strain
        # concentrates and where the effective response is decided.
        surfaces = [t for (d, t) in gmsh.model.getEntities(2)]

        # Classify outer edges by CENTRE OF MASS, not by bounding box: OCC
        # inflates bounding boxes by ~1e-7, so a tight box test silently
        # classifies nothing as outer -- which had the refinement field
        # refining the outer boundary along with the pore rim.
        outer_tol = 1.0e-6 * block
        edges = {}
        hole_curves = []
        for (d, t) in gmsh.model.getEntities(1):
            cx, cy, _cz = gmsh.model.occ.getCenterOfMass(d, t)
            if abs(cx + half) < outer_tol:
                edges["left"] = t
            elif abs(cx - half) < outer_tol:
                edges["right"] = t
            elif abs(cy + half) < outer_tol:
                edges["bottom"] = t
            elif abs(cy - half) < outer_tol:
                edges["top"] = t
            else:
                hole_curves.append(t)
        if len(edges) != 4:
            raise RuntimeError(f"expected 4 outer edges, identified {sorted(edges)}")
        if len(hole_curves) != n_cells ** 2:
            raise RuntimeError(f"expected {n_cells ** 2} hole curves, "
                               f"found {len(hole_curves)}")

        fdist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fdist, "CurvesList", hole_curves)
        gmsh.model.mesh.field.setNumber(fdist, "Sampling", 200)
        fthr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(fthr, "InField", fdist)
        gmsh.model.mesh.field.setNumber(fthr, "SizeMin", size_hole)
        gmsh.model.mesh.field.setNumber(fthr, "SizeMax", size_far)
        gmsh.model.mesh.field.setNumber(fthr, "DistMin", 0.05 * cell)
        gmsh.model.mesh.field.setNumber(fthr, "DistMax", 0.45 * cell)
        gmsh.model.mesh.field.setAsBackgroundMesh(fthr)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        if periodic:
            # Matching node distributions on opposite faces, which periodic
            # BCs require: a slave node must have a master at exactly the
            # translated position, or the constraint cannot be written.
            tx = [1, 0, 0, block, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            ty = [1, 0, 0, 0, 0, 1, 0, block, 0, 0, 1, 0, 0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, [edges["right"]], [edges["left"]], tx)
            gmsh.model.mesh.setPeriodic(1, [edges["top"]], [edges["bottom"]], ty)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tags = tags[order].astype(np.int64)
        coords = xyz.reshape(-1, 3)[order][:, :2]
        tag_to_idx = {int(t): i for i, t in enumerate(tags)}

        tris = []
        for s in surfaces:
            etypes, _etags, enodes = gmsh.model.mesh.getElements(2, s)
            for et, en in zip(etypes, enodes):
                if int(et) == GMSH_TRI6:
                    conn = np.asarray(en, dtype=np.int64).reshape(-1, 6)
                    tris.append(np.vectorize(tag_to_idx.__getitem__)(conn))
        if not tris:
            raise RuntimeError("no Triangle6 elements produced")
        tris = np.vstack(tris)
    finally:
        gmsh.finalize()

    # Drop nodes no element references (OCC construction points can survive).
    referenced = np.zeros(coords.shape[0], dtype=bool)
    referenced[tris.ravel()] = True
    if not np.all(referenced):
        keep = np.where(referenced)[0]
        remap = -np.ones(coords.shape[0], dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        coords = coords[keep]
        tris = remap[tris]
        assert np.all(tris >= 0)

    _verify_tri6_ordering(coords, tris)

    # The Dirichlet set is identified GEOMETRICALLY, on the outer square only.
    # Independent of gmsh tag bookkeeping, and it reproduces exactly what the
    # validated core/rve_geometry.mdpa does (all its 160 dirichlet nodes lie
    # on the outer boundary; the hole rim is free).
    tol = 1.0e-7 * max(1.0, block)
    on_outer = (np.abs(np.abs(coords[:, 0]) - half) < tol) | \
               (np.abs(np.abs(coords[:, 1]) - half) < tol)
    outer_nodes = np.where(on_outer)[0]

    solid = _mesh_area(coords, tris)
    geom = dict(n_cells=int(n_cells), cell_side=cell, block_side=block,
                block_area=block ** 2, porosity_target=float(porosity),
                aspect=float(aspect), angle_deg=float(angle_deg),
                a=float(a), b=float(b), centres=centres,
                solid_area=float(solid),
                porosity_mesh=float(1.0 - solid / block ** 2),
                n_nodes=int(coords.shape[0]), n_elements=int(tris.shape[0]),
                n_dirichlet=int(outer_nodes.size))
    return coords, tris, outer_nodes, geom


def _mesh_area(coords, tris):
    a = coords[tris[:, 0]]
    b = coords[tris[:, 1]]
    c = coords[tris[:, 2]]
    return float(np.sum(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                     - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))))


def _verify_tri6_ordering(coords, tris):
    """Confirm gmsh's Tri6 ordering matches the project's convention (corners
    0,1,2 then mid(0,1), mid(1,2), mid(2,0)) by checking real midpoint
    coordinates, rather than trusting either party's documentation.

    Elements with an edge on the elliptical rim carry a midside node placed ON
    the curve, not at the straight chord midpoint, so a bounded per-edge
    discrepancy is expected and correct. A genuine ordering bug shows up as
    errors of order a full element size on most elements, not on a small
    fraction, so the test is on the MEDIAN.
    """
    h = np.sqrt(_mesh_area(coords, tris) / tris.shape[0])
    err = []
    for (i, j, m) in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
        mid = 0.5 * (coords[tris[:, i]] + coords[tris[:, j]])
        err.append(np.linalg.norm(coords[tris[:, m]] - mid, axis=1))
    err = np.concatenate(err)
    if np.median(err) > 0.05 * h:
        raise RuntimeError(
            f"Tri6 node ordering does not match the project convention: "
            f"median midside offset {np.median(err):.3e} vs element size {h:.3e}")
    return float(np.median(err) / h), float(np.max(err) / h)


def write_mdpa(path, coords, tris, outer_nodes):
    """Write the exact format core/rve_geometry.mdpa uses (1-based ids)."""
    lines = ["Begin ModelPartData", "//  VARIABLE_NAME value", "End ModelPartData", "",
             "Begin Properties 0", "End Properties", "Begin Nodes"]
    for i, (x, y) in enumerate(coords, start=1):
        lines.append(f"    {i}  {x:.10f}  {y:.10f}  0.0000000000")
    lines += ["End Nodes", "",
              "Begin Geometries Triangle2D6 // GUI group identifier: material"]
    for e, t in enumerate(tris, start=1):
        lines.append("    " + str(e) + "   " + "  ".join(str(int(v) + 1) for v in t))
    lines += ["End Geometries", "",
              "Begin SubModelPart material // Group material",
              "    Begin SubModelPartNodes"]
    lines += [f"        {i}" for i in range(1, coords.shape[0] + 1)]
    lines += ["    End SubModelPartNodes", "    Begin SubModelPartGeometries"]
    lines += [f"        {e}" for e in range(1, tris.shape[0] + 1)]
    lines += ["    End SubModelPartGeometries", "End SubModelPart",
              "Begin SubModelPart dirichlet // Group dirichlet",
              "    Begin SubModelPartNodes"]
    lines += [f"        {int(i) + 1}" for i in outer_nodes]
    lines += ["    End SubModelPartNodes", "End SubModelPart", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _self_test():
    """Acceptance checks for this stage's mesh, before any solve trusts it."""
    ok = True
    for n in (1, 2):
        coords, tris, outer, geom = build_mesh(n_cells=n)

        # 1. Porosity actually realised, against the analytic target. The mesh
        #    inscribes a polygon in the ellipse, so solid area comes out
        #    slightly HIGH and porosity slightly LOW -- a signed, bounded
        #    discrepancy, not a free-floating tolerance.
        p_err = geom["porosity_mesh"] - geom["porosity_target"]
        good_p = (-0.01 < p_err <= 1.0e-12)

        # 2. Every Dirichlet node on the outer square, and none on the rim.
        half = geom["block_side"] / 2.0
        d = coords[outer]
        good_d = bool(np.all((np.abs(np.abs(d[:, 0]) - half) < 1.0e-7)
                             | (np.abs(np.abs(d[:, 1]) - half) < 1.0e-7)))

        # 3. Hole rim genuinely free: no node inside any ellipse.
        good_h = True
        t = np.radians(geom["angle_deg"])
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        for (xc, yc) in geom["centres"]:
            loc = (coords - np.array([xc, yc])) @ R.T
            f = (loc[:, 0] / geom["a"]) ** 2 + (loc[:, 1] / geom["b"]) ** 2
            if np.min(f) < 1.0 - 1.0e-6:
                good_h = False

        med, mx = _verify_tri6_ordering(coords, tris)
        stage_ok = good_p and good_d and good_h
        ok = ok and stage_ok
        print(f"n_cells={n}: nodes={geom['n_nodes']} elems={geom['n_elements']} "
              f"dir={geom['n_dirichlet']}")
        print(f"   porosity mesh {geom['porosity_mesh'] * 100:.4f}% vs target "
              f"{geom['porosity_target'] * 100:.2f}%  (err {p_err:+.2e}) "
              f"{'OK' if good_p else 'FAIL'}")
        print(f"   dirichlet all on outer square: {'OK' if good_d else 'FAIL'}")
        print(f"   no node inside any hole:       {'OK' if good_h else 'FAIL'}")
        print(f"   Tri6 midside offset / h: median {med:.2e}, max {mx:.2e}  OK")
    print("MESH_SELFTEST_PASS" if ok else "MESH_SELFTEST_FAIL")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _self_test() else 1)
    coords, tris, outer, geom = build_mesh(n_cells=1, verbose=False)
    out = HERE / "rve_cell_1x1.mdpa"
    write_mdpa(out, coords, tris, outer)
    for k, v in geom.items():
        if k != "centres":
            print(f"  {k:20s} {v}")
    print(f"written: {out}")
