"""Isolated material-B geometry; reuse only the existing Tri6 MDPA writer.

Lengths use the same arbitrary cell scale as material A. No A defaults change.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_rve"))
from gen_rve_mesh import _verify_tri6_ordering, write_mdpa


def cavity_parameters(spec):
    L = spec["cell_side"]
    result = []
    for hole in spec["cavities"]:
        b = L * np.sqrt(hole["area_over_cell"] / (np.pi * hole["aspect"]))
        result.append(dict(center=L * np.array(hole["center_over_L"]),
                           a=hole["aspect"] * b, b=b, angle=hole["angle_deg"]))
    return result


def geometry_checks(spec):
    """Conservative separation certificate using enclosing circles, not sampling."""
    L = spec["cell_side"]
    holes = cavity_parameters(spec)
    gaps = []
    for i, hi in enumerate(holes):
        for j, hj in enumerate(holes):
            for shift in itertools.product((-1, 0, 1), repeat=2):
                if i == j and shift == (0, 0):
                    continue
                delta = hj["center"] + L * np.array(shift) - hi["center"]
                gaps.append(np.linalg.norm(delta) - hi["a"] - hj["a"])
    boundary_gaps = []
    for h in holes:
        t = np.radians(h["angle"])
        extents = np.array([np.hypot(h["a"] * np.cos(t), h["b"] * np.sin(t)),
                            np.hypot(h["a"] * np.sin(t), h["b"] * np.cos(t))])
        boundary_gaps.extend(L / 2 - np.abs(h["center"]) - extents)
    porosity = sum(h["area_over_cell"] for h in spec["cavities"])
    if abs(porosity - spec["porosity"]) > 1e-12:
        raise ValueError("Cavity areas do not sum to the target porosity")
    if min(gaps) <= 0 or min(boundary_gaps) <= 0:
        raise ValueError("Conservative cavity separation or boundary clearance failed")
    return dict(porosity_analytic=porosity,
                periodic_ligament_lower_bound=float(min(gaps)),
                boundary_clearance_lower_bound=float(min(boundary_gaps)),
                rationale="Disjoint enclosed ellipses leave a connected solid. "
                "Nearest periodic images suffice; farther translations increase at least "
                "one coordinate separation. Circle gaps are lower bounds, not exact ligaments.")


def check_mesh(xy, triangles, L):
    """Numerical checks, including sampled curved-Tri6 reference Jacobians."""
    _verify_tri6_ordering(xy, triangles)
    face_errors, counts = [], []
    for axis in (0, 1):
        faces = [np.sort(xy[np.isclose(xy[:, axis], s * L / 2,
                                      atol=1e-8, rtol=0), 1 - axis]) for s in (-1, 1)]
        if len(faces[0]) != len(faces[1]) or not len(faces[0]):
            raise ValueError("Periodic face node count mismatch")
        face_errors.append(float(np.max(np.abs(faces[0] - faces[1]))))
        counts.append(len(faces[0]))
    if max(face_errors) > 1e-8:
        raise ValueError("Periodic face coordinates mismatch")
    rows = np.repeat(triangles[:, :1], 5, axis=1).ravel()
    cols = triangles[:, 1:].ravel()
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(xy), len(xy)))
    components = connected_components(graph, directed=False, return_labels=False)
    if components != 1:
        raise ValueError("Disconnected solid mesh")
    min_det = np.inf
    # Shape functions: [l(2l-1), r(2r-1), s(2s-1), 4lr, 4rs, 4sl].
    for i in range(11):
        for j in range(11 - i):
            r, s = i / 10, j / 10
            ell = 1 - r - s
            grad = np.array([[1-4*ell, 1-4*ell], [4*r-1, 0], [0, 4*s-1],
                             [4*(ell-r), -4*r], [4*s, 4*r], [-4*s, 4*(ell-s)]])
            jac = np.einsum("eni,nj->eij", xy[triangles], grad)
            min_det = min(min_det, float(np.linalg.det(jac).min()))
    if min_det <= 0:
        raise ValueError("Nonpositive sampled reference-element Jacobian")
    corners = xy[triangles[:, :3]]
    edges = np.roll(corners, -1, axis=1) - corners
    twice_area = np.abs(edges[:, 0, 0] * (-edges[:, 2, 1])
                        - edges[:, 0, 1] * (-edges[:, 2, 0]))
    quality = 2 * np.sqrt(3) * twice_area / np.sum(edges**2, axis=(1, 2))
    return dict(n_nodes=len(xy), n_elements=len(triangles), solid_components=int(components),
                periodic_face_nodes=counts, periodic_max_mismatch=max(face_errors),
                sampled_min_reference_jacobian=min_det,
                min_corner_triangle_quality=float(quality.min()),
                note="Jacobian positivity checked at 66 points per Tri6; not an everywhere proof.")


def build(spec, mesh, output):
    import gmsh
    checks = geometry_checks(spec)
    L = spec["cell_side"]
    half = L / 2
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.model.add(spec["candidate_id"])
        square = gmsh.model.occ.addRectangle(-half, -half, 0, L, L)
        disks = []
        for hole in cavity_parameters(spec):
            x, y = hole["center"]
            disk = gmsh.model.occ.addDisk(x, y, 0, hole["a"], hole["b"])
            gmsh.model.occ.rotate([(2, disk)], x, y, 0, 0, 0, 1,
                                  np.radians(hole["angle"]))
            disks.append((2, disk))
        gmsh.model.occ.cut([(2, square)], disks)
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        if len(surfaces) != 1:
            raise ValueError("Expected one connected OCC surface")
        edges, rims = {}, []
        for dim, tag in gmsh.model.getEntities(1):
            cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
            if abs(cx + half) < 1e-6: edges["left"] = tag
            elif abs(cx - half) < 1e-6: edges["right"] = tag
            elif abs(cy + half) < 1e-6: edges["bottom"] = tag
            elif abs(cy - half) < 1e-6: edges["top"] = tag
            else: rims.append(tag)
        if len(edges) != 4 or len(rims) != len(disks):
            raise ValueError("Unexpected boundary topology")
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList", rims)
        gmsh.model.mesh.field.setNumber(dist, "Sampling", 200)
        threshold = gmsh.model.mesh.field.add("Threshold")
        for key, value in dict(InField=dist, SizeMin=mesh["size_hole"],
                               SizeMax=mesh["size_far"], DistMin=.05*L,
                               DistMax=.45*L).items():
            gmsh.model.mesh.field.setNumber(threshold, key, value)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        for option in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
            gmsh.option.setNumber("Mesh." + option, 0)
        for slave, master, dx, dy in (("right", "left", L, 0), ("top", "bottom", 0, L)):
            transform = [1, 0, 0, dx, 0, 1, 0, dy, 0, 0, 1, 0, 0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, [edges[slave]], [edges[master]], transform)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        tag_map = {int(tag): i for i, tag in enumerate(tags)}
        xy = xyz.reshape(-1, 3)[:, :2]
        _element_tags, nodes = gmsh.model.mesh.getElementsByType(9)
        triangles = np.array([tag_map[int(tag)] for tag in nodes]).reshape(-1, 6)
        used = np.unique(triangles)
        mapping = np.full(len(xy), -1, dtype=int)
        mapping[used] = np.arange(len(used))
        xy, triangles = xy[used], mapping[triangles]
        checks["gmsh_version"] = gmsh.__version__
        checks["occ_solid_area"] = gmsh.model.occ.getMass(*surfaces[0])
    finally:
        gmsh.finalize()
    # Match the precision actually written into the solver input.
    xy = np.round(xy, 10)
    checks.update(check_mesh(xy, triangles, L))
    outer = np.where(np.any(np.isclose(np.abs(xy), half, atol=1e-8, rtol=0), axis=1))[0]
    write_mdpa(output.with_suffix(".mdpa"), xy, triangles, outer)
    np.savez_compressed(output.with_suffix(".npz"), xy=xy, triangles=triangles)
    return checks, xy, triangles
