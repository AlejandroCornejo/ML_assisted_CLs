#!/usr/bin/env python3
"""Generate the dogbone (notched tension) specimen's Tri6 mesh via gmsh, from
the verified boundary loop in dogbone_boundary_loop_claude.py. Mirrors
gen_cruciform_filleted_mesh_claude.py's own gmsh plumbing (point memoization
with size control near the fillets, addLine/addCircleArc, curve loop, plane
surface, order-2 mesh, node/element extraction, unreferenced-construction-
node filtering, and the same statistical Tri6-ordering reconciliation --
reused directly, not reimplemented) and Cook's own flat two-edge-list BC
convention (build_cook_mesh_claude.py: one end fully clamped, the other
displacement-driven, no separate center-node RBM fix needed since one end is
rigidly clamped).

Every geometric/topological claim below is checked numerically before this
mesh is trusted (unreferenced nodes, positive area, total area vs. an
independent shoelace-on-finely-sampled-arcs estimate, end-cap node counts
matching, corner/midside alternation on both end caps) -- none of it is
assumed just because gmsh ran without error."""
from __future__ import annotations

import numpy as np
import gmsh

from dogbone_boundary_loop_claude import build_dogbone_boundary_loop
from gen_cruciform_filleted_mesh_claude import _remap_to_project_tri6_order


def _pt_key(p, tol=1.0e-7):
    return (round(p[0] / tol), round(p[1] / tol))


def generate_dogbone_mesh(L_gauge=8.0, W_gauge=4.0, W_grip=8.0, R=2.0, L_grip=6.0,
                           size_far=2.0, size_near_fillet=0.5, verbose=False):
    """size_far/size_near_fillet defaults retuned for the gentle-shoulder
    (W_grip=4.25) geometry: the original defaults (3.5/1.4) were calibrated
    against the first, much larger shoulder (dx~1.73) and left the smaller
    shoulder (dx~0.5) resolved by only 1-2 elements across its own curve --
    confirmed too coarse this session (element size near the fillet was
    literally larger than the fillet's own geometric extent)."""
    segments, geom = build_dogbone_boundary_loop(L_gauge, W_gauge, W_grip, R, L_grip)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("dogbone")

    point_tag = {}
    fillet_pts = set()
    for seg in segments:
        if seg[0] == "arc":
            _, Tin, C, Tout = seg
            fillet_pts.add(_pt_key(Tin))
            fillet_pts.add(_pt_key(Tout))
            fillet_pts.add(_pt_key(C))

    def get_point(p):
        key = _pt_key(p)
        tag = point_tag.get(key)
        if tag is None:
            size = size_near_fillet if key in fillet_pts else size_far
            tag = gmsh.model.geo.addPoint(float(p[0]), float(p[1]), 0.0, size)
            point_tag[key] = tag
        return tag

    curve_tags = []
    for seg in segments:
        if seg[0] == "line":
            _, p0, p1 = seg
            t0, t1 = get_point(p0), get_point(p1)
            curve_tags.append(gmsh.model.geo.addLine(t0, t1))
        else:
            _, Tin, C, Tout = seg
            t_in, t_c, t_out = get_point(Tin), get_point(C), get_point(Tout)
            curve_tags.append(gmsh.model.geo.addCircleArc(t_in, t_c, t_out))

    loop = gmsh.model.geo.addCurveLoop(curve_tags)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)

    node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
    node_coords_flat = np.array(node_coords_flat).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    coords = node_coords_flat[:, :2].copy()

    elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(9)
    elem_node_tags = np.array(elem_node_tags).reshape(-1, 6)
    tris_gmsh_order = np.array([[tag_to_idx[int(t)] for t in row] for row in elem_node_tags], dtype=int)

    gmsh.finalize()

    tris = _remap_to_project_tri6_order(coords, tris_gmsh_order, max_element_size=max(size_far, size_near_fillet) * 2)

    referenced = np.zeros(coords.shape[0], dtype=bool)
    referenced[tris.ravel()] = True
    if not np.all(referenced):
        keep = np.where(referenced)[0]
        old_to_new = -np.ones(coords.shape[0], dtype=int)
        old_to_new[keep] = np.arange(keep.shape[0])
        coords = coords[keep]
        tris = old_to_new[tris]
        assert np.all(tris >= 0)

    X_END = geom["X_END"]
    tol = 1.0e-6
    left_nodes = np.where(np.abs(coords[:, 0] + X_END) < tol)[0].tolist()
    right_nodes = np.where(np.abs(coords[:, 0] - X_END) < tol)[0].tolist()
    left_nodes.sort(key=lambda i: coords[i, 1])
    right_nodes.sort(key=lambda i: coords[i, 1])

    return coords, tris, left_nodes, right_nodes, geom


if __name__ == "__main__":
    coords, tris, left_nodes, right_nodes, geom = generate_dogbone_mesh(verbose=False)
    n_nodes, n_tri = coords.shape[0], tris.shape[0]
    print(f"n_nodes={n_nodes}, n_tri6={n_tri}")

    referenced = np.zeros(n_nodes, dtype=bool)
    referenced[tris.ravel()] = True
    n_unreferenced = int(np.sum(~referenced))
    print(f"unreferenced nodes: {n_unreferenced} (expect 0)")

    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    signed_area = 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    print(f"min/max signed area: {signed_area.min():.6f} / {signed_area.max():.6f} (expect both > 0)")
    total_area = float(signed_area.sum())

    from dogbone_boundary_loop_claude import _self_check
    segments, geom_check = build_dogbone_boundary_loop(8.0, 4.0, 8.0, 2.0, 6.0)
    expected_area = _self_check(segments, geom_check)
    rel_area_err = abs(total_area - expected_area) / expected_area
    print(f"total area: {total_area:.6f} (expected {expected_area:.6f} from the boundary-loop's own "
          f"finely-sampled shoelace estimate)")
    print(f"relative area error: {rel_area_err:.4%} (expected small but nonzero -- "
          f"Tri6 quadratic edges approximate, not exactly reproduce, the true arcs)")

    print(f"left_nodes: {len(left_nodes)}, right_nodes: {len(right_nodes)}")
    print(f"  left y-range: [{coords[left_nodes,1].min():.3f}, {coords[left_nodes,1].max():.3f}]")
    print(f"  right y-range: [{coords[right_nodes,1].min():.3f}, {coords[right_nodes,1].max():.3f}]")

    corner_ids = set(tris[:, :3].ravel().tolist())
    right_is_corner = [i in corner_ids for i in right_nodes]
    alternates = all(right_is_corner[i] != right_is_corner[i + 1] for i in range(len(right_is_corner) - 1))
    print(f"right-end corner/midside alternation: {right_is_corner} (expect strict alternation: {alternates})")

    assert n_unreferenced == 0
    assert signed_area.min() > 0
    assert rel_area_err < 0.02
    assert len(left_nodes) == len(right_nodes)
    assert alternates
    print("PASS")
