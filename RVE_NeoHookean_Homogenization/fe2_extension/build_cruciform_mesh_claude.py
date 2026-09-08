#!/usr/bin/env python3
"""Builds a conforming Tri6 (6-node quadratic triangle) mesh of a
cruciform (plus-shaped) biaxial specimen: a square central body with 4
rectangular arms extending outward, sharp (unfilleted) internal corners
-- same level of geometric simplicity as Cook.gid/build_cook_mesh_claude.
py's own trapezoid (which also has sharp corners), reusing the SAME
per-patch bilinear-grid-to-Tri6 splitting scheme, applied to 5 rectangular
patches (body + 4 arms) instead of Cook's single trapezoid, stitched into
one global mesh by coordinate matching at shared edges.

Why a cruciform, not Cook's own geometry: this project's RVE was
characterized (10 training trajectories) almost entirely on BIAXIAL
stretch states (E11, E22 growing together, up to 200% strain, with only a
small superimposed shear offset) -- the opposite of Cook's own shear/
bending-dominated state (verified this session: Cook's own mean strain
direction has shear as its dominant component). A cruciform pulled
symmetrically from its 4 arms produces a purely biaxial, shear-free state
in its entire central region regardless of the pull ratio between arm
pairs (unlike an inflating/expanding ring, where the apparent shear in a
fixed material frame varies with angular position unless the expansion is
exactly equibiaxial) -- so its bulk state matches training by construction,
not by accident, with only the small re-entrant corner regions (where each
arm meets the body) carrying any local complexity.

Local Tri6 node order matches Cook's own convention exactly (verified
against core/fom_solver_rve.py's _tri6_DN_local): 0=(0,0), 1=(1,0),
2=(0,1), 3=mid(0,1), 4=mid(1,2), 5=mid(2,0).
"""
from __future__ import annotations

import numpy as np


def _build_patch(x0, x1, y0, y1, nx, ny):
    """Same fine-grid-splitting recipe as Cook's build_mesh, but on an
    axis-aligned rectangle (no bilinear blend needed -- the 4 corners of
    an axis-aligned rectangle already give a trivial affine map)."""
    nfx, nfy = 2 * nx + 1, 2 * ny + 1
    xs = np.linspace(x0, x1, nfx)
    ys = np.linspace(y0, y1, nfy)
    coords = np.empty((nfx, nfy, 2))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            coords[i, j] = (x, y)

    def idx(i, j):
        return i * nfy + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            p00, p20, p22, p02 = (2 * i, 2 * j), (2 * i + 2, 2 * j), (2 * i + 2, 2 * j + 2), (2 * i, 2 * j + 2)
            m00_20 = (2 * i + 1, 2 * j)
            m20_22 = (2 * i + 2, 2 * j + 1)
            m22_00 = (2 * i + 1, 2 * j + 1)
            m02_00 = (2 * i, 2 * j + 1)
            m22_02 = (2 * i + 1, 2 * j + 2)
            tris.append([idx(*p00), idx(*p20), idx(*p22), idx(*m00_20), idx(*m20_22), idx(*m22_00)])
            tris.append([idx(*p00), idx(*p22), idx(*p02), idx(*m22_00), idx(*m22_02), idx(*m02_00)])

    coords_flat = coords.reshape(-1, 2)
    edge_nodes = {
        "left": [idx(0, j) for j in range(nfy)],
        "right": [idx(nfx - 1, j) for j in range(nfy)],
        "bottom": [idx(i, 0) for i in range(nfx)],
        "top": [idx(i, nfy - 1) for i in range(nfx)],
    }
    return coords_flat, np.array(tris, dtype=int), edge_nodes


def build_cruciform_mesh(n_body: int, n_arm_len: int, L_body: float = 12.0,
                          arm_width_fraction: float = 1.0 / 3.0, L_arm: float = 8.0,
                          tol: float = 1.0e-7):
    """5-patch cruciform: central n_body x n_body square body (side
    L_body), 4 arms of length L_arm and width L_body*arm_width_fraction,
    each n_arm_len x n_width coarse cells (n_width chosen so the arm's own
    cell size along its width matches the body's -- required for the
    shared edge's nodes to coincide exactly).

    Matching cell size alone is NOT sufficient for a conforming Tri6
    mesh: the arm's own corner-node row at its width boundary
    (w_lo = -W_arm/2) must also land on one of the body's own CORNER
    rows, not one of its mid-side rows. If it lands on a mid-side row
    instead, the raw node coordinates still coincide and get merged by
    get_global_id below, but the arm's corner-to-corner Tri6 edges there
    have no matching corner-to-corner edge on the body side -- a real,
    if easy to miss, non-conforming interface (confirmed directly this
    session: at n_body=9, arm_width_fraction=2/3, the merged mesh's own
    total open/boundary edge length comes out to 176m against a true
    analytic perimeter of 112m, i.e. each arm floats, disconnected from
    the body, along its entire width). The alignment condition is
    k := n_body*(1-arm_width_fraction)/2 must be an integer (that
    integer is the body's own corner-row index the arm's w_lo lands on);
    for the default arm_width_fraction=1/3 this reduces to n_body%3==0,
    which is the only case an earlier version of this assertion checked.

    Returns (coords (n_nodes,2), tris (n_tri,6) global 0-based,
    tip_nodes dict with 'px','mx','py','my' (far-end column/row of each
    arm), center_node (single node at the body's own centroid, for the
    rigid-body-motion constraint)).
    """
    k_align = n_body * (1.0 - arm_width_fraction) / 2.0
    assert abs(k_align - round(k_align)) < 1e-9, (
        f"n_body={n_body} does not align arm_width_fraction={arm_width_fraction}'s arm "
        f"edges with the body's own corner grid (need n_body*(1-arm_width_fraction)/2 "
        f"integer, got {k_align}); the arm's corner grid would sit exactly on the body's "
        f"mid-side nodes instead of its corners, producing a non-conforming Tri6 mesh at "
        f"the interface even though the raw node coordinates still coincide and merge."
    )
    h = L_body / n_body
    W_arm = L_body * arm_width_fraction
    n_width = int(round(W_arm / h))
    assert abs(n_width * h - W_arm) < 1e-9, "arm width is not an integer multiple of the body's own cell size"
    w_lo = -W_arm / 2.0
    w_hi = W_arm / 2.0

    patches = {
        "body": _build_patch(-L_body / 2, L_body / 2, -L_body / 2, L_body / 2, n_body, n_body),
        "px": _build_patch(L_body / 2, L_body / 2 + L_arm, w_lo, w_hi, n_arm_len, n_width),
        "mx": _build_patch(-L_body / 2 - L_arm, -L_body / 2, w_lo, w_hi, n_arm_len, n_width),
        "py": _build_patch(w_lo, w_hi, L_body / 2, L_body / 2 + L_arm, n_width, n_arm_len),
        "my": _build_patch(w_lo, w_hi, -L_body / 2 - L_arm, -L_body / 2, n_width, n_arm_len),
    }

    global_coords = []
    coord_to_global = {}

    def get_global_id(xy):
        key = (round(xy[0] / tol), round(xy[1] / tol))
        gid = coord_to_global.get(key)
        if gid is None:
            gid = len(global_coords)
            coord_to_global[key] = gid
            global_coords.append(xy)
        return gid

    global_tris = []
    patch_edge_globals = {}
    for name, (coords, tris, edge_nodes) in patches.items():
        local_to_global = np.array([get_global_id(xy) for xy in coords], dtype=int)
        for tri in tris:
            global_tris.append(local_to_global[tri].tolist())
        patch_edge_globals[name] = {side: local_to_global[ids].tolist() for side, ids in edge_nodes.items()}

    coords_arr = np.array(global_coords, dtype=float)
    tris_arr = np.array(global_tris, dtype=int)

    tip_nodes = {
        "px": patch_edge_globals["px"]["right"],
        "mx": patch_edge_globals["mx"]["left"],
        "py": patch_edge_globals["py"]["top"],
        "my": patch_edge_globals["my"]["bottom"],
    }
    center_dist = np.sum(coords_arr ** 2, axis=1)
    center_node = int(np.argmin(center_dist))

    return coords_arr, tris_arr, tip_nodes, center_node


if __name__ == "__main__":
    coords, tris, tip_nodes, center_node = build_cruciform_mesh(n_body=6, n_arm_len=4)
    print(f"n_nodes={coords.shape[0]}, n_tri6={tris.shape[0]}")

    # No duplicate/orphan nodes: every node must be referenced by >=1 element.
    referenced = np.zeros(coords.shape[0], dtype=bool)
    referenced[tris.ravel()] = True
    print(f"unreferenced nodes: {int(np.sum(~referenced))} (expect 0)")

    # Positive signed area (CCW) for every element.
    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    signed_area = 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    print(f"min/max signed area: {signed_area.min():.6f} / {signed_area.max():.6f} (expect both > 0)")
    total_area = float(signed_area.sum())
    L_body, arm_frac, L_arm = 12.0, 1.0 / 3.0, 8.0
    expected_area = L_body ** 2 + 4 * (L_body * arm_frac) * L_arm
    print(f"total area: {total_area:.6f} (expected {expected_area:.6f})")

    print(f"tip node counts: px={len(tip_nodes['px'])}, mx={len(tip_nodes['mx'])}, "
          f"py={len(tip_nodes['py'])}, my={len(tip_nodes['my'])}")
    print(f"center node: {center_node}, coords={coords[center_node]} (expect ~(0,0))")

    assert int(np.sum(~referenced)) == 0
    assert signed_area.min() > 0
    assert abs(total_area - expected_area) < 1e-6
    assert np.allclose(coords[center_node], [0.0, 0.0], atol=1e-9)
    print("PASS")
