#!/usr/bin/env python3
"""Plain rectangular bar, uniform cross-section, no notch/fillet anywhere --
the limiting case of the dogbone with the shoulder removed entirely, to
isolate whether the shoulder's own geometric concentration (confirmed this
session: peak shoulder shear/E11 ratio ~0.90-0.92, essentially constant
across load level, exactly the textbook notion of a load-independent strain-
concentration factor) is really what caps how large an E11 this project can
reach while staying inside the trained box, or whether something else (e.g.
a St Venant-type boundary effect right at the clamped/pulled ends, even with
no notch at all) would cap it first.

Reuses build_cruciform_mesh_claude.py's own _build_patch directly (same
helper build_square_panel_mesh_claude.py already reuses for a plain square),
just with an elongated rectangle instead of a square -- no new mesh-
generation code needed at all.

Same total length as the dogbone (2*X_END = 26.9282...) and same width as
the dogbone's own gauge section (4.0), for a direct, apples-to-apples
comparison at the same nominal delta values."""
from __future__ import annotations

import numpy as np

from build_cruciform_mesh_claude import _build_patch

L_TOTAL_DEFAULT = 26.928203230275506  # matches the dogbone's own 2*X_END
W_DEFAULT = 4.0  # matches the dogbone's own gauge width


def generate_plain_bar_mesh(L=L_TOTAL_DEFAULT, W=W_DEFAULT, nx=25, ny=4):
    half_L, half_W = L / 2.0, W / 2.0
    coords, tris, edge_nodes = _build_patch(-half_L, half_L, -half_W, half_W, nx, ny)
    left_nodes = edge_nodes["left"]
    right_nodes = edge_nodes["right"]
    geom = {"L": L, "W": W, "X_END": half_L}
    return coords, tris, left_nodes, right_nodes, geom


if __name__ == "__main__":
    coords, tris, left_nodes, right_nodes, geom = generate_plain_bar_mesh()
    n_nodes, n_tri = coords.shape[0], tris.shape[0]
    print(f"n_nodes={n_nodes}, n_tri6={n_tri}, geom={geom}")

    referenced = np.zeros(n_nodes, dtype=bool)
    referenced[tris.ravel()] = True
    print(f"unreferenced nodes: {int(np.sum(~referenced))} (expect 0)")

    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    signed_area = 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    total_area = float(signed_area.sum())
    expected_area = geom["L"] * geom["W"]
    print(f"min/max signed area: {signed_area.min():.6f} / {signed_area.max():.6f} (expect both > 0)")
    print(f"total area: {total_area:.6f} (expected exactly {expected_area:.6f}, plain rectangle)")

    print(f"left_nodes: {len(left_nodes)}, right_nodes: {len(right_nodes)}")
    print(f"  left y-range: [{coords[left_nodes,1].min():.3f}, {coords[left_nodes,1].max():.3f}]")
    print(f"  right y-range: [{coords[right_nodes,1].min():.3f}, {coords[right_nodes,1].max():.3f}]")

    corner_ids = set(tris[:, :3].ravel().tolist())
    right_is_corner = [i in corner_ids for i in right_nodes]
    alternates = all(right_is_corner[i] != right_is_corner[i + 1] for i in range(len(right_is_corner) - 1))
    print(f"right-end corner/midside alternation: expect True: {alternates}")

    assert int(np.sum(~referenced)) == 0
    assert signed_area.min() > 0
    assert abs(total_area - expected_area) < 1.0e-9
    assert len(left_nodes) == len(right_nodes)
    assert alternates
    print("PASS")
