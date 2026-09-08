#!/usr/bin/env python3
"""Plain square panel mesh (no arms, no holes, no fillets) -- the exactly-
uniform biaxial benchmark: with a homogeneous macro material and an affine
uniform displacement BC on a simple convex domain, the unique FE solution is
the trivial uniform state everywhere (constant stress -> equilibrium holds
trivially; the BC is met exactly) -- E12=0 identically, not approximately,
by uniqueness rather than by any geometric argument about corners.

Reuses build_cruciform_mesh_claude.py's own _build_patch (the same
bilinear-grid-to-Tri6 splitting recipe already used and verified for every
Cook/Cruciform patch this project has built) on a single n_body x n_body
square patch -- no new meshing logic at all."""
from __future__ import annotations

import numpy as np

from build_cruciform_mesh_claude import _build_patch


def build_square_panel_mesh(n_body: int, L_body: float = 12.0):
    half = L_body / 2.0
    coords, tris, edge_nodes = _build_patch(-half, half, -half, half, n_body, n_body)

    tip_nodes = {
        "px": edge_nodes["right"],
        "mx": edge_nodes["left"],
        "py": edge_nodes["top"],
        "my": edge_nodes["bottom"],
    }
    center_node = int(np.argmin(np.sum(coords ** 2, axis=1)))
    return coords, tris, tip_nodes, center_node


if __name__ == "__main__":
    L_body = 12.0
    coords, tris, tip_nodes, center_node = build_square_panel_mesh(n_body=8, L_body=L_body)
    print(f"n_nodes={coords.shape[0]}, n_tri6={tris.shape[0]}")

    referenced = np.zeros(coords.shape[0], dtype=bool)
    referenced[tris.ravel()] = True
    print(f"unreferenced nodes: {int(np.sum(~referenced))} (expect 0)")

    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    signed_area = 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    total_area = float(signed_area.sum())
    print(f"min/max signed area: {signed_area.min():.6f} / {signed_area.max():.6f} (expect both > 0)")
    print(f"total area: {total_area:.6f} (expected {L_body ** 2:.6f})")

    print(f"edge node counts: px={len(tip_nodes['px'])}, mx={len(tip_nodes['mx'])}, "
          f"py={len(tip_nodes['py'])}, my={len(tip_nodes['my'])}")
    print(f"center node: {center_node}, coords={coords[center_node]} (expect ~(0,0))")

    assert int(np.sum(~referenced)) == 0
    assert signed_area.min() > 0
    assert abs(total_area - L_body ** 2) < 1e-6
    assert np.allclose(coords[center_node], [0.0, 0.0], atol=1e-9)
    print("PASS")
