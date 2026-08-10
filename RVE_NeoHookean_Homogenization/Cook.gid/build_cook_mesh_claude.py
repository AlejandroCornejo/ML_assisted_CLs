"""Builds a conforming Tri6 (6-node quadratic triangle) mesh of Cook's
membrane directly via the Kratos Python API (no mdpa file, no GiD
Geometries indirection). Classic Cook's-membrane corners (0,0), (48,44),
(48,60), (0,44) [meters here, matching Cook.gid's own convention], meshed
by a bilinear blend of a fine (2*nx+1)x(2*ny+1) parametric grid, split
into two Tri6 per coarse cell sharing the diagonal midside node -- a
standard, conforming way to build a quadratic-triangle mesh without any
duplicate-node bookkeeping.

Kratos Triangle2D6 local node order (verified against
core/fom_solver_rve.py's _tri6_DN_local): 0=(0,0), 1=(1,0), 2=(0,1),
3=mid(0,1), 4=mid(1,2), 5=mid(2,0).
"""
from __future__ import annotations

import numpy as np

# Classic Cook's membrane corners.
A = np.array([0.0, 0.0])
B = np.array([48.0, 44.0])
C = np.array([48.0, 60.0])
D = np.array([0.0, 44.0])


def _bilinear(xi, eta):
    return (1 - xi) * (1 - eta) * A + xi * (1 - eta) * B + xi * eta * C + (1 - xi) * eta * D


def build_mesh(nx: int, ny: int):
    """Returns (coords (n_nodes,2), tri6_connectivity (n_tri,6) 0-based,
    left_edge_node_ids, right_edge_node_ids [all 0-based]).
    """
    nfx, nfy = 2 * nx + 1, 2 * ny + 1
    xis = np.linspace(0.0, 1.0, nfx)
    etas = np.linspace(0.0, 1.0, nfy)
    coords = np.empty((nfx, nfy, 2))
    for i, xi in enumerate(xis):
        for j, eta in enumerate(etas):
            coords[i, j] = _bilinear(xi, eta)

    def idx(i, j):
        return i * nfy + j

    coords_flat = coords.reshape(-1, 2)

    tris = []
    for i in range(nx):
        for j in range(ny):
            # fine-grid corners of this coarse cell
            p00 = (2 * i, 2 * j)
            p20 = (2 * i + 2, 2 * j)
            p22 = (2 * i + 2, 2 * j + 2)
            p02 = (2 * i, 2 * j + 2)
            # edge/diagonal midsides
            m00_20 = (2 * i + 1, 2 * j)
            m20_22 = (2 * i + 2, 2 * j + 1)
            m22_00 = (2 * i + 1, 2 * j + 1)  # shared diagonal midside
            m02_00 = (2 * i, 2 * j + 1)
            m22_02 = (2 * i + 1, 2 * j + 2)

            # Triangle A: 0=p00, 1=p20, 2=p22 ; 3=mid(0,1), 4=mid(1,2), 5=mid(2,0)
            tris.append([idx(*p00), idx(*p20), idx(*p22),
                         idx(*m00_20), idx(*m20_22), idx(*m22_00)])
            # Triangle B: 0=p00, 1=p22, 2=p02 ; 3=mid(0,1)=diagonal(shared), 4=mid(1,2), 5=mid(2,0)
            tris.append([idx(*p00), idx(*p22), idx(*p02),
                         idx(*m22_00), idx(*m22_02), idx(*m02_00)])

    tris = np.array(tris, dtype=int)

    left_nodes = [idx(0, j) for j in range(nfy)]
    right_nodes = [idx(nfx - 1, j) for j in range(nfy)]

    return coords_flat, tris, left_nodes, right_nodes


if __name__ == "__main__":
    coords, tris, left_nodes, right_nodes = build_mesh(nx=10, ny=10)
    print("n_nodes", coords.shape[0], "n_tri6", tris.shape[0])
    # sanity: signed area of every triangle (corners 0,1,2) must be positive (CCW)
    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    signed_area = 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                         - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    print("min/max signed area:", signed_area.min(), signed_area.max())
    print("total area:", signed_area.sum(), "(expected ~ (44+16)/2*48 =", (44 + 16) / 2 * 48, ")")
    print("n_left", len(left_nodes), "n_right", len(right_nodes))
