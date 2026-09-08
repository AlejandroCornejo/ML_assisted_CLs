#!/usr/bin/env python3
"""Macro mesh: the ASTM D638 Type I in-plane profile.

Geometry, worked out from the standard's own numbers rather than assumed.
Half-widths are W/2 = 6.5 mm in the gauge and WO/2 = 9.5 mm in the grip, so
the fillet rises 3.0 mm. A single arc of R = 76 mm tangent to the gauge edge
spans

    dx = sqrt(R^2 - (R - rise)^2) = sqrt(76^2 - 73^2) = 21.14 mm

and arrives at the grip half-width with slope dx/73 = 0.290, i.e. 16.1 deg.
So the fillet is tangent to the NARROW section and meets the wide section at a
kink. That is a consequence of the standard specifying a single R: tangency at
both ends would require a two-arc S-curve. The kink sits in the wide section,
where stress is lower by the width ratio 13/19 = 0.68, and far from the gauge.

The cotes close, which is the check that the profile is the standard's and not
an invention: gauge half-length 28.5 + fillet 21.14 = 49.64, then straight
grip out to LO/2 = 82.5, and the grips themselves clamp at D/2 = 57.5, i.e.
inside the straight region as they must.

NO mirror symmetry is exploited, deliberately. The material is group-free
anisotropic, so reflecting about the long axis maps the cell's ellipse from
+30 deg to -30 deg -- a different material. Only point symmetry survives, and
the pre-pass is cheap enough (no nested RVE) to run on the full coupon.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

GMSH_TRI6 = 9


def profile():
    """Key profile dimensions in metres, all derived from config."""
    w_g = cfg.COUPON_W_GAUGE / 2.0
    w_o = cfg.COUPON_W_GRIP / 2.0
    R = cfg.COUPON_FILLET_R
    rise = w_o - w_g
    if R < rise:
        raise ValueError("fillet radius smaller than the rise")
    dx = np.sqrt(R ** 2 - (R - rise) ** 2)
    x_gauge = cfg.COUPON_L_GAUGE / 2.0
    x_fillet_end = x_gauge + dx
    x_end = cfg.COUPON_L_TOTAL / 2.0
    if x_fillet_end >= x_end:
        raise ValueError("fillet runs past the specimen end")
    if cfg.COUPON_GRIP_SEP / 2.0 <= x_fillet_end:
        raise ValueError("grips would clamp inside the fillet, not the straight grip")
    return dict(w_gauge=w_g, w_grip=w_o, R=R, rise=rise, dx=dx,
                x_gauge=x_gauge, x_fillet_end=x_fillet_end, x_end=x_end,
                arc_centre_y=w_g + R,
                fillet_end_slope_deg=float(np.degrees(np.arctan2(dx, R - rise))))


def build_mesh(size_gauge=None, size_grip=None, verbose=False):
    """(coords (N,2), tris (M,6) 0-based, left_nodes, right_nodes, geom)."""
    import gmsh

    p = profile()
    size_gauge = cfg.COUPON_W_GAUGE / 8.0 if size_gauge is None else size_gauge
    size_grip = cfg.COUPON_W_GAUGE / 4.0 if size_grip is None else size_grip

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("coupon")
        g = gmsh.model.occ

        # Half profile above the axis, then mirrored, as one closed loop.
        # Points run counter-clockwise from the bottom-left corner.
        pts = []

        def P(x, y):
            pts.append(g.addPoint(x, y, 0.0))
            return pts[-1]

        # bottom edge, left to right
        p_bl = P(-p["x_end"], -p["w_grip"])
        p_br = P(p["x_end"], -p["w_grip"])
        # right cap
        p_tr = P(p["x_end"], p["w_grip"])
        p_tl = P(-p["x_end"], p["w_grip"])
        # fillet junctions, top
        p_t_fr = P(p["x_fillet_end"], p["w_grip"])
        p_t_gr = P(p["x_gauge"], p["w_gauge"])
        p_t_gl = P(-p["x_gauge"], p["w_gauge"])
        p_t_fl = P(-p["x_fillet_end"], p["w_grip"])
        # fillet junctions, bottom
        p_b_fr = P(p["x_fillet_end"], -p["w_grip"])
        p_b_gr = P(p["x_gauge"], -p["w_gauge"])
        p_b_gl = P(-p["x_gauge"], -p["w_gauge"])
        p_b_fl = P(-p["x_fillet_end"], -p["w_grip"])
        # arc centres
        c_t_r = P(p["x_gauge"], p["arc_centre_y"])
        c_t_l = P(-p["x_gauge"], p["arc_centre_y"])
        c_b_r = P(p["x_gauge"], -p["arc_centre_y"])
        c_b_l = P(-p["x_gauge"], -p["arc_centre_y"])

        L = []
        L.append(g.addLine(p_bl, p_b_fl))                 # bottom grip, left
        L.append(g.addCircleArc(p_b_fl, c_b_l, p_b_gl))   # bottom fillet, left
        L.append(g.addLine(p_b_gl, p_b_gr))               # bottom gauge
        L.append(g.addCircleArc(p_b_gr, c_b_r, p_b_fr))   # bottom fillet, right
        L.append(g.addLine(p_b_fr, p_br))                 # bottom grip, right
        L.append(g.addLine(p_br, p_tr))                   # right end face
        L.append(g.addLine(p_tr, p_t_fr))                 # top grip, right
        L.append(g.addCircleArc(p_t_fr, c_t_r, p_t_gr))   # top fillet, right
        L.append(g.addLine(p_t_gr, p_t_gl))               # top gauge
        L.append(g.addCircleArc(p_t_gl, c_t_l, p_t_fl))   # top fillet, left
        L.append(g.addLine(p_t_fl, p_tl))                 # top grip, left
        L.append(g.addLine(p_tl, p_bl))                   # left end face

        loop = g.addCurveLoop(L)
        surf = g.addPlaneSurface([loop])
        g.synchronize()

        # Finer in the gauge and the fillet, coarser in the grips, which carry
        # a near-uniform field.
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size_grip)
        for tag in (p_t_gr, p_t_gl, p_b_gr, p_b_gl, p_t_fr, p_t_fl, p_b_fr, p_b_fl):
            gmsh.model.mesh.setSize([(0, tag)], size_gauge)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tags = tags[order].astype(np.int64)
        coords = xyz.reshape(-1, 3)[order][:, :2]
        t2i = {int(t): i for i, t in enumerate(tags)}

        tris = []
        etypes, _et, enodes = gmsh.model.mesh.getElements(2, surf)
        for et, en in zip(etypes, enodes):
            if int(et) == GMSH_TRI6:
                conn = np.asarray(en, dtype=np.int64).reshape(-1, 6)
                tris.append(np.vectorize(t2i.__getitem__)(conn))
        if not tris:
            raise RuntimeError("no Triangle6 elements produced")
        tris = np.vstack(tris)
    finally:
        gmsh.finalize()

    referenced = np.zeros(coords.shape[0], dtype=bool)
    referenced[tris.ravel()] = True
    if not np.all(referenced):
        keep = np.where(referenced)[0]
        remap = -np.ones(coords.shape[0], dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        coords = coords[keep]
        tris = remap[tris]

    tol = 1.0e-9 * cfg.COUPON_L_TOTAL
    left = np.where(np.abs(coords[:, 0] + p["x_end"]) < tol)[0]
    right = np.where(np.abs(coords[:, 0] - p["x_end"]) < tol)[0]

    geom = dict(p)
    geom.update(n_nodes=int(coords.shape[0]), n_elements=int(tris.shape[0]),
                n_left=int(left.size), n_right=int(right.size),
                area=_area(coords, tris))
    return coords, tris, left, right, geom


def _area(coords, tris):
    a, b, c = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    return float(np.sum(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                     - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))))


def _self_test():
    p = profile()
    print("profile (mm):")
    for k in ("w_gauge", "w_grip", "rise", "dx", "x_gauge", "x_fillet_end", "x_end"):
        print(f"  {k:14s} {p[k] * 1e3:9.4f}")
    print(f"  fillet end slope {p['fillet_end_slope_deg']:.2f} deg "
          f"(a kink, since the standard gives a single R)")

    # Cotes close, against the standard's own independent dimensions.
    ok_dx = abs(p["dx"] - np.sqrt(76.0e-3 ** 2 - 73.0e-3 ** 2)) < 1e-12
    ok_grip = cfg.COUPON_GRIP_SEP / 2.0 > p["x_fillet_end"]

    coords, tris, left, right, geom = build_mesh()

    # Analytic area of the profile: gauge strip + 2 fillet regions + 2 grips.
    R, rise = p["R"], p["rise"]
    half_ang = np.arctan2(p["dx"], R - rise)
    seg = 0.5 * R ** 2 * (half_ang - np.sin(half_ang) * np.cos(half_ang))
    fillet_half = p["dx"] * p["w_grip"] - seg      # under one fillet, one side
    a_gauge = 2.0 * p["w_gauge"] * cfg.COUPON_L_GAUGE
    a_fillet = 4.0 * fillet_half
    a_grip = 2.0 * (p["x_end"] - p["x_fillet_end"]) * 2.0 * p["w_grip"]
    a_exact = a_gauge + a_fillet + a_grip
    a_err = (geom["area"] - a_exact) / a_exact

    # End faces should carry the same node count, and lie at +-LO/2.
    ok_faces = (geom["n_left"] == geom["n_right"]) and geom["n_left"] > 4

    print(f"\nmesh: {geom['n_nodes']} nodes, {geom['n_elements']} elements, "
          f"{geom['n_left']}/{geom['n_right']} end-face nodes")
    print(f"  area {geom['area']:.8e} vs analytic {a_exact:.8e}  "
          f"rel {a_err:+.2e}  {'OK' if abs(a_err) < 2e-3 else 'FAIL'}")
    print(f"  dx from standard dims:      {'OK' if ok_dx else 'FAIL'}")
    print(f"  grips clamp straight region: {'OK' if ok_grip else 'FAIL'}")
    print(f"  end faces matched:           {'OK' if ok_faces else 'FAIL'}")
    ok = abs(a_err) < 2e-3 and ok_dx and ok_grip and ok_faces
    print("COUPON_MESH_SELFTEST_PASS" if ok else "COUPON_MESH_SELFTEST_FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _self_test() else 1)
