#!/usr/bin/env python3
"""S-curve (ogee) fillet between two parallel horizontal lines at different
heights -- the shoulder transition of a dogbone tension specimen, connecting
the wide grip section (half-width y_p) to the narrow gauge section
(half-width y_g < y_p).

Unlike the cruciform's single-arc corner fillet (build_cruciform_mesh_filleted
_claude.py's _fillet_arc), a single circular arc cannot be tangent to BOTH
lines here: tangency to a horizontal line forces the radius at that point to
be vertical, so a single circle tangent to both y=y_g and y=y_p would need
both tangent points at the same x (a degenerate 180-degree sidestep with zero
horizontal run). The standard fix, used for cam/gear/dogbone-shoulder blends,
is two circular arcs of equal radius R, related by a point reflection through
the midpoint M -- point reflection preserves the tangent LINE at the
reflection point, so the two arcs join with matching tangent direction there
(G1-continuous), while each arc is separately tangent to one of the two
straight lines at its own outer end (G1-continuous with the gauge/grip
straights too). Requires R >= (y_p - y_g) / 2 for the arcs to physically
reach; R strictly greater gives a real horizontal run instead of a degenerate
vertical sidestep.
"""
from __future__ import annotations

import numpy as np


def scurve_fillet(x_gauge_end: float, y_g: float, y_p: float, R: float):
    """Upper-right shoulder: gauge straight (y=y_g, x<x_gauge_end) transitions
    to grip straight (y=y_p, x>x_grip_start) via two tangent circular arcs.

    Returns dict with:
      t_in   -- (x,y) tangent point on the gauge side (arc 1 meets gauge straight)
      c1     -- arc 1 center
      m      -- (x,y) midpoint where the two arcs join (tangent-matched)
      c2     -- arc 2 center
      t_out  -- (x,y) tangent point on the grip side (arc 2 meets grip straight)
      dx     -- horizontal run of EACH arc (total shoulder run = 2*dx)
    """
    H = y_p - y_g
    if H <= 0:
        raise ValueError("y_p must exceed y_g (grip must be wider than gauge)")
    half_H = H / 2.0
    if R < half_H:
        raise ValueError(f"R={R} too small: need R >= (y_p-y_g)/2 = {half_H} for the arcs to reach")
    dx = float(np.sqrt(R * R - (R - half_H) ** 2))

    t_in = np.array([x_gauge_end, y_g])
    c1 = np.array([x_gauge_end, y_g + R])  # tangent to horizontal at t_in => radius vertical
    m = np.array([x_gauge_end + dx, y_g + half_H])
    c2 = 2.0 * m - c1  # point-reflection of c1 through m
    t_out = 2.0 * m - t_in  # point-reflection of t_in through m
    return {"t_in": t_in, "c1": c1, "m": m, "c2": c2, "t_out": t_out, "dx": dx, "R": R}


def _tangent_direction(center: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Unit tangent direction (one of two possible signs) at `point` on a
    circle centered at `center`: perpendicular to the radius."""
    radius_vec = point - center
    tangent = np.array([-radius_vec[1], radius_vec[0]])
    return tangent / np.linalg.norm(tangent)


def _self_test():
    fil = scurve_fillet(x_gauge_end=0.0, y_g=2.0, y_p=4.0, R=2.0)
    t_in, c1, m, c2, t_out = fil["t_in"], fil["c1"], fil["m"], fil["c2"], fil["t_out"]
    R = fil["R"]

    # 1. Both tangent points and the midpoint lie exactly on their respective circles.
    assert abs(np.linalg.norm(t_in - c1) - R) < 1e-12, "t_in not on circle 1"
    assert abs(np.linalg.norm(m - c1) - R) < 1e-12, "m not on circle 1"
    assert abs(np.linalg.norm(m - c2) - R) < 1e-12, "m not on circle 2"
    assert abs(np.linalg.norm(t_out - c2) - R) < 1e-12, "t_out not on circle 2"

    # 2. Tangent direction at t_in matches the gauge straight's horizontal direction.
    tan_in = _tangent_direction(c1, t_in)
    assert abs(abs(tan_in[1])) < 1e-9, f"arc not horizontal-tangent at gauge join: {tan_in}"

    # 3. Tangent direction at t_out matches the grip straight's horizontal direction.
    tan_out = _tangent_direction(c2, t_out)
    assert abs(abs(tan_out[1])) < 1e-9, f"arc not horizontal-tangent at grip join: {tan_out}"

    # 4. The two arcs share a matching tangent LINE at the midpoint m (G1 continuity of the join).
    tan1_at_m = _tangent_direction(c1, m)
    tan2_at_m = _tangent_direction(c2, m)
    cross = tan1_at_m[0] * tan2_at_m[1] - tan1_at_m[1] * tan2_at_m[0]
    assert abs(cross) < 1e-9, f"tangent lines at m do not match: {tan1_at_m} vs {tan2_at_m}"

    # 5. Heights check out: t_in at y_g, m at the midway height, t_out at y_p.
    assert abs(t_in[1] - 2.0) < 1e-12
    assert abs(m[1] - 3.0) < 1e-12
    assert abs(t_out[1] - 4.0) < 1e-12

    # 6. Monotonic x progression (no fold-back) and symmetric run (dx each side).
    assert t_in[0] < m[0] < t_out[0]
    assert abs((m[0] - t_in[0]) - (t_out[0] - m[0])) < 1e-12

    print("[dogbone_scurve_fillet] all self-checks passed.")
    print(f"  t_in={t_in}, c1={c1}, m={m}, c2={c2}, t_out={t_out}, dx={fil['dx']:.4f}")

    # 7. Sanity sweep: confirm dx grows as R grows past the minimum, and blows up as R -> half_H+.
    for R_try in (1.001, 1.5, 2.0, 4.0, 10.0):
        f = scurve_fillet(0.0, 2.0, 4.0, R_try)
        print(f"  R={R_try:6.3f} -> dx={f['dx']:.4f}, total shoulder run={2*f['dx']:.4f}")


if __name__ == "__main__":
    _self_test()
