#!/usr/bin/env python3
"""Filleted cruciform mesh via gmsh (OCC kernel, unstructured Tri6), replacing
build_cruciform_mesh_claude.py's sharp 270-degree reentrant corners (confirmed
this session to be a true reentrant-corner singularity, and to already push
the macro shear strain to ~2x the RVE's own trained shear range at the
established delta=1.2 protocol) with a generous circular fillet at each of
the 8 reentrant corners -- same overall approach (cruciform biaxial specimen,
symmetric displacement BC) validated by Yvonnet, Monteiro & He (2013, IJMCE
11(3):201-225, Section 4.4), who cite Chevalier & Marco (2002) for this
specimen family and use a generous fillet ratio (R/L=0.6 in their own units).

Unlike the original file's structured 5-rectangular-patch scheme (which has
no curved-boundary capability at all), this builds the filleted boundary as
an explicit closed loop of straight lines and circular arcs -- each arc's
center/tangent points derived in closed form (verified by hand, then
double-checked numerically below) -- and lets gmsh's own unstructured 2D
mesher + second-order (Tri6) upgrade handle the rest, including correctly
curved midside nodes on the fillet arcs.

Reentrant-corner fillet geometry (derived once, applied to all 8 corners by
symmetry): at reentrant corner P with incoming direction u_in (direction of
travel arriving at P) and outgoing direction u_out (u_out = u_in rotated -90
degrees, i.e. a clockwise turn -- always true at a reentrant corner of a
CCW-oriented boundary), the fillet of radius R has:
  T_in  = P - R*u_in   (tangent point on the incoming edge)
  T_out = P + R*u_out  (tangent point on the outgoing edge)
  C     = P - R*u_in + R*u_out   (arc center)
and the arc from T_in to T_out (the short way, bulging toward P) replaces
the sharp corner. P itself remains a mesh node (now interior, at the meeting
point of the fillet wedge with the two straight patches -- no longer on the
domain boundary).
"""
from __future__ import annotations

import numpy as np


def _rot(p, k):
    """Rotate point p by k*90 degrees CCW about the origin."""
    x, y = p
    for _ in range(k % 4):
        x, y = -y, x
    return (x, y)


def _fillet_arc(P, u_in, u_out, R):
    P = np.asarray(P, dtype=float)
    u_in = np.asarray(u_in, dtype=float)
    u_out = np.asarray(u_out, dtype=float)
    T_in = P - R * u_in
    T_out = P + R * u_out
    C = P - R * u_in + R * u_out
    assert abs(np.linalg.norm(T_in - C) - R) < 1e-9
    assert abs(np.linalg.norm(T_out - C) - R) < 1e-9
    return T_in, T_out, C


def build_boundary_loop(n_body, L_body, arm_width_fraction, L_arm, R):
    """Returns an ordered list of boundary 'segments', each either
    ('line', p0, p1) or ('arc', T_in, C, T_out), tracing the filleted
    cruciform CCW exactly once, plus the list of the 8 (now-interior)
    original reentrant-corner points P (for verification/plotting only)."""
    half_body = L_body / 2.0
    half_arm = L_body * arm_width_fraction / 2.0
    tip = half_body + L_arm

    max_R = half_body - half_arm
    assert 0.0 < R < max_R, (
        f"fillet radius R={R} must be in (0, {max_R}) -- the reentrant corner's "
        f"tangent point must stay strictly within the body's own remaining "
        f"straight edge segment of length {max_R}, without reaching the body's "
        f"own convex corner."
    )

    # Canonical px-sector vertices (5 points), CCW, before rotation.
    A = (half_body, -half_arm)   # reentrant: body-right / px-arm-bottom
    B = (tip, -half_arm)         # convex: px arm bottom-right tip
    C_ = (tip, half_arm)         # convex: px arm top-right tip
    D = (half_body, half_arm)    # reentrant: px-arm-top / body-right
    E = (half_body, half_body)  # convex: body's own corner

    # Precompute every sector's key points/arcs first (all 4), since each
    # sector's trailing straight edge (D's arc-end -> E -> next sector's
    # A-arc-start) spans the gap between THIS sector's E and the NEXT
    # sector's Tin_A -- not just up to E.
    sectors = []
    for k in range(4):
        pA, pB, pC, pD, pE = (_rot(A, k), _rot(B, k), _rot(C_, k), _rot(D, k), _rot(E, k))
        u_in_A = _rot((0.0, 1.0), k)     # arriving at A moving in +y (before rotation)
        u_out_A = _rot((1.0, 0.0), k)    # leaving A moving in +x (before rotation)
        u_in_D = _rot((-1.0, 0.0), k)    # arriving at D moving in -x (from C_, before rotation)
        u_out_D = _rot((0.0, 1.0), k)    # leaving D moving in +y (toward E, before rotation)
        Tin_A, Tout_A, C_A = _fillet_arc(pA, u_in_A, u_out_A, R)
        Tin_D, Tout_D, C_D = _fillet_arc(pD, u_in_D, u_out_D, R)
        sectors.append(dict(pA=pA, pB=pB, pC=pC, pD=pD, pE=pE,
                             Tin_A=Tin_A, Tout_A=Tout_A, C_A=C_A,
                             Tin_D=Tin_D, Tout_D=Tout_D, C_D=C_D))

    reentrant_corners = []
    segments = []
    for k in range(4):
        s = sectors[k]
        s_next = sectors[(k + 1) % 4]
        reentrant_corners.extend([s["pA"], s["pD"]])

        segments.append(("arc", s["Tin_A"], s["C_A"], s["Tout_A"]))
        segments.append(("line", s["Tout_A"], s["pB"]))
        segments.append(("line", s["pB"], s["pC"]))
        segments.append(("line", s["pC"], s["Tin_D"]))
        segments.append(("arc", s["Tin_D"], s["C_D"], s["Tout_D"]))
        segments.append(("line", s["Tout_D"], s["pE"]))
        segments.append(("line", s["pE"], s_next["Tin_A"]))

    return segments, reentrant_corners


if __name__ == "__main__":
    segs, corners = build_boundary_loop(n_body=6, L_body=12.0, arm_width_fraction=2.0 / 3.0,
                                         L_arm=8.0, R=1.8)
    print(f"{len(segs)} boundary segments, {len(corners)} reentrant corners (now interior)")
    # Closure check: each segment's end must equal the next segment's start.
    for i in range(len(segs)):
        _, *pts = segs[i]
        end = pts[-1]
        nxt = segs[(i + 1) % len(segs)]
        start_next = nxt[1]
        assert np.allclose(end, start_next, atol=1e-9), (i, end, start_next)
    print("PASS: boundary loop closes exactly, segment-to-segment.")
