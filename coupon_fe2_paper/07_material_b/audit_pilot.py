#!/usr/bin/env python3
"""Read-only audit of saved pilot fields; write only a new audit result file.

Contact screening uses straight polygons through the deformed cavity nodes.
This is not a contact model or a certificate for the curved FE boundary.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from geometry import cavity_parameters


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def point_inside(point, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    crosses = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    denominator = np.where(crosses, b[:, 1]-a[:, 1], 1.0)
    x = a[:, 0] + (point[1]-a[:, 1])*(b[:, 0]-a[:, 0])/denominator
    return bool(np.count_nonzero(crosses & (point[0] < x)) % 2)


def polygon_distance(first, second):
    """Segment intersection test and minimum endpoint-to-segment distance."""
    if point_inside(first[0], second) or point_inside(second[0], first):
        return 0.0
    a, b = first[:, None, :], np.roll(first, -1, axis=0)[:, None, :]
    c, d = second[None, :, :], np.roll(second, -1, axis=0)[None, :, :]
    ab, cd = b-a, d-c
    c1, c2 = cross(ab, c-a), cross(ab, d-a)
    c3, c4 = cross(cd, a-c), cross(cd, b-c)
    intersects = (c1*c2 < 0) & (c3*c4 < 0)
    def point_segment(point, start, edge):
        length2 = np.sum(edge**2, axis=-1)
        fraction = np.clip(np.sum((point-start)*edge, axis=-1)/length2, 0, 1)
        return np.linalg.norm(point - (start + fraction[..., None]*edge), axis=-1)
    distance = min(float(point_segment(a, c, cd).min()),
                   float(point_segment(b, c, cd).min()),
                   float(point_segment(c, a, ab).min()),
                   float(point_segment(d, a, ab).min()))
    return 0.0 if intersects.any() else distance


def self_intersection(polygon):
    a, b = polygon[:, None, :], np.roll(polygon, -1, axis=0)[:, None, :]
    c, d = polygon[None, :, :], np.roll(polygon, -1, axis=0)[None, :, :]
    ab, cd = b-a, d-c
    intersects = (cross(ab, c-a)*cross(ab, d-a) < 0) & (cross(cd, a-c)*cross(cd, b-c) < 0)
    ids = np.arange(len(polygon))
    adjacent = (ids[:, None] == ids[None, :]) | (
        (ids[:, None] - ids[None, :]) % len(ids) == 1) | (
        (ids[None, :] - ids[:, None]) % len(ids) == 1)
    return bool((intersects & ~adjacent).any())


def rank_one_screen(strain, stress, tangent):
    """All a via an acoustic-matrix eigenvalue, at 72 sampled unit b directions."""
    e, s, D = np.asarray(strain), np.asarray(stress), np.asarray(tangent)
    C = np.array([[1+2*e[0], e[2]], [e[2], 1+2*e[1]]])
    vals, vectors = np.linalg.eigh(C)
    F = (vectors*np.sqrt(vals)) @ vectors.T
    S = np.array([[s[0], s[2]], [s[2], s[1]]])
    minimum = np.inf
    for angle in np.arange(72)*np.pi/72:
        b = np.array([np.cos(angle), np.sin(angle)])
        edot = []
        for a in np.eye(2):
            H = np.outer(a, b)
            dE = (F.T@H + H.T@F)/2
            edot.append([dE[0, 0], dE[1, 1], 2*dE[0, 1]])
        edot = np.array(edot)
        acoustic = edot@D@edot.T + np.eye(2)*(b@S@b)
        minimum = min(minimum, float(np.linalg.eigvalsh(acoustic).min()))
    return minimum


def audit(directory):
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "pilot complete":
        raise RuntimeError("Wait for the full pilot before auditing saved fields")
    spec = report["spec"]
    L = spec["cell_side"]
    checks, states, rank_one = [], [], []
    spec_hash = hashlib.sha256((directory / "spec.json").read_bytes()).hexdigest()
    checks.append(dict(check="frozen specification hash", passed=spec_hash == report["spec_sha256"]))
    for mesh in report["meshes"]:
        name = mesh["name"]
        for state in mesh.get("states", []):
            if state["ok"]:
                curvature = rank_one_screen(state["strain"], state["stress"], state["tangent"])
                rank_one.append(dict(mesh=name, path=state["path"], fraction=state["fraction"],
                                     sampled_min_rank_one_curvature=curvature))
        mesh_hash = hashlib.sha256((directory / (name + ".mdpa")).read_bytes()).hexdigest()
        checks.append(dict(check=name + ": mesh hash", passed=mesh_hash == mesh["mesh_sha256"]))
        data = np.load(directory / (name + ".npz"))
        xy = data["xy"]
        rims = []
        for hole in cavity_parameters(spec):
            t = np.radians(hole["angle"])
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            local = (xy - hole["center"]) @ R
            level = (local[:, 0]/hole["a"])**2 + (local[:, 1]/hole["b"])**2
            ids = np.where(np.abs(level-1) < 1e-7)[0]
            if len(ids) < 6:
                raise RuntimeError("Cannot recover cavity rim nodes")
            order = np.argsort(np.arctan2(local[ids, 1]/hole["b"], local[ids, 0]/hole["a"]))
            rims.append(ids[order])
        for path in spec["paths"]:
            file = directory / (name + "_" + path + "_endpoint.npz")
            if not file.exists():
                checks.append(dict(check=name + "/" + path + ": saved endpoint", passed=False))
                continue
            data = np.load(file)
            row = dict(mesh=name, path=path)
            if "node_displacement" not in data.files:
                row["note"] = "Original run predates saved node displacement; no polygon/jump audit"
                states.append(row)
                continue
            displacement = data["node_displacement"]
            deformed = xy + displacement
            e = data["strain"]
            C = np.array([[1+2*e[0], e[2]], [e[2], 1+2*e[1]]])
            vals, vectors = np.linalg.eigh(C)
            Fbar = (vectors * np.sqrt(vals)) @ vectors.T
            jump_errors = []
            for axis in (0, 1):
                first = np.where(np.isclose(xy[:, axis], -L/2, atol=1e-8, rtol=0))[0]
                last = np.where(np.isclose(xy[:, axis], L/2, atol=1e-8, rtol=0))[0]
                first = first[np.argsort(xy[first, 1-axis])]
                last = last[np.argsort(xy[last, 1-axis])]
                actual = displacement[last] - displacement[first]
                expected = (xy[last] - xy[first]) @ (Fbar-np.eye(2)).T
                jump_errors.append(float(np.linalg.norm(actual - expected, axis=1).max()))
            polygons = [deformed[ids] for ids in rims]
            minimum_gap = np.inf
            for i, first in enumerate(polygons):
                for j, second in enumerate(polygons):
                    for shift in itertools.product((-1, 0, 1), repeat=2):
                        if i == j and shift == (0, 0):
                            continue
                        translated = second + Fbar @ (L*np.array(shift))
                        minimum_gap = min(minimum_gap, polygon_distance(first, translated))
            row.update(max_periodic_jump_error=max(jump_errors),
                       min_micro_J=float(np.linalg.det(data["F_micro"]).min()),
                       min_cavity_polygon_gap=float(minimum_gap),
                       cavity_polygon_self_intersection=any(self_intersection(p) for p in polygons))
            states.append(row)
            checks.append(dict(check=name + "/" + path + ": saved fields/jumps/polygon screen",
                passed=max(jump_errors) < 1e-8 and row["min_micro_J"] > 0
                and minimum_gap > 1e-6 and not row["cavity_polygon_self_intersection"]))
    return dict(status="saved-field audit complete", checks=checks, states=states,
                rank_one_screen=rank_one,
                rank_one_screen_nonnegative=all(r["sampled_min_rank_one_curvature"] >= 0 for r in rank_one),
                passed=all(c["passed"] for c in checks),
                scope="Numerical endpoint audit. Polygon screening is not contact detection "
                "for the entire curved boundary, nor for intermediate continuation states. "
                "Positive quadrature determinants do not establish positivity everywhere. "
                "Rank-one screening uses the stress-dependent chain rule and 72 b directions; "
                "it is not proof of rank-one convexity or polyconvexity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(args.directory)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Saved-field audit passed:", result["passed"])
    print("Output:", args.out)
    raise SystemExit(0 if result["passed"] else 1)
