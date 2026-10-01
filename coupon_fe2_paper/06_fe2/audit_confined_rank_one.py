#!/usr/bin/env python3
"""Rank-one curvature at the Gauss points of the confined-compression runs (declared post-hoc diagnostic).

The six incremental runs of ``results/confined_compression_v1`` are repeated with ``--save-fields``; every
recorded outcome must be unchanged.  At each converged increment the finite-strain curvature of Eq. (lh) of
the supplement, D^2_F W[a x b, a x b], is evaluated with each energy's own stress and tangent at the 384 Gauss
points of the block and minimized over unit directions a, b.  Negative minima are confirmed by second
differences of W(F +- hH).  The frozen outcomes of the campaign are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAMPAIGN = HERE / "results/confined_compression_v1"
OUT = CAMPAIGN / "rank_one"
sys.path[:0] = [str(HERE), str(ROOT / "06_pann")]

# Three-point rule of the quadratic triangles; the reconstructed minimum J must match the runner's record.
GAUSS = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
RECORD_KEYS = ("step", "pressure_Pa", "iterations", "mean_stretch", "min_J")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape_gradients(xi: float, eta: float) -> np.ndarray:
    """Derivatives of the six quadratic shape functions (corner nodes, then mid-edges 01, 12, 20)."""
    l1, l2, l3 = 1 - xi - eta, xi, eta
    return np.array([[-(4 * l1 - 1), -(4 * l1 - 1)], [4 * l2 - 1, 0], [0, 4 * l3 - 1],
                     [4 * (l1 - l2), -4 * l2], [4 * l3, 4 * l2], [-4 * l3, 4 * (l1 - l3)]])


def gauss_deformation_gradients(coords, triangles, eq_map, u):
    nodal = np.column_stack((u[eq_map[:, 0]], u[eq_map[:, 1]]))
    F = []
    for element in triangles:
        X, U = coords[element], nodal[element]
        for xi, eta in GAUSS:
            dN = shape_gradients(xi, eta)
            grad = dN @ np.linalg.inv(X.T @ dN)
            F.append(np.eye(2) + U.T @ grad)
    return np.asarray(F)


def green(F):
    C = np.einsum("nki,nkj->nij", F, F)
    return np.column_stack((0.5 * (C[:, 0, 0] - 1), 0.5 * (C[:, 1, 1] - 1), C[:, 0, 1])), np.sqrt(np.linalg.det(C))


def rank_one_minimum(law, F, n_angles):
    """Minimum over unit a, b of Eq. (lh) at every Gauss point, with the minimizing directions."""
    E, _ = green(F)
    ans = law.response(E, tangent=True)
    S, D = ans["stress"], ans["tangent"]
    S_matrix = np.stack((np.stack((S[:, 0], S[:, 2]), -1), np.stack((S[:, 2], S[:, 1]), -1)), 1)
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    units = np.column_stack((np.cos(angles), np.sin(angles)))
    best = np.full(len(F), np.inf)
    arg = np.zeros((len(F), 2), dtype=int)
    for i, a in enumerate(units):
        for j, b in enumerate(units):
            FH = np.einsum("nki,kj->nij", F, np.outer(a, b))
            dE = 0.5 * (FH + FH.transpose(0, 2, 1))
            de = np.column_stack((dE[:, 0, 0], dE[:, 1, 1], 2 * dE[:, 0, 1]))
            value = np.einsum("ni,nij,nj->n", de, D, de) + np.einsum("i,nij,j->n", b, S_matrix, b)
            better = value < best
            best[better], arg[better] = value[better], (i, j)
    return best, units[arg[:, 0]], units[arg[:, 1]]


def second_difference(law, F, a, b, h=1.0e-4):
    H = np.outer(a, b)
    energy = [law.response(green((F + s * h * H)[None])[0], tangent=False)["energy"][0] for s in (1, 0, -1)]
    return (energy[0] - 2 * energy[1] + energy[2]) / h**2


def rerun(tag, model, direction):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    command = [sys.executable, str(HERE / "run_pann_confined_block.py"), "--tier", model["tier"],
               "--checkpoint", model["checkpoint"], "--direction", direction, "--pressure", "5e8",
               "--n-steps", "20", "--threads", "2", "--output-dir", str(OUT), "--save-fields", "--tag", tag]
    with open(OUT / f"{tag}.log", "w", encoding="utf-8") as log:
        subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)


def same_outcome(frozen, repeated):
    if (frozen["status"], frozen["n_steps_completed"]) != (repeated["status"], repeated["n_steps_completed"]):
        return False
    return all(all(a[k] == b[k] for k in RECORD_KEYS) for a, b in zip(frozen["steps"], repeated["steps"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", type=int, default=90, help="Unit directions per half circle for a and b.")
    parser.add_argument("--reuse", action="store_true", help="Reuse repeated runs already in rank_one/.")
    args = parser.parse_args()
    from flexible_pann_law import FlexiblePANNLaw
    from free_pann_law import CouponFreePANNLaw

    OUT.mkdir(exist_ok=True)
    rule = json.loads((CAMPAIGN / "rule.json").read_text())
    cases, identical = {}, {}
    for name, model in rule["models"].items():
        assert sha256(model["checkpoint"]) == model["checkpoint_sha256"], name
        law = (CouponFreePANNLaw if model["tier"] == "free" else FlexiblePANNLaw)(model["checkpoint"])
        for direction in ("x", "y"):
            tag = f"{name}_{direction}_incremental_500"
            if not (args.reuse and (OUT / f"{tag}.npz").is_file()):
                rerun(tag, model, direction)
            frozen = json.loads((CAMPAIGN / f"{tag}.json").read_text())
            repeated = json.loads((OUT / f"{tag}.json").read_text())
            identical[tag] = same_outcome(frozen, repeated)
            fields = np.load(OUT / f"{tag}.npz")
            triangles = fields["triangles"].astype(int)
            centroids = 1e3 * fields["coords"][triangles[:, :3]].mean(axis=1)
            steps = []
            for record, u in zip(frozen["steps"], fields["u"]):
                F = gauss_deformation_gradients(fields["coords"], triangles, fields["eq_map"], u)
                _, J = green(F)
                assert abs(J.min() - record["min_J"]) < 1e-9, (tag, record["step"])
                q, a, b = rank_one_minimum(law, F, args.angles)
                k = int(np.argmin(q))
                negative = np.flatnonzero(q < 0)
                steps.append(dict(
                    step=record["step"], pressure_MPa=record["pressure_Pa"] / 1e6,
                    shortening=1 - record["mean_stretch"], min_J=record["min_J"],
                    min_curvature_MPa=float(q[k] / 1e6), negative_points=int(negative.size),
                    min_point_J=float(J[k]), min_point_centroid_mm=centroids[k // 3].round(3).tolist(),
                    second_difference_MPa=[float(second_difference(law, F[g], a[g], b[g]) / 1e6) for g in negative]))
            cases[tag] = dict(model=name, direction=direction, status=frozen["status"],
                              n_steps_completed=frozen["n_steps_completed"], gauss_points=int(len(q)), steps=steps)
    constrained = [s["min_curvature_MPa"] for c in cases.values() if c["model"] != "Unconstrained" for s in c["steps"]]
    report = dict(
        status="complete" if all(identical.values()) else "outcome_mismatch",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        purpose="Post-hoc diagnostic: rank-one curvature (Eq. lh) at the Gauss points of each converged increment.",
        runner_sha256=sha256(HERE / "run_pann_confined_block.py"), script_sha256=sha256(__file__),
        directions_per_half_circle=args.angles, identical_records=identical,
        constrained_minimum_MPa=float(min(constrained)), cases=cases)
    (OUT / "rank_one_audit.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], constrained_minimum_MPa=report["constrained_minimum_MPa"])))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
