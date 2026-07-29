#!/usr/bin/env python3
"""Construct the local anisotropic-to-isotropic tangent map used by PANN-T.

The construction adapts the closest-isotropic map of Rossi, Zorrilla and
Codina to the engineering-strain coordinates of this RVE:

    e = [E11, E22, gamma12],  gamma12=2 E12,
    s = [S11, S22, S12],       D0 = ds/de at e=0.

For an isotropic tangent in these work-conjugate coordinates,

             [lambda+2mu, lambda,     0]
    Dhat =  [lambda,     lambda+2mu,   0].
             [0,              0,     mu]

We preserve the uniform-dilation modulus kappa=(D00+2D01+D11)/4 and minimise
the Euclidean/Frobenius difference to D0.  This gives

    mu=(D00-2D01+D11+D22)/5,  lambda=kappa-mu.

For SPD D0 and Dhat, T=Dhat^{-1/2}D0^{1/2} then satisfies exactly
``D0 = T.T Dhat T``.  T is a local change of coordinates, not a claim that
the nonlinear RVE is isotropic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "fom_tangent_stability_test" / "results" / "h_1.000e-02" / "fom_energy_hessian_audit.npz"
OUTPUT = HERE / "data" / "local_isotropic_metric.json"


def symmetric_square_root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if np.min(values) <= 0.0:
        raise ValueError(f"The mapping requires a positive-definite tangent; eigenvalues are {values}.")
    return (vectors * np.sqrt(values)) @ vectors.T


def symmetric_inverse_square_root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if np.min(values) <= 0.0:
        raise ValueError(f"The mapping requires a positive-definite tangent; eigenvalues are {values}.")
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def main() -> None:
    with np.load(AUDIT) as audit:
        names = np.asarray(audit["names"]).astype(str)
        reference = int(np.flatnonzero(names == "reference")[0])
        d0 = np.asarray(audit["tangent_fd"][reference], dtype=float)
    d0 = 0.5 * (d0 + d0.T)
    d00, d01, d11, d22 = d0[0, 0], d0[0, 1], d0[1, 1], d0[2, 2]
    kappa = (d00 + 2.0 * d01 + d11) / 4.0
    mu = (d00 - 2.0 * d01 + d11 + d22) / 5.0
    lame_lambda = kappa - mu
    dhat = np.array(
        ((lame_lambda + 2.0 * mu, lame_lambda, 0.0),
         (lame_lambda, lame_lambda + 2.0 * mu, 0.0),
         (0.0, 0.0, mu)),
        dtype=float,
    )
    transform = symmetric_inverse_square_root(dhat) @ symmetric_square_root(d0)
    reconstruction = transform.T @ dhat @ transform
    payload = {
        "source": {
            "fom_tangent_audit": str(AUDIT.relative_to(HERE.parent)),
            "state": "reference configuration",
            "finite_difference_step": 1.0e-2,
            "coordinates": "e=[E11,E22,gamma12], gamma12=2E12; s=[S11,S22,S12]",
        },
        "closest_isotropic_projection": {
            "formula": "kappa=(D00+2D01+D11)/4; mu=(D00-2D01+D11+D22)/5; lambda=kappa-mu",
            "kappa": float(kappa),
            "mu": float(mu),
            "lambda": float(lame_lambda),
            "D_hat": dhat.tolist(),
        },
        "D0": d0.tolist(),
        "metric_transform_T": transform.tolist(),
        "identity": "D0 = T^T D_hat T",
        "checks": {
            "D0_eigenvalues": np.linalg.eigvalsh(d0).tolist(),
            "D_hat_eigenvalues": np.linalg.eigvalsh(dhat).tolist(),
            "relative_reconstruction_error": float(np.linalg.norm(reconstruction - d0) / np.linalg.norm(d0)),
            "relative_distance_of_D0_to_D_hat": float(np.linalg.norm(d0 - dhat) / np.linalg.norm(d0)),
            "relative_distance_of_T_to_identity": float(np.linalg.norm(transform - np.eye(3)) / np.linalg.norm(np.eye(3))),
        },
        "interpretation": "T is a fixed local tangent preconditioner. It neither imposes material isotropy nor proves global nonlinear stability or polyconvexity.",
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
