"""Direct macro constitutive adapter for the selected flexible PANN models.

The adapter preserves the contract of ``VectorizedAssembler``: engineering
Green strain ``[E11, E22, gamma12]`` in, PK2 stress and its consistent energy
tangent out.  It is deliberately direct: no RVE, POD, ANN decoder, or ECM
assembly is evaluated online.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANN = ROOT / "06_pann"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class FlexiblePANNLaw:
    """Batched physical-units ICNN/ICKAN energy law for macro FE solves."""

    def __init__(self, checkpoint: str | Path):
        # ``flexible_pann`` is intentionally imported lazily.  The FOM/HPROM
        # drivers do not depend on PyTorch, and this adapter should be equally
        # usable for the selected ICNN and (later) the selected C2 ICKAN.
        from flexible_pann import load_flexible, physical_response

        self.checkpoint_path = Path(checkpoint).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.model, self.checkpoint = load_flexible(self.checkpoint_path)
        self._physical_response = physical_response
        self.calls = 0
        self.points = 0
        self.seconds = 0.0

    @property
    def metadata(self) -> dict:
        cfg = self.checkpoint["configuration"]
        return dict(
            checkpoint=str(self.checkpoint_path),
            checkpoint_sha256=sha256(self.checkpoint_path),
            core=cfg["core"],
            features=len(cfg["specs"]),
            widths=list(cfg["widths"]),
            learn_features=bool(cfg.get("learn_features", False)),
            dynamic_center=bool(cfg.get("dynamic_center", False)),
            spline_basis=cfg.get("spline_basis"),
            analytic_stress=bool(cfg.get("analytic_stress", False)),
            strain_scale=float(self.checkpoint["strain_scale"]),
            energy_scale=float(self.checkpoint["energy_scale"]),
        )

    def response(self, E, *, tangent: bool):
        """Physical response, retained for independent adapter checks."""
        E = np.asarray(E, dtype=np.float64).reshape(-1, 3)
        ans = self._physical_response(self.model, self.checkpoint, E, tangent=tangent)
        for name, value in ans.items():
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"non-finite flexible-PANN {name}")
        return ans

    def __call__(self, E_flat, young=None, poisson=None):
        """Return physical ``(S, dS/dE)`` for a macro assembler batch."""
        del young, poisson
        E = np.asarray(E_flat, dtype=np.float64).reshape(-1, 3)
        tic = time.perf_counter()
        ans = self.response(E, tangent=True)
        S, C = ans["stress"], ans["tangent"]
        if S.shape != E.shape or C.shape != (E.shape[0], 3, 3):
            raise RuntimeError("flexible-PANN adapter returned an invalid batch shape")
        self.calls += 1
        self.points += int(E.shape[0])
        self.seconds += time.perf_counter() - tic
        return S, C


def finite_difference_tangent_check(law: FlexiblePANNLaw, states, h: float = 1.0e-6):
    """Return an energy-tangent consistency error on selected physical states."""
    states = np.asarray(states, dtype=np.float64).reshape(-1, 3)
    ref = law.response(states, tangent=True)
    worst = 0.0
    for j in range(3):
        shift = np.zeros_like(states)
        shift[:, j] = h * np.maximum(1.0, np.abs(states[:, j]))
        plus = law.response(states + shift, tangent=False)["stress"]
        minus = law.response(states - shift, tangent=False)["stress"]
        fd = (plus - minus) / (2.0 * shift[:, j, None])
        err = np.linalg.norm(fd - ref["tangent"][:, :, j]) / max(np.linalg.norm(fd), 1.0)
        worst = max(worst, float(err))
    asym = np.linalg.norm(ref["tangent"] - np.swapaxes(ref["tangent"], 1, 2))
    asym /= max(np.linalg.norm(ref["tangent"]), 1.0)
    return dict(tangent_fd_relative=worst, tangent_asymmetry_relative=float(asym))
