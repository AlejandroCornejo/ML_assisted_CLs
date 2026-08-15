"""Shared guard against a real monkeypatch-collision bug: core/fom_solver_rve.py's
_neo_hookean_pk2_2d_vectorized is a single module-level name shared by EVERY
VectorizedAssembler instance in the process. When Cook.gid's driver monkeypatches
it (to route the *macro* material call to a PANN or an RVE surrogate), that same
monkeypatch also silently hijacks the RVE surrogate's OWN internal VectorizedAssembler
(used to solve/evaluate the RVE's own microscale mesh) -- which needs the RVE's TRUE
Neo-Hookean law, not a recursive call back into the surrogate.

Fix: capture the true original function at import time (before any outer script has
a chance to monkeypatch it), and temporarily restore it around any block that touches
an RVE-internal VectorizedAssembler.
"""
from __future__ import annotations

import contextlib

import fom_solver_rve as _fom_module

_TRUE_NEO_HOOKEAN = _fom_module._neo_hookean_pk2_2d_vectorized


@contextlib.contextmanager
def true_neo_hookean_active():
    saved = _fom_module._neo_hookean_pk2_2d_vectorized
    _fom_module._neo_hookean_pk2_2d_vectorized = _TRUE_NEO_HOOKEAN
    try:
        yield
    finally:
        _fom_module._neo_hookean_pk2_2d_vectorized = saved
