#!/usr/bin/env python3
"""The classic fixed-weight ECM's accuracy-vs-cost curve, for both rules.

This is the baseline MAW-ECM has to beat, and it had never been measured at the
point counts MAW-ECM operates at. Everything here is numpy on the stored
integrand: no assembly, no training.

Also verifies `maw_lab.field_weights` (a hand-written numpy forward pass of the
softmax field) against torch, because a closed form derived by hand is exactly
the kind of thing that is 99% right.
"""
from __future__ import annotations

import sys

import numpy as np

import maw_lab as L

RANKS = (4, 9, 14, 19, 29, 39, 72, 120)


def main():
    d = L.load()
    q, ne = d["q"], d["ne"]
    fi, vi = L.split(q.shape[0])
    print(f"states {q.shape[0]}  ({fi.size} fit / {vi.size} held out)  "
          f"n_elements {ne}\n")

    out = {}
    for nm, C in (("res", d["C_res"]), ("sig", d["C_sig"])):
        b = L.targets(C)
        print(f"=== {nm}: classic fixed-weight ECM, {C.shape[2]} physical rows "
              f"+ volume row ===")
        # The ECM support is selected on the FIT states only, so the held-out
        # numbers measure generalization for the classic rule on the same
        # footing as for the trained field.
        for r in RANKS:
            try:
                z, w, sv = L.classic_ecm(C, r, states=fi)
            except Exception as e:                       # noqa: BLE001
                print(f"  rank {r:4d}: ECM failed ({type(e).__name__}: {e})")
                continue
            A = L.blocks(C, z)
            L.report(f"rank {r:4d}", A, b, w, vi)
            out[f"{nm}_r{r}_z"] = z
            out[f"{nm}_r{r}_w"] = w
        tail = np.cumsum(sv[::-1] ** 2)[::-1]
        tot = sv[0] ** 2 if sv.size else 1.0
        e = np.sqrt(np.maximum(tail, 0.0) / np.sum(sv ** 2))
        idx = [i for i in (9, 39, 99, 299) if i < e.size]
        print("  integrand spectrum, relative tail energy: "
              + "  ".join(f"r={i + 1}: {e[i]:.2e}" for i in idx) + "\n")

    np.savez_compressed(L.HERE / "classic_curve.npz", **out)

    # --- verify the numpy field evaluation against torch -------------------
    C = d["C_res"]
    z, w, _ = L.classic_ecm(C, 9, states=fi)
    A, b = L.blocks(C, z), L.targets(C)
    m = L.fit_field(q, A, b, ne, fi[:400], vi[:60], hidden=(16, 16),
                    epochs=40, patience=40)
    err = np.max(np.abs(L.field_weights(m, q) - m["W"])) / np.max(np.abs(m["W"]))
    print(f"numpy field vs torch field: max rel err {err:.3e}")
    print("BASELINE_DONE" if err < 1e-10 else "BASELINE_FIELD_MISMATCH")
    return 0 if err < 1e-10 else 1


if __name__ == "__main__":
    sys.exit(main())
