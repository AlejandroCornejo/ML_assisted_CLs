#!/usr/bin/env python3
"""The official table: three reduced models, with the FOM as the only reference.

The PROM is not part of the deliverable. It was useful while establishing that
the reduced basis itself was sound, but it hyperreduces nothing -- it assembles
all 1546 elements -- so it belongs to the development record, not to a table
about hyperreduction.

Regenerated from `official_models.py`, so the numbers in the paper and the
models used by stage 06 cannot drift apart. In-envelope and out-of-envelope on
the same states, timed serially -- run this alone.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from official_models import ROOT, OfficialModels

N_EVAL = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--no-fom", action="store_true",
                    help="skip the FOM row (it is 96% of the runtime)")
    a_ = ap.parse_args()

    from _material_law_guard_claude import true_neo_hookean_active

    d = np.load(ROOT / "03_data" / "data.npz")
    with true_neo_hookean_active():
        M = OfficialModels()
        nel = M.elements()

        MODELS = [
            ("FOM", nel["FOM"], M.fom),
            ("HPROM", nel["HPROM"], M.hprom),
            ("MAW-HPROM-ANN", nel["MAW_HPROM_ANN"], M.maw_hprom_ann),
            ("MAW-D-HPROM-ANN", nel["MAW_D_HPROM_ANN"], M.maw_d_hprom_ann),
        ]
        if a_.no_fom:
            MODELS = MODELS[1:]

        sets = {}
        for nm in ("test", "probe"):
            S_s = d[f"S_{nm}"]
            ok = np.isfinite(S_s).all(axis=1)
            Es, Ss = d[f"E_{nm}"][ok], S_s[ok]
            idx = np.random.default_rng(3).choice(
                Es.shape[0], min(a_.n_eval, Es.shape[0]), replace=False)
            sets[nm] = (Es[idx], Ss[idx])

        out = {}
        for lb, ne_, fn in MODELS:
            for nm, (Es, Ss) in sets.items():
                vals, keep, t, nf = [], [], 0.0, 0
                for j in range(Es.shape[0]):
                    try:
                        t0 = time.perf_counter()
                        vals.append(fn(Es[j]))
                        t += time.perf_counter() - t0
                        keep.append(j)
                    except Exception:                          # noqa: BLE001
                        nf += 1
                if not keep:
                    out[(lb, nm)] = dict(t=np.nan, frob=np.nan, n=0, nf=nf)
                    continue
                A, B = np.array(vals), Ss[keep]
                out[(lb, nm)] = dict(
                    t=t / len(keep),
                    frob=float(np.linalg.norm(A - B) / np.linalg.norm(B)),
                    n=len(keep), nf=nf, ne=ne_)
                print(f"  {lb:<18} {nm:>6}  {t / len(keep) * 1e3:9.2f} ms  "
                      f"frob {out[(lb, nm)]['frob']:.4e}  "
                      f"n={len(keep)} fail={nf}", flush=True)

    ref = out.get(("FOM", "test"))
    print("\n=== OFFICIAL TABLE, homogenized stress vs FOM ===\n")
    hdr = (f"{'model':<18} {'elems':>6} {'ms/state':>10} {'speed-up':>10} "
           f"{'in-envelope':>13} {'out-of-env':>13} {'degrad':>8} {'fails':>6}")
    print(hdr)
    print("-" * len(hdr))
    for lb, ne_, _ in MODELS:
        a, p = out[(lb, "test")], out[(lb, "probe")]
        su = f"{ref['t'] / a['t']:9.1f}x" if ref else f"{'-':>10}"
        print(f"{lb:<18} {ne_:>6} {a['t'] * 1e3:>10.2f} {su} "
              f"{a['frob']:>13.4e} {p['frob']:>13.4e} "
              f"{p['frob'] / a['frob']:>7.1f}x {p['nf']:>6d}")

    np.savez_compressed(
        Path(__file__).resolve().parent / "official_table.npz",
        **{f"{lb}_{nm}_{k}": v for (lb, nm), r in out.items()
           for k, v in r.items() if k in ("t", "frob", "n", "nf")})
    print("\nOFFICIAL_TABLE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
