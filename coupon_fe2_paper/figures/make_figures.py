#!/usr/bin/env python3
"""Visual verification of stages 00-02.

Not decoration. Each panel checks something the numeric tables cannot:

  * the rotated ellipse is actually at +30 deg and not mirrored, and the
    periodic faces really carry matching node positions
  * the coupon profile really looks like a D638 coupon, fillet tangent to the
    gauge and meeting the grip at the 16.15 deg kink the cotes predict
  * WHERE the shear concentrates -- claimed to be the fillet region from the
    numbers alone, never looked at
  * the strain cloud really is a cone, and the sampling grid really covers it
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(ROOT / "00_rve"), str(ROOT / "01_macro_prepass"),
          str(ROOT / "02_sampling"), str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import config as cfg  # noqa: E402


def tri_patches(coords, tris):
    """T6 elements drawn via their 3 corner nodes."""
    import matplotlib.tri as mtri
    return mtri.Triangulation(coords[:, 0], coords[:, 1], tris[:, :3])


def fig_rve():
    from gen_rve_mesh import build_mesh
    coords, tris, outer, geom = build_mesh(periodic=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))

    t = tri_patches(coords, tris)
    ax[0].triplot(t, lw=0.35, color="0.45")
    ax[0].plot(coords[outer, 0], coords[outer, 1], ".", ms=3, color="tab:red",
               label=f"outer boundary ({outer.size} nodes)")
    # the ellipse as specified, to check the mesh hole matches it
    a, b = cfg.ellipse_semi_axes()
    th = np.linspace(0, 2 * np.pi, 400)
    rot = np.radians(cfg.ELLIPSE_ANGLE_DEG)
    ex = a * np.cos(th) * np.cos(rot) - b * np.sin(th) * np.sin(rot)
    ey = a * np.cos(th) * np.sin(rot) + b * np.sin(th) * np.cos(rot)
    ax[0].plot(ex, ey, "-", lw=2.0, color="tab:blue",
               label=f"specified ellipse, {cfg.ELLIPSE_ANGLE_DEG:g} deg")
    # major axis, to make the orientation unmistakable
    ax[0].plot([-a * np.cos(rot), a * np.cos(rot)],
               [-a * np.sin(rot), a * np.sin(rot)], "--", lw=1.4, color="tab:blue")
    ax[0].set_aspect("equal")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"RVE cell: {geom['n_elements']} elements, "
                    f"porosity {geom['porosity_mesh'] * 100:.2f}%")

    # periodic face matching: plot the y of left-face nodes against the y of
    # right-face nodes, sorted. A perfect match is the identity line.
    half = geom["block_side"] / 2.0
    tol = 1e-7
    L = coords[np.abs(coords[:, 0] + half) < tol, 1]
    R = coords[np.abs(coords[:, 0] - half) < tol, 1]
    B = coords[np.abs(coords[:, 1] + half) < tol, 0]
    T = coords[np.abs(coords[:, 1] - half) < tol, 0]
    ax[1].plot(np.sort(L), np.sort(R), "o", ms=4, label=f"left vs right ({L.size})")
    ax[1].plot(np.sort(B), np.sort(T), "s", ms=4, mfc="none",
               label=f"bottom vs top ({B.size})")
    lim = [-half, half]
    ax[1].plot(lim, lim, "-", lw=1, color="0.3")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("master-face coordinate")
    ax[1].set_ylabel("slave-face coordinate")
    ax[1].legend(fontsize=8)
    ax[1].set_title("periodic face matching (must lie on the diagonal)")
    fig.tight_layout()
    fig.savefig(HERE / "fig1_rve_cell.png", dpi=140)
    plt.close(fig)
    return f"fig1: {geom['n_elements']} elems, faces {L.size}/{R.size}, {B.size}/{T.size}"


def fig_coupon():
    from gen_coupon_mesh import build_mesh, profile
    coords, tris, left, right, geom = build_mesh()
    p = profile()
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.6))
    t = tri_patches(coords, tris)
    ax[0].triplot(t, lw=0.25, color="0.5")
    ax[0].plot(coords[left, 0], coords[left, 1], ".", ms=5, color="tab:red")
    ax[0].plot(coords[right, 0], coords[right, 1], ".", ms=5, color="tab:red")
    for xv, lbl in ((p["x_gauge"], "gauge end"), (p["x_fillet_end"], "fillet end"),
                    (cfg.COUPON_GRIP_SEP / 2.0, "grip line D/2")):
        for sgn in (-1, 1):
            ax[0].axvline(sgn * xv, ls=":", lw=0.9, color="tab:blue")
        ax[0].text(xv, p["w_grip"] * 1.15, lbl, fontsize=7, ha="center",
                   color="tab:blue")
    ax[0].set_aspect("equal")
    ax[0].set_title(f"ASTM D638 Type I profile: {geom['n_elements']} elements "
                    f"(grips clamp at D/2 = 57.5 mm, outside the fillet end at "
                    f"{p['x_fillet_end'] * 1e3:.2f} mm)")

    # zoom on one fillet, to see tangency at the gauge and the kink at the grip
    ax[1].triplot(t, lw=0.4, color="0.5")
    ax[1].set_xlim(p["x_gauge"] * 0.92, p["x_fillet_end"] * 1.14)
    ax[1].set_ylim(p["w_gauge"] * 0.80, p["w_grip"] * 1.06)
    ax[1].axhline(p["w_gauge"], ls="--", lw=1.1, color="tab:green",
                  label="gauge half-width (fillet is tangent here)")
    ax[1].axhline(p["w_grip"], ls="--", lw=1.1, color="tab:orange",
                  label=f"grip half-width (kink, {p['fillet_end_slope_deg']:.2f} deg)")
    ax[1].set_aspect("equal")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_title("fillet detail")
    fig.tight_layout()
    fig.savefig(HERE / "fig2_coupon.png", dpi=140)
    plt.close(fig)
    return f"fig2: {geom['n_elements']} elems, kink {p['fillet_end_slope_deg']:.2f} deg"


def fig_macro_fields():
    """Where does the shear actually concentrate? Claimed, never looked at."""
    from gen_coupon_mesh import build_mesh
    from macro_prepass import (MacroCoupon, load_C0, svk_material,
                               write_coupon_mdpa)
    C0 = load_C0()
    coords, tris, left, right, geom = build_mesh()
    base = ROOT / "01_macro_prepass" / "coupon_mesh"
    write_coupon_mdpa(str(base) + ".mdpa", coords, tris,
                      np.concatenate((left, right)))
    with svk_material(C0):
        m = MacroCoupon(base)
        m.solve(20.0166e-3, n_steps=8, record=False)
        E = m.assembler._E_voigt.reshape(-1, 3)
        conn = np.asarray(m.assembler.connectivity, dtype=np.int64)
        xc = m.xy[conn[:, :3], 0].mean(axis=1)
        yc = m.xy[conn[:, :3], 1].mean(axis=1)
    ng = m.assembler.n_gauss
    xg, yg = np.repeat(xc, ng), np.repeat(yc, ng)

    fig, ax = plt.subplots(3, 1, figsize=(12, 7.2))
    for k, (j, nm) in enumerate(((0, "E11"), (1, "E22"), (2, "gamma12"))):
        v = E[:, j]
        sc = ax[k].scatter(xg * 1e3, yg * 1e3, c=v, s=5, cmap="RdBu_r",
                           vmin=-np.abs(v).max(), vmax=np.abs(v).max())
        plt.colorbar(sc, ax=ax[k], fraction=0.02, pad=0.01)
        i = int(np.argmax(np.abs(v)))
        ax[k].plot(xg[i] * 1e3, yg[i] * 1e3, "k*", ms=13)
        ax[k].set_aspect("equal")
        ax[k].set_title(f"{nm}: range [{v.min():+.4f}, {v.max():+.4f}], "
                        f"extremum (star) at x = {xg[i] * 1e3:+.1f} mm, "
                        f"y = {yg[i] * 1e3:+.1f} mm", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig3_macro_fields.png", dpi=140)
    plt.close(fig)
    i = int(np.argmax(np.abs(E[:, 2])))
    return (f"fig3: max|g12| at x={xg[i] * 1e3:+.2f}mm y={yg[i] * 1e3:+.2f}mm "
            f"(gauge ends at +-{cfg.COUPON_L_GAUGE / 2 * 1e3:.1f}, "
            f"fillet ends at +-49.64)")


def fig_cloud_grid():
    """Two figures, deliberately separate.

    fig4 shows the DESIGN LOGIC: the box was derived from the measured cloud,
    so the cloud and the box belong together and nothing else should compete
    for attention.

    fig5 shows the SAMPLING SETS: training, test and probe. The cloud is left
    out because at this scale it is a thin line that visually dominates and
    hides the three sets that the figure is actually about.
    """
    cloud = np.load(ROOT / "01_macro_prepass" / "prepass_cloud_svk.npz")["cloud"]
    g = np.load(ROOT / "02_sampling" / "train_grid.npz")
    ev = np.load(ROOT / "02_sampling" / "eval_sets.npz")
    train, test, probe = g["E"], ev["test"], ev["probe"]
    blo, bhi = g["blo"], g["bhi"]
    pairs = ((0, 1, "E11", "E22"), (0, 2, "E11", "gamma12"), (1, 2, "E22", "gamma12"))

    def box(axk, i, j, **kw):
        axk.add_patch(plt.Rectangle((blo[i], blo[j]), bhi[i] - blo[i],
                                    bhi[j] - blo[j], fill=False, lw=1.8,
                                    ec="k", **kw))

    # ---- fig4: cloud and box only ----
    sub = cloud[::13]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for k, (i, j, nx, ny) in enumerate(pairs):
        ax[k].plot(sub[:, i], sub[:, j], ".", ms=2.2, color="tab:blue",
                   label="strain states the coupon visits")
        box(ax[k], i, j, label="training box: cloud + 40% margin")
        ax[k].set_xlabel(nx)
        ax[k].set_ylabel(ny)
        if k == 0:
            ax[k].legend(fontsize=8, loc="lower left")
    fig.suptitle("Where the box comes from: the measured pre-pass cloud, widened by the margin",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "fig4_cloud_and_box.png", dpi=140)
    plt.close(fig)

    # ---- fig5: the three sampling sets, no cloud ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for k, (i, j, nx, ny) in enumerate(pairs):
        ax[k].plot(train[:, i], train[:, j], ".", ms=1.6, color="0.62",
                   label=f"training ({train.shape[0]})")
        ax[k].plot(test[:, i], test[:, j], ".", ms=4, color="tab:green",
                   label=f"test, inside ({test.shape[0]})")
        ax[k].plot(probe[:, i], probe[:, j], ".", ms=4, color="tab:red",
                   label=f"probe, outside ({probe.shape[0]})")
        box(ax[k], i, j)
        ax[k].set_xlabel(nx)
        ax[k].set_ylabel(ny)
        if k == 0:
            ax[k].legend(fontsize=8, loc="lower left")
    fig.suptitle("The three sampling sets (cloud omitted: at this scale it hides them)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "fig5_sampling_sets.png", dpi=140)
    plt.close(fig)
    return "fig4: cloud + box    fig5: train / test / probe"


if __name__ == "__main__":
    from _material_law_guard_claude import true_neo_hookean_active  # noqa: F401
    import os as _os
    _only = _os.environ.get("ONLY_FIGS")
    _all = dict(rve=fig_rve, coupon=fig_coupon, fields=fig_macro_fields, cloud=fig_cloud_grid)
    _sel = [_all[k] for k in (_only.split(",") if _only else _all)]
    for f in _sel:
        print(f(), flush=True)
    print("FIGURES_DONE")
