"""Vector method diagrams and a POD diagnostic from the actual r39 artifact."""
from pathlib import Path
import hashlib
import json
import sys
import os
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / ".pydeps"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = HERE / "figures"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def native_diagram(stem):
    """Build PDF/PNG previews from the same native LaTeX figure used in the paper."""
    engine = os.environ.get("TECTONIC") or shutil.which("tectonic")
    fallback = Path("/tmp/coupon-manuscript-tools/tectonic")
    if engine is None and fallback.is_file():
        engine = str(fallback)
    if engine is None:
        raise RuntimeError("Set TECTONIC to a Tectonic executable to build figure previews.")
    FIG.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.setdefault("XDG_CACHE_HOME", "/tmp/coupon-tectonic-cache")
    subprocess.run(
        [engine, "--outdir", str(FIG), f"{stem}_standalone.tex"],
        cwd=FIG, env=env, check=True,
    )
    (FIG / f"{stem}_standalone.pdf").replace(FIG / f"{stem}.pdf")
    subprocess.run(
        ["pdftoppm", "-singlefile", "-r", "220", "-png",
         str(FIG / f"{stem}.pdf"), str(FIG / stem)],
        check=True,
    )


def pod_diagnostic():
    path = ROOT / "04_training/decoder_basis_B_r39.npz"
    z = np.load(path)
    sig = z["sv"]
    n = np.arange(1, 81)
    # Sum the discarded spectrum directly: 1-cumsum loses the small tail.
    tail = np.r_[np.cumsum(sig[::-1]**2)[::-1][1:], 0.] / np.sum(sig**2)
    E, xi = z["E_train"], z["q_M"].T
    error = xi-E
    V, Vbar, A, Phi = (z[k] for k in ("Phi_M", "Phi_S", "A_M", "Phi_ROM"))
    orth = float(np.linalg.norm(V.T @ Vbar))
    projected = Phi @ (Phi.T @ V)
    span_error = float(np.linalg.norm(projected-V)/np.linalg.norm(V))
    # q = V^T Phi Q_POD and xi = T_m Q_POD imply A_m^-1 = Z Sigma.
    coordinate_identity = float(np.linalg.norm(z["T_m"] @ Phi.T @ V @ A-np.eye(3)))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.15), layout="constrained")
    axes[0].semilogy(n, np.sqrt(tail[:80]), color="#244f79", lw=1.7)
    axes[0].axvline(39, ls="--", color="#a8453c", lw=1)
    axes[0].annotate("Retained POD span: 39", (39, np.sqrt(tail[38])),
                     (42, 2e-3), fontsize=8,
                     arrowprops={"arrowstyle": "-", "color": "#a8453c"})
    axes[0].set(xlabel="Number of POD modes", ylabel="Relative snapshot truncation norm",
                title="(a) Compression before the primary / secondary rotation")
    labels = [r"$E_{11}$", r"$E_{22}$", r"$\gamma_{12}$"]
    for j, (label, color) in enumerate(zip(labels, ["#244f79","#a8453c","#458364"])):
        axes[1].scatter(E[::10,j], xi[::10,j], s=5, alpha=.45, label=label, color=color)
    lim = (-.18,.21)
    axes[1].plot(lim,lim, color=".35", lw=.8, ls="--")
    axes[1].set(xlim=lim, ylim=lim, xlabel="Macroscopic engineering strain component",
                ylabel="Corresponding strain-informed coordinate",
                title="(b) Fitted primary coordinate relation")
    axes[1].legend(frameon=False, markerscale=2, loc="upper left", fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.title.set_fontsize(9)
    save(fig, "pod_primary_coordinates")
    return {
        "source": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "number_of_snapshots": int(E.shape[0]), "retained_dimension": int(z["r_ROM"]),
        "primary_dimension": int(V.shape[1]), "secondary_dimension": int(Vbar.shape[1]),
        "relative_truncation_norm_r39": float(np.sqrt(tail[38])),
        "coordinate_relative_array_error": float(np.linalg.norm(error)/np.linalg.norm(E)),
        "coordinate_max_absolute_error": float(np.max(np.abs(error))),
        "orthogonality_V_Vbar": orth, "V_in_original_POD_span_error": span_error,
        "coordinate_transform_identity_error": coordinate_identity,
        "plot_subsampling": "Every tenth training snapshot; metrics use all snapshots.",
        "interpretation": "Compression and fitted-coordinate diagnostics, not an equilibrium or generalization error.",
    }


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    FIG.mkdir(exist_ok=True)
    native_diagram("model_hierarchy")
    native_diagram("cubature_matrices")
    native_diagram("convex_core_architectures")
    result = pod_diagnostic()
    (HERE / "method_figures_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
