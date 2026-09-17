"""Plot approved material-B FOM paths against their reference linearization.

No solver, neural checkpoint or statistical test labels are loaded. Stresses
are work-conjugate second-Piola components, not Cauchy or equivalent stresses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .prepare_design import digest
except ImportError:
    from prepare_design import digest


def linearization(strain, reference_stress, reference_energy, reference_tangent):
    e = np.asarray(strain)
    s0, d0 = np.asarray(reference_stress), np.asarray(reference_tangent)
    return (s0 + e @ d0.T,
            reference_energy + e @ s0 + .5*np.einsum("...i,ij,...j->...", e, d0, e))


def path_metrics(strain, stress, tangent, reference_stress, reference_energy,
                 reference_tangent):
    """Exclude the common reference from relative metrics; reject zero norms."""
    linear_stress, _ = linearization(strain, reference_stress, reference_energy,
                                     reference_tangent)
    stress_norm = np.linalg.norm(stress)
    endpoint_norm = np.linalg.norm(stress[-1])
    tangent_norm = np.linalg.norm(reference_tangent)
    if min(stress_norm, endpoint_norm, tangent_norm) <= 0:
        raise ValueError("Relative path diagnostic has a zero denominator")
    return dict(
        stress_path_relative_L2=float(np.linalg.norm(linear_stress-stress)/stress_norm),
        stress_endpoint_relative_deviation=float(np.linalg.norm(
            linear_stress[-1]-stress[-1])/endpoint_norm),
        tangent_endpoint_relative_change=float(np.linalg.norm(
            tangent[-1]-reference_tangent)/tangent_norm))


def verify_model_lock(spec: dict, lock: dict) -> None:
    expected = {(model, seed) for model in spec["models"]["primary"]
                for seed in spec["models"]["initialization_seeds"]}
    checkpoints = lock.get("checkpoints", [])
    actual = {(row["model"], row["seed"]) for row in checkpoints}
    if lock.get("status") != "locked" or actual != expected or len(checkpoints) != len(expected):
        raise RuntimeError("Every predeclared model/seed must be locked before reading path labels")
    for row in checkpoints:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("Locked checkpoint hash changed: " + row["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assembly-report", type=Path, required=True)
    parser.add_argument("--checkpoint-lock", type=Path, required=True,
                        help="JSON status=locked with model, seed, path, sha256 for every checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    approved = json.loads(args.assembly_report.read_text())
    if not approved["passed"] or approved["status"] != "complete":
        raise RuntimeError("Do not plot an unapproved assembled campaign")
    if digest(args.data) != approved["assembled_data_sha256"]:
        raise ValueError("Assembled data hash differs from its approval record")
    spec_path = Path(__file__).with_name("data_protocol_v1.json")
    if digest(spec_path) != approved["protocol_sha256"]:
        raise ValueError("Protocol changed after dataset approval")
    lock = json.loads(args.checkpoint_lock.read_text())
    if lock.get("protocol_sha256") != approved["protocol_sha256"]:
        raise ValueError("Checkpoint lock does not match the approved protocol")
    verify_model_lock(json.loads(spec_path.read_text()), lock)
    with np.load(args.data, allow_pickle=False) as data:
        names = [str(name) for name in data["path_names"]]
        parameter = np.asarray(data["path_parameter"])
        count = len(parameter)-1
        e = np.asarray(data["E_paths"]).reshape(len(names), count, 3)
        s = np.asarray(data["S_paths"]).reshape(len(names), count, 3)
        d = np.asarray(data["D_paths"]).reshape(len(names), count, 3, 3)
        w = np.asarray(data["W_paths"]).reshape(len(names), count)
        s0 = np.asarray(data["S_reference"])[0]
        d0 = np.asarray(data["D_reference"])[0]
        w0 = float(data["W_reference"][0])
    curves, metrics = {}, {}
    for i, name in enumerate(names):
        strain = np.vstack((np.zeros(3), e[i]))
        stress = np.vstack((s0, s[i]))
        linear_stress, linear_energy = linearization(strain, s0, w0, d0)
        curves[name] = dict(strain=strain, stress=stress, linear_stress=linear_stress,
                            energy=np.r_[w0, w[i]], linear_energy=linear_energy)
        metrics[name] = path_metrics(e[i], s[i], d[i], s0, w0, d0)

    # Matplotlib is deliberately imported only here: helper tests need no GUI.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    args.out.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({"font.size": 10, "axes.titlesize": 10,
                         "font.family": "DejaVu Sans", "text.usetex": False}):
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
        colors = ("#165a87", "#cf7a24")
        for panel, (name, axis_index) in enumerate((("axial_x", 0), ("axial_y", 1))):
            curve = curves[name]
            x = curve["strain"][:, axis_index]
            for component, color in enumerate(colors):
                axes[panel].plot(x, curve["stress"][:, component]/1e6,
                                 color=color, lw=1.8, label=rf"$S_{{{component+1}{component+1}}}$")
                axes[panel].plot(x, curve["linear_stress"][:, component]/1e6,
                                 color=color, lw=1.3, ls="--", alpha=.8)
            axes[panel].set_xlabel(rf"$E_{{{axis_index+1}{axis_index+1}}}$")
            axes[panel].set_title(f"({chr(97+panel)}) Strain-controlled {name[-1]} extension")
            axes[panel].legend(frameon=False, loc="upper left")
        negative, positive = curves["pure_shear_negative"], curves["pure_shear_positive"]
        x = np.r_[negative["strain"][::-1, 2], positive["strain"][1:, 2]]
        shear = np.r_[negative["stress"][::-1, 2], positive["stress"][1:, 2]]
        linear = np.r_[negative["linear_stress"][::-1, 2], positive["linear_stress"][1:, 2]]
        axes[2].plot(x, shear/1e6, color="#257a4b", lw=1.8, label=r"$S_{12}$")
        axes[2].plot(x, linear/1e6, color="#257a4b", lw=1.3, ls="--", alpha=.8)
        axes[2].set_xlabel(r"$\gamma_{12}=2E_{12}$")
        axes[2].set_title("(c) Green-strain shear, both signs")
        axes[2].legend(frameon=False, loc="upper left")
        for axis in axes:
            axis.set_ylabel("Second-Piola stress [MPa]")
            axis.grid(color="#d9d9d9", lw=.5, alpha=.7)
            axis.axhline(0, color="#888888", lw=.5)
        fig.legend(handles=[Line2D([], [], color="#333333", lw=1.8, label="FOM"),
                            Line2D([], [], color="#333333", lw=1.3, ls="--",
                                       label="Reference linearization")],
                   loc="lower center", ncol=2, frameon=False)
        fig.subplots_adjust(left=.075, right=.985, top=.9, bottom=.22, wspace=.36)
        fig.savefig(args.out / "fom_response.pdf", bbox_inches="tight")
        fig.savefig(args.out / "fom_response.png", bbox_inches="tight", dpi=220)
        plt.close(fig)
    result = dict(status="complete", metrics=metrics,
        stress_convention="S=(S11,S22,S12), conjugate to e=(E11,E22,2E12)",
        reference_linearization="S0+D0 e; W0+S0.e+0.5 e.D0.e; no finite-strain labels fitted",
        definitions=dict(stress_path_relative_L2="norm(S_linear-S_FOM)/norm(S_FOM), all nonreference path states",
            stress_endpoint_relative_deviation="norm(S_linear-S_FOM)/norm(S_FOM), endpoint",
            tangent_endpoint_relative_change="norm(D_FOM-D0)_F/norm(D0)_F, endpoint"),
        data_sha256=digest(args.data), assembly_report_sha256=digest(args.assembly_report),
        checkpoint_lock_sha256=digest(args.checkpoint_lock),
        script_sha256=digest(Path(__file__)),
        scope="Finite FOM path diagnostics, not neural accuracy or stability proof. Axial rays hold transverse strain zero, not transverse stress. Green-strain shear is not isochoric simple shear. No out-of-plane or equivalent stress is inferred.")
    (args.out / "nonlinearity.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
