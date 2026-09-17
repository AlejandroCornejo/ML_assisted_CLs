"""Plot the frozen material-B coordinate design; no labels are loaded."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if digest(args.design) != report["design_sha256"]:
        raise ValueError("Coordinate-design hash differs from its report")
    with np.load(args.design, allow_pickle=False) as data:
        fit, kind = data["E_fit"], data["fit_kind"]
        validation, test = data["E_validation"], data["E_test"]
        paths = data["path_states"]

    labels = (r"$E_{11}$", r"$E_{22}$", r"$\gamma_{12}=2E_{12}$")
    pairs = ((0, 1), (0, 2), (1, 2))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.65))
    volume = kind == "volume"
    boundary = ~volume
    for panel, (axis, (i, j)) in enumerate(zip(axes, pairs)):
        axis.scatter(fit[volume, i], fit[volume, j], s=4, c="#8fb9d6", alpha=.20,
                     linewidths=0, rasterized=True, label="fit: volume")
        axis.scatter(fit[boundary, i], fit[boundary, j], s=10, c="#165a87", alpha=.75,
                     linewidths=0, rasterized=True, label="fit: faces/corners")
        axis.scatter(validation[:, i], validation[:, j], s=8, facecolors="none",
                     edgecolors="#d9822b", linewidths=.45, alpha=.55, rasterized=True,
                     label="validation")
        axis.scatter(test[:, i], test[:, j], s=8, marker="x", c="#7b3294",
                     linewidths=.45, alpha=.48, rasterized=True, label="test")
        for path_index, path in enumerate(paths):
            axis.plot(path[:, i], path[:, j], color="#257a4b", linewidth=1.0,
                      alpha=.78, label="held-out paths" if path_index == 0 else None)
        axis.set_xlabel(labels[i])
        axis.set_ylabel(labels[j])
        axis.grid(color="#d9d9d9", linewidth=.45, alpha=.7)
        axis.text(.025, .97, f"({chr(97+panel)})", transform=axis.transAxes,
                  ha="left", va="top", fontweight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, .01), ncol=5,
               frameon=False, columnspacing=1.25, handletextpad=.35)
    fig.subplots_adjust(left=.07, right=.985, top=.97, bottom=.22, wspace=.34)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "Material B frozen constitutive sampling design",
                "Subject": "Coordinate coverage only; no FOM labels or model predictions",
                "Keywords": report["design_sha256"]}
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight",
                metadata={"Description": metadata["Subject"], "design_sha256": report["design_sha256"]})
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
