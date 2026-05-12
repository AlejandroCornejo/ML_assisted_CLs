import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()

def plot_svd_decay(out_path):
    # Load singular values
    s = np.load("../pod_rbf_model/singular_values.npy")
    
    with open("../pod_rbf_model/metadata.json", "r") as f:
        meta = json.load(f)
        
    rank = meta["rank"]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    modes = np.arange(1, len(s)+1)
    
    # Plot first 100 modes
    ax.plot(modes[:100], s[:100], 'ko-', markersize=4, linewidth=1.5, label='Singular Values $\\sigma_i$')
    ax.axvline(x=rank, color='red', linestyle='--', label=f'Truncation $r={rank}$')
    
    # Also plot cumulative energy on right axis
    energy = np.cumsum(s**2) / np.sum(s**2)
    ax2 = ax.twinx()
    ax2.plot(modes[:100], energy[:100], 'b--', linewidth=1.5, label='Cumulative Energy (right)')
    ax2.set_ylabel('Cumulative Energy Fraction', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax.set_yscale('log')
    ax.set_xlabel('Mode Index $i$')
    ax.set_ylabel('Singular Value $\\sigma_i$')
    ax.set_title(f'POD Singular Value Decay (from {meta["total_snapshots"]} snapshots)')
    ax.grid(True, alpha=0.3, which='both')
    
    # Handle legends
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    plot_svd_decay("pod_decay.png")
