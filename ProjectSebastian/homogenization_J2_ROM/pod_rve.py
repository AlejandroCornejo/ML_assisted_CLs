#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POD of displacement snapshots for the RVE J2 solver.

- Loads data_set/all_displacement_snapshots.npz (key: 'U')
- Builds snapshot matrix S of shape (n_dof, n_snapshots)
- Computes thin SVD: S = U @ diag(s) @ Vt
- Plots squared energy loss vs mode index with tolerance markers
- Saves truncated bases for several squared-loss tolerances
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# Matplotlib basic styling
# ------------------------------------------------------------
plt.rc('text', usetex=False)
plt.rc('font', family='serif')


# ------------------------------------------------------------
# 1. Load displacement snapshots and build snapshot matrix
# ------------------------------------------------------------
def load_snapshot_matrix(npz_path="data_set/all_displacement_snapshots.npz"):
    """
    Loads the displacement tensor U and reshapes it into a 2D snapshot matrix S.

    Expected format in the npz:
        U : array of shape (n_batches, n_steps+1, n_dof)

    Returns
    -------
    S : ndarray, shape (n_dof, n_snapshots)
        Snapshot matrix with each column = one displacement snapshot.
    U_tensor : ndarray, original tensor (n_batches, n_steps+1, n_dof)
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path)
    if "U" not in data.files:
        raise KeyError(f"'U' key not found in {npz_path}. Available: {data.files}")

    U_tensor = data["U"]  # (n_batches, n_steps+1, n_dof)
    n_batches, n_steps_plus_1, n_dof = U_tensor.shape

    print(f"[INFO] Loaded U with shape (n_batches, n_steps+1, n_dof) "
          f"= {U_tensor.shape}")
    print(f"       n_batches       = {n_batches}")
    print(f"       n_steps+1       = {n_steps_plus_1}")
    print(f"       n_dof (FOM DoFs)= {n_dof}")

    # Flatten batches and time into a single snapshot index
    # Result: (n_batches * (n_steps+1), n_dof)
    U_flat = U_tensor.reshape(-1, n_dof)

    # Snapshot matrix S: (n_dof, n_snapshots)
    S = U_flat.T
    n_snapshots = S.shape[1]

    print(f"[INFO] Snapshot matrix S has shape (n_dof, n_snapshots) = {S.shape}")
    print(f"       n_snapshots = {n_snapshots}")

    return S, U_tensor


# ------------------------------------------------------------
# 2. SVD helpers (same logic as your reference)
# ------------------------------------------------------------
def n_for_tol_squared(sigmas, eps2):
    """
    Given singular values and a squared-loss tolerance eps2, returns the
    smallest n such that
        1 - (sum_{i=1}^n sigma_i^2) / (sum_{j=1}^r sigma_j^2) <= eps2
    """
    s2 = sigmas**2
    csum = np.cumsum(s2)
    total = csum[-1]
    loss2 = 1.0 - csum / total
    # first n with loss <= eps2
    return int(np.argmax(loss2 <= eps2) + 1)


def n_list_for_tols_squared(sigmas, tol_list):
    return [n_for_tol_squared(sigmas, e2) for e2 in tol_list]


def plot_squared_energy_loss(sigmas,
                             tol_list=None,
                             savepath="modes/singular_value_decay_squared_loss.pdf",
                             show=True):
    """
    Plots the squared energy loss:
        eps^2(n) = 1 - (sum_{i=1}^n sigma_i^2) / (sum_{j=1}^r sigma_j^2)
    vs n, on a log scale.

    Optionally adds vertical lines at the n required to meet each tolerance
    in tol_list.
    """
    s2 = sigmas**2
    csum = np.cumsum(s2)
    total = csum[-1]
    loss2 = 1.0 - csum / total

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(loss2, linewidth=2)
    plt.yscale('log')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(r"Singular value index $n$", fontsize=13)
    plt.ylabel(r"$\epsilon^2(n)=1-\frac{\sum_{i=1}^{n}\sigma_i^2}{\sum_{j=1}^{r}\sigma_j^2}$",
               fontsize=13)

    if tol_list is not None:
        colors = ['k', 'm', 'g', 'b', 'r', 'c', 'y']
        n_needed = n_list_for_tols_squared(sigmas, tol_list)
        for k, (eps2, nm) in enumerate(zip(tol_list, n_needed)):
            plt.axvline(x=nm, ymin=0.05, ymax=0.95,
                        color=colors[k % len(colors)],
                        label=rf"$\epsilon^2={eps2:.0e}$ ($n={nm}$)")
        plt.legend()

    plt.tight_layout()
    plt.savefig(savepath, format='pdf')
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------
# 3. Main: run POD and generate plots + saved modes
# ------------------------------------------------------------
def main():
    # 1) Load snapshot matrix
    S, _ = load_snapshot_matrix("data_set/all_displacement_snapshots.npz")

    # 2) Thin SVD
    print("[INFO] Computing thin SVD of S ...")
    Uleft, s, Vt = np.linalg.svd(S, full_matrices=False)
    r = s.size
    print(f"[INFO] Thin SVD done. Rank r = {r}")
    print(f"       Uleft shape = {Uleft.shape}")
    print(f"       s shape     = {s.shape}")
    print(f"       Vt shape    = {Vt.shape}")

    # 3) Choose squared-loss tolerances
    eps2_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-12, 1e-16]
    n_modes_list = n_list_for_tols_squared(s, eps2_list)
    print("\n[INFO] Modes for squared-loss tolerances:")
    for e2, nm in zip(eps2_list, n_modes_list):
        print(f"  epsilon^2 = {e2:.0e}  ->  n = {nm}")

    # Ensure modes directory
    os.makedirs("modes", exist_ok=True)

    # 4) Plot squared energy loss
    plot_squared_energy_loss(
        s,
        tol_list=eps2_list,
        savepath="modes/singular_value_decay_squared_loss.pdf",
        show=True
    )

    # 5) Save leading modes for later PROM experiments
    for e2, nm in zip(eps2_list, n_modes_list):
        fname_modes = f"modes/U_modes_tol_{e2:.0e}.npy"
        fname_svals = f"modes/Singular_values_modes_tol_{e2:.0e}.npy"
        np.save(fname_modes, Uleft[:, :nm])
        np.save(fname_svals, s[:nm])

    # Optionally: save full left singular vectors and singular values
    np.save("modes/Uleft_full.npy", Uleft)
    np.save("modes/S_full.npy", s)

    print("\n[INFO] Saved:")
    print("  - modes/singular_value_decay_squared_loss.pdf")
    print("  - modes/U_modes_tol_*.npy")
    print("  - modes/Singular_values_modes_tol_*.npy")
    print("  - modes/Uleft_full.npy")
    print("  - modes/S_full.npy")


if __name__ == "__main__":
    main()
