#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POD of displacement snapshots for the RVE J2 solver.

- Loads all displacement files U_*.npy from a given directory (e.g. 'training_set')
- Each file is expected to have shape (n_steps+1, n_dof)
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
def load_snapshot_matrix_from_dir(
    snapshots_dir="training_set",
    prefix="U_",
    verbose=True
):
    """
    Loads all displacement snapshots U_*.npy from a directory and reshapes them
    into a 2D snapshot matrix S.

    Expected per-file format:
        U_file : array of shape (n_steps+1, n_dof)

    Parameters
    ----------
    snapshots_dir : str
        Directory where U_*.npy are stored (e.g. 'training_set' or 'testing_set').
    prefix : str
        Filename prefix for displacement files (default: 'U_').
    verbose : bool

    Returns
    -------
    S : ndarray, shape (n_dof, n_snapshots)
        Snapshot matrix with each column = one displacement snapshot.
    U_tensor : ndarray or None
        If all files share the same shape, returns a tensor of shape
        (n_batches, n_steps+1, n_dof). Otherwise, returns None.
    """
    if not os.path.isdir(snapshots_dir):
        raise NotADirectoryError(f"Directory not found: {snapshots_dir}")

    # Collect all U_*.npy files
    files = [
        f for f in os.listdir(snapshots_dir)
        if f.startswith(prefix) and f.endswith(".npy")
    ]
    files.sort()

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files matching '{prefix}*.npy' found in {snapshots_dir}"
        )

    if verbose:
        print(f"[INFO] Found {len(files)} displacement files in '{snapshots_dir}':")
        for f in files:
            print(f"       {f}")

    U_list = []
    shapes = []

    for f in files:
        path = os.path.join(snapshots_dir, f)
        U = np.load(path)  # expected (n_steps+1, n_dof)
        if U.ndim != 2:
            raise ValueError(
                f"File {path} has shape {U.shape}, expected 2D (n_steps+1, n_dof)"
            )
        U_list.append(U)
        shapes.append(U.shape)

    # Determine if all batches share same shape
    same_shape = all(sh == shapes[0] for sh in shapes)
    if verbose:
        print(f"[INFO] First U shape = {shapes[0]}")
        if same_shape:
            print("[INFO] All batches share the same shape.")
        else:
            print("[WARN] Batches have different shapes; U_tensor will be None.")

    # Concatenate along time/batch dimension
    # Each U is (n_steps+1, n_dof) -> stack them vertically
    U_concat = np.concatenate(U_list, axis=0)  # (sum_over_batches n_steps+1, n_dof)
    n_total_snapshots, n_dof = U_concat.shape

    if verbose:
        print(f"[INFO] Concatenated U_concat shape = {U_concat.shape}")
        print(f"       total_snapshots = {n_total_snapshots}")
        print(f"       n_dof (FOM DoFs)= {n_dof}")

    # Snapshot matrix S: (n_dof, n_snapshots)
    S = U_concat.T
    n_snapshots = S.shape[1]

    if verbose:
        print(f"[INFO] Snapshot matrix S has shape (n_dof, n_snapshots) = {S.shape}")
        print(f"       n_snapshots = {n_snapshots}")

    # Optionally build U_tensor if all shapes equal
    if same_shape:
        U_tensor = np.stack(U_list, axis=0)  # (n_batches, n_steps+1, n_dof)
        if verbose:
            print(f"[INFO] U_tensor shape = {U_tensor.shape}")
    else:
        U_tensor = None

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
    plt.ylabel(r"$\epsilon^2(n)=1-\frac{\sum_{i=1}^{n}\sigma_i^2}"
               r"{\sum_{j=1}^{r}\sigma_j^2}$",
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
    # 1) Load snapshot matrix from directory (e.g. 'training_set')
    snapshots_dir = "training_set"   # change to "testing_set" if needed
    S, _ = load_snapshot_matrix_from_dir(snapshots_dir=snapshots_dir, prefix="U_")

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
