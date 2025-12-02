import numpy as np
import matplotlib.pyplot as plt
from j2_plane_stress_plastic_strain_rve_simo import VonMisesIsotropicPlasticityPlaneStress

def fd_tangent(mat, eps, eps_p_prev, alpha_prev, h):
    """Central FD tangent at strain eps, starting from (eps_p_prev, alpha_prev)."""
    ncomp = 3
    C_fd = np.zeros((ncomp, ncomp))

    for j in range(ncomp):
        de = np.zeros(ncomp)
        de[j] = h

        sig_p, _, _, _ = mat._return_mapping(eps + de, eps_p_prev, alpha_prev)
        sig_m, _, _, _ = mat._return_mapping(eps - de, eps_p_prev, alpha_prev)

        C_fd[:, j] = (sig_p - sig_m) / (2.0 * h)

    return C_fd

def check_tangent_h_convergence(E, nu, sigma_y, H=0.0):
    mat = VonMisesIsotropicPlasticityPlaneStress(E, nu, sigma_y, H)

    # Previous state (n): virgin
    eps_p_prev = mat.PlasticStrain_n.copy()
    alpha_prev = mat.Alpha_n

    # Strain in plastic regime
    eps = np.array([0.002, 0.0, 0.0])

    # Analytic tangent (this is what we want to verify)
    sigma, _, _, Cep = mat._return_mapping(eps, eps_p_prev, alpha_prev)
    print("sigma =", sigma)
    print("Cep (analytical) =\n", Cep)

    # Different h values to probe the error behaviour
    hs = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4,
                   3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7])

    errs = []

    print("\n h        ||Cep - C_fd(h)||_F    ratio(err(h)/err_prev)")
    print("-------------------------------------------------------")
    prev_err = None
    for k, h in enumerate(hs):
        C_fd = fd_tangent(mat, eps, eps_p_prev, alpha_prev, h)
        E = Cep - C_fd

        # use Frobenius norm for a single scalar error measure
        err = np.linalg.norm(E, ord="fro")
        errs.append(err)

        if k == 0:
            ratio = np.nan
        else:
            ratio = err / prev_err if prev_err > 0 else np.nan

        print(f"{h:8.1e}   {err:18.8e}      {ratio:12.3e}")
        prev_err = err

    errs = np.array(errs)

    # ------------------------------------------------------------
    # Plot: error vs h (log–log) + reference h^2 line
    # ------------------------------------------------------------
    plt.figure()

    # Your error
    plt.loglog(hs, errs, marker="o", label=r"$\|C^{ep} - C^{FD}(h)\|$")

    # Build a reference line ~ h^2, scaled to match at some h0
    # Pick a mid-range index to avoid both truncation and roundoff extremes
    ref_idx = len(hs) // 2
    h0 = hs[ref_idx]
    err0 = errs[ref_idx]

    # reference: err_ref(h) = err0 * (h / h0)^2
    err_ref = err0 * (hs / h0) ** 2
    plt.loglog(hs, err_ref, "--", label=r"$\propto h^2$ (reference)")

    plt.gca().invert_xaxis()  # optional: show small h on the right

    plt.xlabel(r"$h$")
    plt.ylabel(r"$\|C^{ep} - C^{FD}(h)\|_F$")
    plt.title("Tangent consistency check: error vs h")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    E = 206.9e9
    nu = 0.29
    sigma_y = 0.24e9
    H = 0.0  # perfect plasticity
    check_tangent_h_convergence(E, nu, sigma_y, H)

