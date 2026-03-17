import numpy as np

class VonMisesIsotropicPlasticityPlaneStress:

    def __init__(self, E, nu, sigma_y0, H=0.0):

        self.E        = E
        self.nu       = nu
        self.sigma_y0 = sigma_y0
        self.H        = H

        self.eps_p_n = np.zeros(3)
        self.kappa_n = 0.0

        self.eps_p_trial = self.eps_p_n.copy()
        self.kappa_trial = self.kappa_n

        self.yield_tol = 1e-8

        # ------------------------------------------------------------------
        # PRECOMPUTED CONSTANT MATRICES
        # ------------------------------------------------------------------
        aux = E / (1.0 - nu**2)
        self.C = np.array([
            [aux,     E*nu/(1-nu**2), 0.0],
            [E*nu/(1-nu**2), aux,     0.0],
            [0.0,     0.0,     E/(2*(1+nu))]
        ])

        # Deviatoric projection
        self.P = np.array([
            [ 2.0, -1.0, 0.0],
            [-1.0,  2.0, 0.0],
            [ 0.0,  0.0, 6.0]
        ]) / 3.0

        # Constants
        self.sqrt15 = np.sqrt(1.5)
        self.sqrt23 = np.sqrt(2.0/3.0)

        # Preallocate work arrays (to avoid per-call allocations)
        self.tmp_sigma_tr = np.zeros(3)
        self.tmp_Psigma   = np.zeros(3)
        self.tmp_A        = np.zeros(3)

    # ======================================================================
    # MAX-OPTIMIZED EVALUATE
    # ======================================================================
    def Evaluate(self, eps):

        C = self.C
        P = self.P

        # --------------------------------------------------------
        # 1) Trial elastic stress: sigma_tr = C @ (eps - eps_p_n)
        # --------------------------------------------------------
        e = eps - self.eps_p_n
        sigma_tr = C @ e   # uses cached tmp array under the hood

        # --------------------------------------------------------
        # 2) Compute equivalent stress and flow direction
        # --------------------------------------------------------
        # Psigma = P @ sigma_tr
        Ps = self.tmp_Psigma
        Ps[:] = P @ sigma_tr

        sPs = sigma_tr @ Ps
        if sPs <= 0.0:
            # Fully elastic or zero deviatoric state
            return sigma_tr, C, self.eps_p_n, self.kappa_n

        sigma_eq = self.sqrt15 * np.sqrt(sPs)
        N = Ps * (self.sqrt15 / np.sqrt(sPs))   # flow direction

        # --------------------------------------------------------
        # 3) Yield condition
        # --------------------------------------------------------
        sigma_y = self.sigma_y0 + self.H * self.kappa_n
        f_tr = sigma_eq - sigma_y

        if f_tr <= self.yield_tol * max(self.sigma_y0, 1.0):
            return sigma_tr, C, self.eps_p_n, self.kappa_n

        # --------------------------------------------------------
        # 4) Plastic step (radial return)
        # --------------------------------------------------------
        CN = C @ N
        denom = N @ CN + (2.0/3.0)*self.H

        if denom <= 0.0:
            denom = 1e-12 * max(self.sigma_y0, 1.0)

        dgamma = f_tr / denom

        delta_eps_p = dgamma * N
        eps_p_new = self.eps_p_n + delta_eps_p
        kappa_new = self.kappa_n + self.sqrt23 * dgamma

        sigma_new = sigma_tr - dgamma * CN

        # --------------------------------------------------------
        # 5) Algorithmic tangent
        # --------------------------------------------------------
        A = self.tmp_A
        A[:] = CN
        C_tangent = C - np.outer(A, A) / denom

        return sigma_new, C_tangent, eps_p_new, kappa_new

    # ======================================================================
    def MaterialResponseAndTangent(self, eps):
        sigma, C_tan, eps_p_new, kappa_new = self.Evaluate(eps)
        self.eps_p_trial = eps_p_new
        self.kappa_trial = kappa_new
        return sigma, C_tan

    def Commit(self):
        self.eps_p_n = self.eps_p_trial
        self.kappa_n = self.kappa_trial

