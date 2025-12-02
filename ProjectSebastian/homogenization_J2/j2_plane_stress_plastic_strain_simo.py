import numpy as np


class VonMisesIsotropicPlasticityPlaneStress:
    """
    Small-strain J2 plasticity in plane stress (Simo–Hughes style),
    with *only isotropic hardening*.

    - Plane-stress Voigt notation: [e_xx, e_yy, gamma_xy]
      where gamma_xy = 2 * e_xy
    - Internal variables:
        * PlasticStrain: eps_p (3,)
        * Alpha: equivalent plastic strain (scalar)

    Yield:  f = q - sqrt(2/3) * (sigma_y + H * alpha)  <= 0
    q^2 = eta^T P eta,  eta = sigma (no kinematic hardening).
    """

    def __init__(self, E, nu, sigma_y, H=0.0):
        """
        Parameters
        ----------
        E : float
            Young's modulus
        nu : float
            Poisson's ratio
        sigma_y : float
            Initial yield stress
        H : float, optional
            Isotropic hardening modulus (K' in Simo).
            H = 0 => perfect plasticity.
        """
        self.E = E
        self.nu = nu
        self.sigma_y = sigma_y
        self.H = H

        # Shear modulus
        self.mu = E / (2.0 * (1.0 + nu))

        # Internal variables
        self.voigt_size = 3
        self.PlasticStrain = np.zeros(self.voigt_size)  # eps_p
        self.Alpha = 0.0                                 # equivalent plastic strain

        # Precompute plane-stress elastic matrix and J2 projector P
        self.C = self._compute_C_plane_stress()
        self.P = self._compute_P_plane_stress()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_C_plane_stress(self):
        """
        Plane-stress elastic constitutive matrix in Voigt form:
        [e_xx, e_yy, gamma_xy]  (gamma_xy = 2 e_xy)

        C = E/(1-nu^2) * [[1,   nu,         0       ],
                          [nu,  1,          0       ],
                          [0,   0,  (1-nu)/2       ]]
        """
        C = np.zeros((3, 3), dtype=float)
        factor = self.E / (1.0 - self.nu ** 2)

        C[0, 0] = factor
        C[0, 1] = factor * self.nu
        C[1, 0] = C[0, 1]
        C[1, 1] = factor
        C[2, 2] = factor * (1.0 - self.nu) * 0.5

        return C

    def _compute_P_plane_stress(self):
        """
        Simo's plane-stress J2 projector (eq. 2.4.9):

        P = (1/3) * [[ 2, -1,  0],
                     [-1,  2,  0],
                     [ 0,  0,  6]]
        """
        P = np.array([[2.0, -1.0, 0.0],
                      [-1.0, 2.0, 0.0],
                      [0.0,  0.0, 6.0]], dtype=float) / 3.0
        return P

    def _yield_radius(self, alpha):
        """K(alpha) = sigma_y + H * alpha."""
        return self.sigma_y + self.H * alpha

    # ------------------------------------------------------------------
    # Main material response
    # ------------------------------------------------------------------
    def CalculateMaterialResponse(self, strain):
        """
        Given total strain (numpy array of size 3):
            strain = [e_xx, e_yy, gamma_xy]
        returns the Cauchy stress:
            stress = [sigma_xx, sigma_yy, tau_xy]

        Internal variables (PlasticStrain, Alpha) are UPDATED in-place.
        """
        strain = np.asarray(strain, dtype=float)

        # 1) Elastic trial state
        eps_p_n = self.PlasticStrain
        alpha_n = self.Alpha

        eps_e_trial = strain - eps_p_n                   # elastic trial strain
        sigma_trial = self.C @ eps_e_trial               # trial stress

        # J2 measure in plane stress: q_trial = sqrt(eta^T P eta)
        eta_trial = sigma_trial                          # no kinematic hardening
        q2_trial = eta_trial @ (self.P @ eta_trial)
        q_trial = np.sqrt(max(q2_trial, 0.0))

        # Equivalent yield radius at step n
        K_alpha_n = self._yield_radius(alpha_n)

        # Trial yield function
        sqrt23 = np.sqrt(2.0 / 3.0)
        f_trial = q_trial - sqrt23 * K_alpha_n

        # 2) Elastic case
        if f_trial <= 0.0 or q_trial < 1e-14:
            # purely elastic increment
            return sigma_trial

        # 3) Plastic case: Simo's closed-form radial return (only isotropic hardening)
        # Direction of plastic flow in strain space:
        # m = (P eta_trial) / q_trial
        a_trial = self.P @ eta_trial
        m = a_trial / q_trial

        # Consistency parameter (Remark 3.3.1, only isotropic hardening)
        denom = 2.0 * self.mu + (2.0 / 3.0) * self.H
        dgamma = f_trial / denom
        if dgamma < 0.0:
            dgamma = 0.0  # numerical safety; in theory should not happen

        # Update internal variables
        self.PlasticStrain = eps_p_n + dgamma * m
        self.Alpha = alpha_n + sqrt23 * dgamma

        # Updated elastic strain and stress
        eps_e_new = strain - self.PlasticStrain
        sigma_new = self.C @ eps_e_new

        return sigma_new
