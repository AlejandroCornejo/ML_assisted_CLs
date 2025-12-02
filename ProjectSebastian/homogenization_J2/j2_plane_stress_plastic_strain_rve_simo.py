#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class VonMisesIsotropicPlasticityPlaneStress:
    """
    Small-strain J2 plasticity in plane stress (Simo–Hughes style),
    with ONLY isotropic hardening.

    Voigt notation (plane stress):
        strain  = [e_xx, e_yy, gamma_xy]
        stress  = [s_xx, s_yy, tau_xy]
    where gamma_xy = 2 * e_xy.

    Internal variables (per integration point):
        - PlasticStrain_n : committed eps_p at tn
        - Alpha_n         : committed equivalent plastic strain at tn
        - PlasticStrain   : current (n+1) eps_p
        - Alpha           : current (n+1) equivalent plastic strain

    Yield function:
        f = q - sqrt(2/3) * (sigma_y + H * alpha)
        q^2 = eta^T P eta,   eta = sigma  (no kinematic hardening)

    For H = 0 -> perfect plasticity.
    """

    def __init__(self, E, nu, sigma_y, H=0.0):
        self.E = float(E)
        self.nu = float(nu)
        self.sigma_y = float(sigma_y)
        self.H = float(H)

        # Shear modulus
        self.mu = self.E / (2.0 * (1.0 + self.nu))

        self.voigt_size = 3

        # Committed state at tn
        self.PlasticStrain_n = np.zeros(self.voigt_size)
        self.Alpha_n = 0.0

        # Current state at tn+1 (after last MaterialResponseAndTangent)
        self.PlasticStrain = self.PlasticStrain_n.copy()
        self.Alpha = self.Alpha_n

        # Elastic C and Simo's plane-stress J2 projector P
        self.C = self._compute_C_plane_stress()
        self.P = self._compute_P_plane_stress()

    # --------------------------------------------------------------
    # Basic matrices
    # --------------------------------------------------------------
    def _compute_C_plane_stress(self):
        """
        Plane-stress isotropic elastic matrix (Voigt: [e_xx, e_yy, gamma_xy]):

        C = E/(1-nu^2) * [[1,   nu,         0     ],
                          [nu,  1,          0     ],
                          [0,   0,   (1-nu)/2    ]]
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
        Simo's plane-stress J2 deviatoric 'projector' (eq. 2.4.9):

        P = (1/3) * [[ 2, -1,  0],
                     [-1,  2,  0],
                     [ 0,  0,  6]]
        """
        P = np.array([[2.0, -1.0, 0.0],
                      [-1.0, 2.0, 0.0],
                      [0.0,  0.0, 6.0]], dtype=float) / 3.0
        return P

    def _yield_radius(self, alpha):
        """Isotropic hardening law K(alpha) = sigma_y + H * alpha."""
        return self.sigma_y + self.H * alpha

    # --------------------------------------------------------------
    # Core return-mapping (Simo) – used by both response and Evaluate
    # --------------------------------------------------------------
    def _return_mapping(self, strain, eps_p_prev, alpha_prev):
        """
        Simo-style radial return in plane stress with isotropic hardening.

        Returns:
            sigma      : (3,)
            eps_p_new  : (3,)
            alpha_new  : float
            Cep        : (3,3) algorithmic tangent, consistent with this algorithm
        """
        strain = np.asarray(strain, dtype=float)

        # 1) Elastic trial state
        eps_e_trial = strain - eps_p_prev
        sigma_trial = self.C @ eps_e_trial
        eta_trial   = sigma_trial

        # deviatoric measure with Simo's P
        a_trial   = self.P @ eta_trial
        q2_trial  = eta_trial @ a_trial
        q_trial   = np.sqrt(max(q2_trial, 0.0))

        sqrt23        = np.sqrt(2.0 / 3.0)
        K_alpha_prev  = self._yield_radius(alpha_prev)  # sigma_y + H * alpha_prev
        f_trial       = q_trial - sqrt23 * K_alpha_prev

        tiny = 1e-14

        # 2) Elastic step
        if f_trial <= 0.0 or q_trial < tiny:
            sigma      = sigma_trial
            eps_p_new  = eps_p_prev.copy()
            alpha_new  = alpha_prev
            Cep        = self.C.copy()
            return sigma, eps_p_new, alpha_new, Cep

        # 3) Plastic step: radial return with linear isotropic hardening
        A      = 2.0 * self.mu + (2.0 / 3.0) * self.H   # constant denominator
        dgamma = f_trial / A

        # flow direction in strain space
        m = a_trial / q_trial

        # update internal variables
        eps_p_new = eps_p_prev + dgamma * m
        alpha_new = alpha_prev + sqrt23 * dgamma

        # updated elastic strain and stress
        eps_e_new = strain - eps_p_new
        sigma     = self.C @ eps_e_new

        # 4) Consistent algorithmic tangent (for THIS algorithm)

        # P*C
        PC = self.P @ self.C                    # 3x3

        # term1 = (Δγ / q) C (P C)
        term1 = (dgamma / q_trial) * (self.C @ PC)

        # term2 = ( -Δγ/q + 1/A ) C [ m (m^T C) ]
        mC    = m @ self.C                      # row vector (1x3)
        MM    = np.outer(m, mC)                 # 3x3, m (m^T C)
        coef_m = -dgamma / q_trial + 1.0 / A
        term2  = coef_m * (self.C @ MM)

        Cep = self.C - term1 - term2

        return sigma, eps_p_new, alpha_new, Cep


    # --------------------------------------------------------------
    # Public API expected by your RVE script
    # --------------------------------------------------------------
    def MaterialResponseAndTangent(self, strain):
        """
        Used during Newton iterations.

        - Uses COMMITTED state (PlasticStrain_n, Alpha_n) as "previous".
        - Updates CURRENT state (PlasticStrain, Alpha) to tn+1.
        - Returns sigma, Cep for assembly of K and f_int.
        """
        sigma, eps_p_new, alpha_new, Cep = self._return_mapping(
            strain, self.PlasticStrain_n, self.Alpha_n
        )

        # Store new state as "current" (not yet committed)
        self.PlasticStrain = eps_p_new
        self.Alpha = alpha_new

        return sigma, Cep


    def Evaluate(self, strain):
        """
        Used for homogenization (postprocessing).

        IMPORTANT: does NOT update internal variables; it computes
        the response starting from the COMMITTED state and discards
        the result, so repeated calls are side-effect free.

        Returns sigma plus dummy placeholders to match your call:
            sigma_gp, eps_p_new, alpha_new, Cep
        """
        sigma, eps_p_eval, alpha_eval, Cep = self._return_mapping(
            strain, self.PlasticStrain_n, self.Alpha_n
        )
        return sigma, eps_p_eval, alpha_eval, Cep

    def Commit(self):
        """
        Commit step: tn+1 -> tn.
        """
        self.PlasticStrain_n = self.PlasticStrain.copy()
        self.Alpha_n = float(self.Alpha)


