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
    # Vectorized / batched J2 plane-stress return mapping
    # --------------------------------------------------------------
    def _return_mapping_batch(self, strain_batch, eps_p_prev_batch, alpha_prev_batch):
        """
        Batched Simo-style radial return in plane stress with isotropic hardening.

        Parameters
        ----------
        strain_batch      : (n_gp, 3)
        eps_p_prev_batch  : (n_gp, 3)
        alpha_prev_batch  : (n_gp,)

        Returns
        -------
        sigma_batch       : (n_gp, 3)
        eps_p_new_batch   : (n_gp, 3)
        alpha_new_batch   : (n_gp,)
        Cep_batch         : (n_gp, 3, 3)
        """
        strain_batch     = np.asarray(strain_batch, dtype=float)
        eps_p_prev_batch = np.asarray(eps_p_prev_batch, dtype=float)
        alpha_prev_batch = np.asarray(alpha_prev_batch, dtype=float)

        # Normalize shapes
        if strain_batch.ndim == 1:
            strain_batch = strain_batch[np.newaxis, :]
        if eps_p_prev_batch.ndim == 1:
            eps_p_prev_batch = eps_p_prev_batch[np.newaxis, :]
        if alpha_prev_batch.ndim == 0:
            alpha_prev_batch = alpha_prev_batch[np.newaxis]

        n_gp   = strain_batch.shape[0]
        tiny   = 1e-14
        sqrt23 = np.sqrt(2.0 / 3.0)

        # ---- 1) Elastic trial (vectorized, same as scalar) ----
        eps_e_trial = strain_batch - eps_p_prev_batch            # (n_gp, 3)
        sigma_trial = (self.C @ eps_e_trial.T).T                 # (n_gp, 3)
        eta_trial   = sigma_trial

        a_trial  = (self.P @ eta_trial.T).T                      # (n_gp, 3)
        q2_trial = np.einsum("ij,ij->i", eta_trial, a_trial)     # eta^T P eta
        q_trial  = np.sqrt(np.maximum(q2_trial, 0.0))            # (n_gp,)

        K_alpha_prev = self._yield_radius(alpha_prev_batch)      # (n_gp,)
        f_trial      = q_trial - sqrt23 * K_alpha_prev           # (n_gp,)

        # Elastic / plastic mask exactly as scalar
        elastic_mask = (f_trial <= 0.0) | (q_trial < tiny)
        plastic_mask = ~elastic_mask

        # Allocate outputs
        sigma_batch      = np.empty_like(sigma_trial)
        eps_p_new_batch  = np.empty_like(eps_p_prev_batch)
        alpha_new_batch  = np.empty_like(alpha_prev_batch)
        Cep_batch        = np.empty((n_gp, 3, 3), dtype=float)

        # ---- 2) Elastic points ----
        sigma_batch[elastic_mask]     = sigma_trial[elastic_mask]
        eps_p_new_batch[elastic_mask] = eps_p_prev_batch[elastic_mask]
        alpha_new_batch[elastic_mask] = alpha_prev_batch[elastic_mask]
        Cep_batch[elastic_mask]       = self.C   # broadcast elastic C

        # ---- 3) Plastic points (vectorized, SAME algorithm as scalar) ----
        if np.any(plastic_mask):
            strain_pl     = strain_batch[plastic_mask]           # (np, 3)
            eps_p_prev_pl = eps_p_prev_batch[plastic_mask]       # (np, 3)
            alpha_prev_pl = alpha_prev_batch[plastic_mask]       # (np,)
            a_trial_pl    = a_trial[plastic_mask]                # (np, 3)
            q_trial_pl    = q_trial[plastic_mask]                # (np,)
            f_trial_pl    = f_trial[plastic_mask]                # (np,)

            # dgamma = f_trial / (2 mu + 2/3 H)
            A = 2.0 * self.mu + (2.0 / 3.0) * self.H             # scalar
            dgamma_pl = f_trial_pl / A                           # (np,)

            # flow direction m = a_trial / q_trial
            m_pl = a_trial_pl / q_trial_pl[:, None]              # (np, 3)

            # update internal variables
            eps_p_new_pl = eps_p_prev_pl + dgamma_pl[:, None] * m_pl
            alpha_new_pl = alpha_prev_pl + sqrt23 * dgamma_pl

            eps_e_new_pl = strain_pl - eps_p_new_pl
            sigma_pl     = (self.C @ eps_e_new_pl.T).T           # (np, 3)

            sigma_batch[plastic_mask]     = sigma_pl
            eps_p_new_batch[plastic_mask] = eps_p_new_pl
            alpha_new_batch[plastic_mask] = alpha_new_pl

            # ---- 4) Consistent tangent (using EXACT scalar formula) ----
            # PC = P C  (constant for this material)
            PC  = self.P @ self.C                                # (3,3)
            CPC = self.C @ PC                                   # (3,3) = C (P C)

            # term1 = (Δγ / q) C (P C)
            factor1_pl = (dgamma_pl / q_trial_pl)[:, None, None]  # (np,1,1)
            term1_pl   = factor1_pl * CPC                         # (np,3,3)

            # term2 = ( -Δγ/q + 1/A ) C [ m (m^T C) ]
            mC_pl = m_pl @ self.C                               # (np, 3)
            MM_pl = m_pl[:, :, None] * mC_pl[:, None, :]        # (np, 3, 3)  m(m^T C)

            coef_m_pl = (-dgamma_pl / q_trial_pl + 1.0 / A)[:, None, None]  # (np,1,1)

            # C @ MM, using einsum to keep the same operation as scalar
            # (CMM[e,i,j] = sum_k C[i,k] * MM[e,k,j])
            CMM_pl = np.einsum("ik,ekj->eij", self.C, MM_pl)     # (np,3,3)
            term2_pl = coef_m_pl * CMM_pl                        # (np,3,3)

            Cep_pl = self.C - term1_pl - term2_pl               # (np,3,3)

            Cep_batch[plastic_mask] = Cep_pl

        return sigma_batch, eps_p_new_batch, alpha_new_batch, Cep_batch



    def _return_mapping(self, strain, eps_p_prev, alpha_prev):
        sigma_batch, eps_p_batch, alpha_batch, Cep_batch = \
            self._return_mapping_batch(
                strain_batch=np.asarray(strain)[np.newaxis, :],
                eps_p_prev_batch=np.asarray(eps_p_prev)[np.newaxis, :],
                alpha_prev_batch=np.asarray(alpha_prev)[np.newaxis]
            )

        sigma      = sigma_batch[0, :]
        eps_p_new  = eps_p_batch[0, :]
        alpha_new  = float(alpha_batch[0])
        Cep        = Cep_batch[0, :, :]

        return sigma, eps_p_new, alpha_new, Cep

        
    def MaterialResponseAndTangent_batch(self, strain_batch, eps_p_prev_batch, alpha_prev_batch):
        return self._return_mapping_batch(strain_batch, eps_p_prev_batch, alpha_prev_batch)

    def Evaluate(self, strain):
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


