import numpy as np


class VonMisesIsotropicPlasticityPlaneStress:
    """
    Small-strain J2 plasticity, plane stress, perfect plasticity.

    Internal variables:
        eps_p : plastic strain (Voigt [exx, eyy, gxy])
        kappa : equivalent plastic strain

    This is a standard radial-return algorithm.
    """

    def __init__(self, E, nu, sigma_y0, H=0.0):
        self.dimension  = 2
        self.voigt_size = 3

        self.E        = E          # Young's modulus
        self.nu       = nu         # Poisson's ratio
        self.sigma_y0 = sigma_y0   # initial yield stress
        self.H        = H          # isotropic hardening modulus (0 = perfect plasticity)

        # internal variables
        self.eps_p = np.zeros(self.voigt_size)  # plastic strain
        self.kappa = 0.0                        # equivalent plastic strain

    # ------------------------------------------------------------------
    # Elastic matrix (plane stress)
    # ------------------------------------------------------------------
    def CalculateConstitutiveMatrix(self):
        E  = self.E
        nu = self.nu

        C = np.zeros((3, 3))
        aux = E / (1.0 - nu**2)
        C[0, 0] = aux
        C[0, 1] = E * nu / (1.0 - nu**2)
        C[1, 0] = C[0, 1]
        C[1, 1] = aux
        C[2, 2] = E / (2.0 * (1.0 + nu))  # shear modulus

        return C

    # ------------------------------------------------------------------
    # J2 machinery
    # ------------------------------------------------------------------
    def GetPMatrix(self):
        """
        Deviatoric operator in plane stress (Voigt: [xx, yy, xy]).
        """
        return np.array([[2.0, -1.0, 0.0],
                         [-1.0, 2.0, 0.0],
                         [0.0,  0.0, 6.0]]) / 3.0

    def CalculateEquivalentStress(self, sigma):
        P = self.GetPMatrix()
        return np.sqrt(1.5 * sigma @ (P @ sigma))

    def yield_stress(self):
        """
        Isotropic hardening law: sigma_y = sigma_y0 + H * kappa.
        """
        return self.sigma_y0 + self.H * self.kappa

    def YieldFunction(self, sigma):
        return self.CalculateEquivalentStress(sigma) - self.yield_stress()

    def FlowDirection(self, sigma):
        """
        d sigma_eq / d sigma = sqrt(3/2) * P*sigma / sqrt(sigma^T P sigma).
        """
        P   = self.GetPMatrix()
        sPs = sigma @ (P @ sigma)
        if sPs <= 0.0:
            return np.zeros_like(sigma)
        return (np.sqrt(1.5) / np.sqrt(sPs)) * (P @ sigma)

    # ------------------------------------------------------------------
    # Radial-return material response
    # ------------------------------------------------------------------
    def CalculateMaterialResponse(self, eps):
        """
        Given total strain eps (Voigt), return Cauchy stress sigma (Voigt).
        Updates internal variables (eps_p, kappa).
        """
        C = self.CalculateConstitutiveMatrix()

        # Trial elastic stress
        sigma_tr = C @ (eps - self.eps_p)
        f_tr     = self.YieldFunction(sigma_tr)

        # Elastic step
        if f_tr <= 0.0:
            return sigma_tr

        # Plastic step: radial return
        N     = self.FlowDirection(sigma_tr)
        denom = N @ (C @ N) + (2.0 / 3.0) * self.H
        dgamma = f_tr / denom

        delta_eps_p = dgamma * N

        # Update internal variables
        self.eps_p += delta_eps_p
        self.kappa += np.sqrt(2.0 / 3.0) * dgamma

        # Updated stress
        sigma = sigma_tr - C @ delta_eps_p
        return sigma
