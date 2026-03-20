
import numpy as np
import matplotlib.pyplot as plt

class Elastoplastic1D:
    def __init__(self, E, sigma_y, H):
        """
        Parameters:
        E        : Young modulus
        sigma_y  : initial yield stress
        H        : isotropic hardening modulus
        """
        self.E = E
        self.sigma_y = sigma_y
        self.H = H

        # Internal variables (state)
        self.eps_p = 0.0
        self.alpha = 0.0

    def reset_state(self):
        """Reset internal variables"""
        self.eps_p = 0.0
        self.alpha = 0.0

    def update(self, eps):
        """
        Perform one constitutive update given total strain eps.
        Returns updated stress.
        """

        # --- Elastic predictor ---
        sigma_trial = self.E * (eps - self.eps_p)

        f_trial = abs(sigma_trial) - (self.sigma_y + self.H * self.alpha)

        # --- Elastic step ---
        if f_trial <= 0:
            return sigma_trial

        # --- Plastic step ---
        delta_gamma = f_trial / (self.E + self.H)

        s = np.sign(sigma_trial) # df/dsigma = sign(sigma) for 1D case

        # Update stress
        sigma = sigma_trial - self.E * delta_gamma * s

        # Update internal variables
        self.eps_p += delta_gamma * s
        self.alpha += delta_gamma

        return sigma

    def run_history(self, eps_history):
        """
        Apply a strain history sequentially.

        Parameters:
        eps_history : array-like

        Returns:
        sigma_history : numpy array
        """
        sigma_history = []

        for eps in eps_history:
            sigma = self.update(eps)
            sigma_history.append(sigma)

        return np.array(sigma_history)

##########################################################

# Material parameters
E = 210e3
sigma_y = 250.0
H = 10000.0

# Create model
mat = Elastoplastic1D(E, sigma_y, H)

# Strain history
eps = np.linspace(0, 0.01, 50)
eps_unload = np.linspace(0.01, 0.0, 50)
eps = np.concatenate([eps, eps_unload])

# Run simulation
sigma = mat.run_history(eps)

# Plot
plt.plot(eps, sigma, label="Plastic model")
plt.xlabel("Strain")
plt.ylabel("Stress")
plt.legend()
plt.grid()
plt.show()

# save results in npz file
np.savez("1D_plasticity_results.npz", eps=eps, sigma=sigma)
