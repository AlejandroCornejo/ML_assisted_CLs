"""
Analytical compressible Neo-Hookean hyperelastic constitutive law (JAX).

The class is a simple, closed-form (non-learned) constitutive law that
matches the interface of the learned CLs in this repository:

    - forward  : strain-energy density  W(E)
    - gradient : Cauchy stress         sigma(E)  (via dW/dE)

Conventions (identical to the rest of this repository, e.g.
MOEKAN/train_MOEKAN_data_labels_b.py):

    - Input strain is the 2D Green-Lagrange strain in Voigt form
      [E11, E22, 2*E12] (Kratos GREEN_LAGRANGE_STRAIN_VECTOR convention).
    - The stress returned by the gradient is the Cauchy stress in Voigt
      form [sigma11, sigma22, 2*sigma12].

Model (compressible Neo-Hookean, 2D):

    W = (mu/2) * (I1 - 3) + (lambda/2) * (J - 1)^2

    with
        C  = I + 2E          (right Cauchy-Green tensor)
        I1 = tr(C)
        J  = sqrt(det(C))    (Jacobian of the deformation gradient)

    The second Piola-Kirchhoff stress is the gradient of W w.r.t. E:

        S = dW/dE = mu * I + lambda * (J - 1) * J * (2*C^{-1} - diag(C^{-1}))

    where diag(C^{-1}) denotes the tensor with the diagonal of C^{-1}
    on the diagonal and zeros off-diagonal.  The factor of 2 on the
    off-diagonal arises because C is a symmetric tensor (C12 = C21 = 2*E12),
    so dJ/dC12 = -C12/J = J*(C^{-1})12, not (J/2)*(C^{-1})12.

    and the Cauchy stress is obtained from the standard push-forward:

        sigma = (1/J) * F * S * F^T,   F = C^{1/2}

    which, for this isotropic law, reduces to the spectral form

        sigma = (mu / J) * C + lambda * (J - 1) * I

    (since F C^{-1} F^T = I and F F^T = C in the eigenbasis of C).
"""

import jax
import jax.numpy as jnp


class CompressibleNeoHookeanCL:
    """
    Analytical compressible Neo-Hookean constitutive law.

    Args:
        mu (float): shear modulus.
        lam (float): Lamé's first parameter (bulk-related).
        dtype: JAX dtype for the parameters (default jnp.float32).
    """

    def __init__(self, mu=1.0, lam=1.0, dtype=jnp.float32):
        self.mu = jnp.array(mu, dtype=dtype)
        self.lam = jnp.array(lam, dtype=dtype)
        self.dtype = dtype

        # 2D identity tensor.
        self.I = jnp.eye(2, dtype=dtype)

    # -----------------------------------------------------------------
    def _strain_to_tensors(self, strain):
        """
        Reconstruct the Green-Lagrange strain tensor E and the
        right Cauchy-Green tensor C = I + 2E from the Voigt input.

        Args:
            strain (jnp.ndarray): array of shape (..., 3) with
                [E11, E22, 2*E12].

        Returns:
            tuple:
                - E (jnp.ndarray): (..., 2, 2) Green-Lagrange strain tensor.
                - C (jnp.ndarray): (..., 2, 2) right Cauchy-Green tensor.
        """
        E11 = strain[..., 0]
        E22 = strain[..., 1]
        E12 = 0.5 * strain[..., 2]  # Voigt 3rd component is 2*E12

        E = jnp.stack(
            [
                jnp.stack([E11, E12], axis=-1),
                jnp.stack([E12, E22], axis=-1),
            ],
            axis=-2,
        )

        C = self.I + 2.0 * E
        return E, C

    # -----------------------------------------------------------------
    def _invariants(self, C):
        """
        Compute I1 = tr(C) and J = sqrt(det(C)) from C.

        Args:
            C (jnp.ndarray): (..., 2, 2) right Cauchy-Green tensor.

        Returns:
            tuple: (I1, J), each of shape (...).
        """
        I1 = jnp.trace(C, axis1=-2, axis2=-1)
        det_C = jnp.linalg.det(C)
        J = jnp.sqrt(jnp.clip(det_C, min=1.0e-12))
        return I1, J

    # -----------------------------------------------------------------
    def forward(self, strain):
        """
        Strain-energy density W(E) (forward pass).

        Args:
            strain (jnp.ndarray): array of shape (..., 3) with the
                Green-Lagrange strain in Voigt form [E11, E22, 2*E12].

        Returns:
            jnp.ndarray: strain-energy density W of shape (...).
        """
        _, C = self._strain_to_tensors(strain)
        I1, J = self._invariants(C)

        W = 0.5 * self.mu * (I1 - 3.0) + 0.5 * self.lam * (J - 1.0) ** 2
        return W

    # -----------------------------------------------------------------
    def gradient(self, strain):
        """
        Cauchy stress sigma(E), computed as the push-forward of
        S = dW/dE obtained via JAX autodiff of the strain-energy density.

        Args:
            strain (jnp.ndarray): array of shape (..., 3) with the
                Green-Lagrange strain in Voigt form [E11, E22, 2*E12].

        Returns:
            jnp.ndarray: Cauchy stress in Voigt form
                [sigma11, sigma22, 2*sigma12] of shape (..., 3).
        """
        # 2nd Piola-Kirchhoff stress S = dW/dE via JAX autodiff.
        # Voigt strain: [E11, E22, 2*E12], so E12 = strain[2]/2.
        # dW/d(strain[0]) = S11,  dW/d(strain[1]) = S22,
        # dW/d(strain[2]) = dW/dE12 * dE12/d(strain[2]) = S12 / 2.
        # Hence S12 = 2 * dW/d(strain[2]).
        # Use vmap+grad to handle both single (3,) and batched (..., 3) inputs.
        dW_dstrain = jax.vmap(jax.grad(self.forward))(strain)
        S11 = dW_dstrain[..., 0]
        S22 = dW_dstrain[..., 1]
        S12 = 2.0 * dW_dstrain[..., 2]

        # Reconstruct the 2nd Piola stress tensor S (..., 2, 2).
        S = jnp.stack(
            [
                jnp.stack([S11, S12], axis=-1),
                jnp.stack([S12, S22], axis=-1),
            ],
            axis=-2,
        )

        # Push forward to Cauchy stress: sigma = (1/J) * F S F^T.
        _, C = self._strain_to_tensors(strain)
        _, J = self._invariants(C)
        J_ = J[..., jnp.newaxis, jnp.newaxis]

        # F = C^{1/2} via eigendecomposition (C is symmetric positive definite).
        eigvals, eigvecs = jnp.linalg.eigh(C)
        # Build batched diagonal matrix: (..., 2, 2)
        sqrt_eigvals = jnp.sqrt(eigvals)  # (..., 2)
        diag_mat = sqrt_eigvals[..., :, jnp.newaxis] * jnp.eye(2)  # (..., 2, 2)
        F = eigvecs @ diag_mat @ jnp.swapaxes(eigvecs, -1, -2)

        sigma = (1.0 / J_) * (F @ S @ jnp.swapaxes(F, -1, -2))

        # Pack into Voigt form [sigma11, sigma22, 2*sigma12].
        sigma11 = sigma[..., 0, 0]
        sigma22 = sigma[..., 1, 1]
        sigma12 = sigma[..., 0, 1]

        return jnp.stack([sigma11, sigma22, 2.0 * sigma12], axis=-1)

    # -----------------------------------------------------------------
    def __call__(self, strain):
        """
        Forward pass: returns the strain-energy density W(E).

        Args:
            strain (jnp.ndarray): array of shape (..., 3) with the
                Green-Lagrange strain in Voigt form [E11, E22, 2*E12].

        Returns:
            jnp.ndarray: strain-energy density W of shape (...).
        """
        return self.forward(strain)


# =============================================================================
# Self-test: verify the analytical stresses against JAX autodiff of W.
# =============================================================================
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    cl = CompressibleNeoHookeanCL(mu=1.0, lam=1.0, dtype=jnp.float64)

    # A few random strain states (Voigt form [E11, E22, 2*E12]).
    key = jax.random.PRNGKey(0)
    strains = jax.random.normal(key, (100, 3)) * 0.1

    # --- Reference 2nd Piola stress S = dW/dE via autodiff -------------------
    # Voigt strain: [E11, E22, 2*E12], so E12 = strain[2]/2.
    # dW/d(strain[2]) = S12/2  =>  S12 = 2 * dW/d(strain[2]).
    dW_dE = jax.vmap(jax.grad(cl.forward))(strains)  # (N, 3)
    S11_ref, S22_ref, S12_ref = dW_dE[..., 0], dW_dE[..., 1], 2.0 * dW_dE[..., 2]
    S_ref = jnp.stack(
        [
            jnp.stack([S11_ref, S12_ref], axis=-1),
            jnp.stack([S12_ref, S22_ref], axis=-1),
        ],
        axis=-2,
    )  # (N, 2, 2)

    # --- Analytical 2nd Piola stress ----------------------------------------
    # S = dW/dE = mu*I + lam*(J-1)*J*(2*C^{-1} - diag(C^{-1}))
    # The factor of 2 on off-diagonal arises because C is symmetric
    # (C12 = C21 = 2*E12), so dJ/dC12 = J*(C^{-1})12, not (J/2)*(C^{-1})12.
    _, C = cl._strain_to_tensors(strains)
    _, J = cl._invariants(C)
    J_ = J[..., jnp.newaxis, jnp.newaxis]
    C_inv = jnp.linalg.inv(C)
    diag_C_inv = C_inv * jnp.eye(2)  # zero out off-diagonal
    S_analytical = cl.mu * cl.I + cl.lam * (J_ - 1.0) * J_ * (2.0 * C_inv - diag_C_inv)

    err_S = jnp.max(jnp.abs(S_analytical - S_ref))
    print(f"Max |S_analytical - dW/dE| = {err_S:.3e}")
    assert err_S < 1.0e-8, "Analytical 2nd Piola stress does not match autodiff!"

    # --- Cauchy stress: push-forward of the autodiff S -----------------------
    # F = C^{1/2} via eigendecomposition (C symmetric positive definite).
    eigvals, eigvecs = jnp.linalg.eigh(C)
    sqrt_eigvals = jnp.sqrt(eigvals)  # (N, 2)
    diag_mat = sqrt_eigvals[..., :, jnp.newaxis] * jnp.eye(2)  # (N, 2, 2)
    F = eigvecs @ diag_mat @ jnp.swapaxes(eigvecs, -1, -2)
    sigma_ref = (1.0 / J_) * (
        jnp.matmul(jnp.matmul(F, S_ref), jnp.swapaxes(F, -1, -2))
    )

    # --- Analytical Cauchy stress (Voigt form) --------------------------------
    sigma_analytical = cl.gradient(strains)
    sigma11 = sigma_analytical[..., 0]
    sigma22 = sigma_analytical[..., 1]
    sigma12 = 0.5 * sigma_analytical[..., 2]
    sigma_analytical_tensor = jnp.stack(
        [
            jnp.stack([sigma11, sigma12], axis=-1),
            jnp.stack([sigma12, sigma22], axis=-1),
        ],
        axis=-2,
    )

    err_sigma = jnp.max(jnp.abs(sigma_analytical_tensor - sigma_ref))
    print(f"Max |sigma_analytical - push_forward(dW/dE)| = {err_sigma:.3e}")
    assert err_sigma < 1.0e-8, "Analytical Cauchy stress does not match push-forward!"

    print("OK: analytical 2nd Piola and Cauchy stresses match JAX autodiff of W.")
