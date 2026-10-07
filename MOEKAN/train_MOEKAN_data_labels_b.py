"""
MOEKAN training on the data_labels_b dataset (data_labels_v1.npz).

Training : E_fit / S_fit  (4200 samples)
Plotting : E_test / S_test (512 samples)

The model learns the strain-energy potential W such that
    dW/dE = S  (stress in Voigt form [S11, S22, S12])
via the chain rule:
    dW/dE = dW/d(moekan_inputs) @ d(moekan_inputs)/dE

The MOEKAN inputs are derived from the Green-Lagrange strain:
    [lambda_x, lambda_y, log(J)]
where lambda_x, lambda_y are the principal stretches and J = det(F).
"""

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from MOEKAN import MOEKAN


# =============================================================================
# STRAIN -> MOEKAN INPUTS TRANSFORMATION
# =============================================================================
"""
TRANSFORMATION FROM GREEN-LAGRANGE STRAIN TO MOEKAN INPUTS:

    strain (N, 3)  ->  moekan_inputs (N, 3)

The strain is the 2D Green-Lagrange strain in Voigt form
[E11, E22, 2*E12] (Kratos GREEN_LAGRANGE_STRAIN_VECTOR convention).

The MOEKAN inputs are:
    [lambda_x, lambda_y, log(J)]

where:
    - lambda_x, lambda_y are the principal stretches of F,
      i.e. the square roots of the eigenvalues of C = I + 2E.
    - J = sqrt(det(C)) is the Jacobian of the deformation gradient.
"""
def _strain_to_moekan_inputs_single(strain):
    """
    Core computation for a SINGLE strain sample.

    Args:
        strain (jnp.ndarray): array of shape (3,) with the
            Green-Lagrange strain in Voigt form [E11, E22, 2*E12].

    Returns:
        jnp.ndarray: array of shape (3,) with
            [lambda_x, lambda_y, log(J)].
    """
    E11 = strain[0]
    E22 = strain[1]
    E12 = 0.5 * strain[2]

    # Right Cauchy-Green deformation tensor: C = I + 2E.
    C = jnp.array(
        [
            [1.0 + 2.0 * E11, E12],
            [E12, 1.0 + 2.0 * E22],
        ]
    )

    # Principal stretches of F: square roots of the eigenvalues of C.
    eigvals = jnp.linalg.eigvalsh(C)
    lambdas = jnp.sqrt(eigvals)

    lambda_x = lambdas[0]
    lambda_y = lambdas[1]

    # J = sqrt(det(C)) = product of the principal stretches.
    J = lambda_x * lambda_y
    log_J = jnp.log(J + 1.0e-12)

    return jnp.array([lambda_x, lambda_y, log_J])


# Vectorized versions over arbitrary leading batch dimensions.
_strain_to_moekan_inputs_vmap = jax.vmap(_strain_to_moekan_inputs_single)
_d_moekan_inputs_d_strain_vmap = jax.vmap(
    jax.jacrev(_strain_to_moekan_inputs_single)
)


def strain_to_moekan_inputs(strain_history):
    """
    Transforms a Green-Lagrange strain history into MOEKAN inputs (JAX).

    Args:
        strain_history (jnp.ndarray): array of shape (..., 3) with the
            Green-Lagrange strain in Voigt form [E11, E22, 2*E12].

    Returns:
        tuple:
            - jnp.ndarray: array of shape (..., 3) with
                [lambda_x, lambda_y, log(J)].
            - jnp.ndarray: Jacobian d(moekan_inputs)/d(strain) of shape
                (..., 3, 3), where the last two axes are
                (output_component, input_component).
    """
    leading_shape = strain_history.shape[:-1]
    strain_flat = strain_history.reshape(-1, 3)

    outputs = _strain_to_moekan_inputs_vmap(strain_flat)
    jacobian = _d_moekan_inputs_d_strain_vmap(strain_flat)

    outputs = outputs.reshape(*leading_shape, 3)
    jacobian = jacobian.reshape(*leading_shape, 3, 3)
    return outputs, jacobian
# =============================================================================


# =============================================================================
# LOAD DATASET
# =============================================================================
DATA_FILE = "data_labels_v1.npz"

print(f"Loading dataset from: {DATA_FILE}")
data = np.load(DATA_FILE, allow_pickle=True)

# --- Training data (label "fit") ---
E_fit = data["E_fit"]          # (4200, 3)  Green-Lagrange strain [E11, E22, 2*E12]
S_fit = data["S_fit"]          # (4200, 3)  Stress [S11, S22, S12] in Pa
W_fit = data["W_fit"]          # (4200,)    Work in Pa
D_fit = data["D_fit"]          # (4200, 3, 3) Tangent stiffness

# --- Test data (label "test") ---
E_test = data["E_test"]        # (512, 3)
S_test = data["S_test"]        # (512, 3)
W_test = data["W_test"]        # (512,)
D_test = data["D_test"]        # (512, 3, 3)

# Convert stress from Pa to MPa.
S_fit = S_fit / 1.0e6
S_test = S_test / 1.0e6

print(f"\nTraining samples: {E_fit.shape[0]}")
print(f"Test samples    : {E_test.shape[0]}")
print(f"Strain range (fit): E11 [{E_fit[:,0].min():.4f}, {E_fit[:,0].max():.4f}], "
      f"E22 [{E_fit[:,1].min():.4f}, {E_fit[:,1].max():.4f}], "
      f"2E12 [{E_fit[:,2].min():.4f}, {E_fit[:,2].max():.4f}]")
print(f"Stress range (fit, MPa): S11 [{S_fit[:,0].min():.2f}, {S_fit[:,0].max():.2f}], "
      f"S22 [{S_fit[:,1].min():.2f}, {S_fit[:,1].max():.2f}], "
      f"S12 [{S_fit[:,2].min():.2f}, {S_fit[:,2].max():.2f}]")

# =============================================================================


# =============================================================================
# TRANSFORM STRAIN -> MOEKAN INPUTS
# =============================================================================
# Training
moekan_inputs_fit, d_moekan_inputs_d_strain_fit = strain_to_moekan_inputs(
    jnp.asarray(E_fit, dtype=jnp.float32)
)

# Test
moekan_inputs_test, d_moekan_inputs_d_strain_test = strain_to_moekan_inputs(
    jnp.asarray(E_test, dtype=jnp.float32)
)

print(f"\nMOEKAN inputs (fit)  shape: {moekan_inputs_fit.shape}")
print(f"Jacobian (fit)       shape: {d_moekan_inputs_d_strain_fit.shape}")
print(f"MOEKAN inputs (test) shape: {moekan_inputs_test.shape}")
print(f"Jacobian (test)      shape: {d_moekan_inputs_d_strain_test.shape}")

# =============================================================================
# PHYSICS-PRESERVING NORMALIZATION (max-value scaling, NO centering)
# =============================================================================
"""
We normalize by dividing by the maximum absolute value of each quantity
computed over the FIT (training) set. No mean-centering is applied so that
the physics is preserved:
    - the stretches lambda_x, lambda_y stay positive,
    - J = lambda_x * lambda_y stays positive (log(J) keeps its sign),
    - the signs of the stress components are kept.

MOEKAN inputs:
    u_norm[i] = u[i] / s_u[i]
where s_u[i] = max |u[i]| over the fit set.

Chain-rule Jacobian (so the physics is preserved):
    The physical stress is  S = dW/dE = dW/du @ du/dE.
    With normalized inputs u_norm = u / s_u, the network gradient is
    dW/du_norm. To recover the physical stress we need
        S = dW/du_norm @ (du_norm/dE)
    where  du_norm/dE = diag(1/s_u) @ du/dE.
    Hence each row i of the Jacobian is scaled by 1/s_u[i]:
        J_norm[i, j] = J[i, j] / s_u[i]
    and the chain rule  dW/du_norm @ J_norm  yields the physical stress.

Stress targets:
    S_norm = S / s_S,  s_S[i] = max |S[i]| over the fit set.
    The loss compares the normalized physical stress (chain rule / s_S)
    against S_norm, so the loss is O(1).
"""
# Per-component max-abs scales from the FIT set (physics-preserving, no centering).
s_u = jnp.max(jnp.abs(moekan_inputs_fit), axis=0)          # (3,)
s_S = jnp.max(jnp.abs(jnp.asarray(S_fit, dtype=jnp.float32)), axis=0)  # (3,)

# Guard against a zero scale (should not happen for this dataset).
s_u = jnp.where(s_u < 1.0e-12, 1.0, s_u)
s_S = jnp.where(s_S < 1.0e-12, 1.0, s_S)

# Normalized MOEKAN inputs.
moekan_inputs_fit_norm = moekan_inputs_fit / s_u[None, :]
moekan_inputs_test_norm = moekan_inputs_test / s_u[None, :]

# Normalized chain-rule Jacobians: scale row i by 1/s_u[i].
inv_s_u = 1.0 / s_u
d_inputs_d_strain_fit_norm = d_moekan_inputs_d_strain_fit * inv_s_u[:, None]
d_inputs_d_strain_test_norm = d_moekan_inputs_d_strain_test * inv_s_u[:, None]

# Normalized stress targets.
stress_ref_fit_norm = jnp.asarray(S_fit, dtype=jnp.float32) / s_S[None, :]

print(f"\nNormalization scales (max-abs, fit set):")
print(f"  s_u (MOEKAN inputs) : {np.asarray(s_u)}")
print(f"  s_S (stress, MPa)   : {np.asarray(s_S)}")
print(f"  MOEKAN inputs (fit)  range after norm: "
      f"[{np.asarray(moekan_inputs_fit_norm).min():.3f}, "
      f"{np.asarray(moekan_inputs_fit_norm).max():.3f}]")
print(f"  Stress (fit)        range after norm: "
      f"[{np.asarray(stress_ref_fit_norm).min():.3f}, "
      f"{np.asarray(stress_ref_fit_norm).max():.3f}]")

# =============================================================================


# =============================================================================
# MOEKAN TRAINING
# =============================================================================
"""
MOEKAN TRAINING:
    inputs  : MOEKAN inputs (N, 3)
    targets : reference stress in Voigt form [S11, S22, S12] (N, 3)
    loss    : L2 norm of the stress difference, where the predicted stress
              is obtained via the chain rule:
                  dW/dE = dW/d(moekan_inputs) @ d(moekan_inputs)/dE
"""
def stress_l2_loss(model, params, inputs, d_inputs_d_strain, stress_ref, s_S):
    """
    L2 loss on the stress (dW/dE) obtained via the chain rule.

        dW/dE = dW/d(moekan_inputs) @ d(moekan_inputs)/dE

    The chain rule with the NORMALIZED Jacobian (rows scaled by 1/s_u)
    yields the PHYSICAL stress in MPa. It is divided by the stress scale
    s_S so that it is compared against the normalized target (S / s_S),
    keeping the loss O(1).

    The shear component S12 is weighted 10x to compensate for its
    typically much smaller magnitude compared to S11 and S22.

    Args:
        model: MOEKAN model.
        params: model parameters.
        inputs: normalized MOEKAN inputs, shape (N, 3).
        d_inputs_d_strain: normalized Jacobian d(u_norm)/d(strain), shape (N, 3, 3).
        stress_ref: normalized reference stress [S11, S22, S12] / s_S, shape (N, 3).
        s_S: per-component stress scale (max-abs over fit set), shape (3,).

    Returns:
        Weighted mean squared error of the (normalized) stress.
    """
    # dW/d(moekan_inputs): gradient of the network output W w.r.t. its inputs.
    dW_d_inputs = model.gradient(inputs, params=params)  # (N, 3)

    # Chain rule: dW/dE = dW/d(u_norm) @ d(u_norm)/dE  ->  PHYSICAL stress (MPa).
    dW_d_strain = jnp.einsum(
        '...i,...ij->...j', dW_d_inputs, d_inputs_d_strain
    )

    # Normalize the physical stress to match the normalized target.
    dW_d_strain_norm = dW_d_strain / s_S[None, :]

    diff = dW_d_strain_norm - stress_ref

    # Component-wise weights: [S11, S22, S12]
    weights = jnp.array([1.0, 1.0, 10.0])
    return jnp.mean(weights * diff ** 2)


def train_model(
    model,
    inputs,
    d_inputs_d_strain,
    stress_ref,
    s_S,
    learning_rate=1e-3,
    epochs=150_000,
    patience=1e-5,
):
    device = jax.devices("cpu")[0]
    print(f"Using JAX device: {device}")

    inputs = jax.device_put(inputs, device)
    d_inputs_d_strain = jax.device_put(d_inputs_d_strain, device)
    stress_ref = jax.device_put(stress_ref, device)
    s_S = jax.device_put(s_S, device)
    params = jax.device_put(model.params, device)

    optimizer = optax.adam(
        learning_rate=learning_rate,
    )

    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state):
        def loss_fn(current_params):
            return stress_l2_loss(
                model,
                current_params,
                inputs,
                d_inputs_d_strain,
                stress_ref,
                s_S,
            )

        loss_value, gradients = jax.value_and_grad(
            loss_fn
        )(params)

        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            params,
            value=loss_value,
            grad=gradients,
            value_fn=loss_fn,
        )

        params = optax.apply_updates(
            params,
            updates,
        )

        return params, optimizer_state, loss_value

    for epoch in range(1, epochs + 1):
        params, optimizer_state, loss_value = train_step(
            params,
            optimizer_state,
        )

        if epoch == 1 or epoch % 1000 == 0:
            loss_float = float(loss_value)

            print(
                f"Epoch {epoch}/{epochs}, "
                f"loss={loss_float:.6e}"
            )

            if loss_float < patience:
                print(
                    f"Early stopping at epoch {epoch}"
                )
                break

    model.params = params
    return model


# Prepare flattened inputs and targets for training (NORMALIZED).
# The model is trained on normalized MOEKAN inputs and normalized stress
# targets. The chain-rule Jacobian is the normalized one (rows scaled by
# 1/s_u) so that  dW/du_norm @ J_norm  gives the physical stress, which is
# then compared against the normalized stress target in the loss.
stress_ref_fit = stress_ref_fit_norm

print(f"\nFlattened MOEKAN inputs shape: {moekan_inputs_fit_norm.shape}")
print(f"Flattened Jacobian shape     : {d_inputs_d_strain_fit_norm.shape}")
print(f"Flattened stress shape       : {stress_ref_fit.shape}")

# Create the MOEKAN model: 3 inputs, n hidden, 1 output (W).
model = MOEKAN(
    width=(3, 15, 8, 1),
    temperature=1.0,
)

print(
    "Trainable parameters:",
    model.parameter_count(),
)

# Train the model.
print("\nLaunching the training of MOEKAN...")
model = train_model(
    model,
    moekan_inputs_fit_norm,
    d_inputs_d_strain_fit_norm,
    stress_ref_fit,
    s_S,
    learning_rate=1e-2,
    epochs=100_000,
    patience=1e-5,
)

# Final loss on training data (normalized).
final_loss = float(
    stress_l2_loss(
        model,
        model.params,
        moekan_inputs_fit_norm,
        d_inputs_d_strain_fit_norm,
        stress_ref_fit,
        s_S,
    )
)
print(f"\nFinal stress L2 loss (fit, normalized): {final_loss:.6e}")

# =============================================================================


# =============================================================================
# PLOTS: evaluate on TEST data
# =============================================================================
"""
PLOTS: evolution of each stress component against each strain component
       on the TEST set (512 samples).
"""
# Reference strain components: [E11, E22, 2*E12].
Ex_test = E_test[:, 0]        # E11
Ey_test = E_test[:, 1]        # E22
Gamma_xy_test = E_test[:, 2]  # 2*E12 (engineering shear)

# Reference stress components: [S11, S22, S12].
S11_ref_test = S_test[:, 0]
S22_ref_test = S_test[:, 1]
S12_ref_test = S_test[:, 2]

# Predicted stress via the chain rule using the NORMALIZED inputs and the
# normalized Jacobian (rows scaled by 1/s_u). This yields the PHYSICAL stress
# in MPa:  S = dW/du_norm @ (du_norm/dE).
dW_d_inputs_test = model.gradient(moekan_inputs_test_norm, params=model.params)  # (N, 3)
dW_d_strain_test = jnp.einsum(
    '...i,...ij->...j', dW_d_inputs_test, d_inputs_d_strain_test_norm
)
S11_pred_test = np.asarray(dW_d_strain_test[:, 0])
S22_pred_test = np.asarray(dW_d_strain_test[:, 1])
S12_pred_test = np.asarray(dW_d_strain_test[:, 2])

# --- Plot 1: Stress vs E11 ---
stress_components = [
    (r"$S_{11}$", S11_ref_test, S11_pred_test, Ex_test, r"$E_{xx}$"),
    (r"$S_{22}$", S22_ref_test, S22_pred_test, Ex_test, r"$E_{xx}$"),
    (r"$S_{12}$", S12_ref_test, S12_pred_test, Ex_test, r"$E_{xx}$"),
]

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, (label, ref_values, pred_values, x_values, x_label) in zip(axes, stress_components):
    axis.scatter(
        x_values,
        ref_values,
        s=8,
        alpha=0.35,
        color="black",
        label="reference (test)",
    )
    axis.scatter(
        x_values,
        pred_values,
        s=8,
        alpha=0.35,
        color="tab:blue",
        label="MOEKAN (test)",
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(label)
    axis.set_title(rf"{label} versus $E_{{xx}}$ (test)")
    axis.grid(True)
    axis.legend()

figure.tight_layout()
figure.savefig(
    "MOEKAN_data_labels_b_stress_vs_E11_test.pdf",
    bbox_inches="tight",
)
plt.show()

# --- Plot 2: Stress vs E22 ---
stress_components_ey = [
    (r"$S_{11}$", S11_ref_test, S11_pred_test, Ey_test, r"$E_{yy}$"),
    (r"$S_{22}$", S22_ref_test, S22_pred_test, Ey_test, r"$E_{yy}$"),
    (r"$S_{12}$", S12_ref_test, S12_pred_test, Ey_test, r"$E_{yy}$"),
]

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, (label, ref_values, pred_values, x_values, x_label) in zip(axes, stress_components_ey):
    axis.scatter(
        x_values,
        ref_values,
        s=8,
        alpha=0.35,
        color="black",
        label="reference (test)",
    )
    axis.scatter(
        x_values,
        pred_values,
        s=8,
        alpha=0.35,
        color="tab:blue",
        label="MOEKAN (test)",
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(label)
    axis.set_title(rf"{label} versus $E_{{yy}}$ (test)")
    axis.grid(True)
    axis.legend()

figure.tight_layout()
figure.savefig(
    "MOEKAN_data_labels_b_stress_vs_E22_test.pdf",
    bbox_inches="tight",
)
plt.show()

# --- Plot 3: Stress vs 2*E12 (shear) ---
stress_components_gamma = [
    (r"$S_{11}$", S11_ref_test, S11_pred_test, Gamma_xy_test, r"$2E_{xy}$"),
    (r"$S_{22}$", S22_ref_test, S22_pred_test, Gamma_xy_test, r"$2E_{xy}$"),
    (r"$S_{12}$", S12_ref_test, S12_pred_test, Gamma_xy_test, r"$2E_{xy}$"),
]

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, (label, ref_values, pred_values, x_values, x_label) in zip(axes, stress_components_gamma):
    axis.scatter(
        x_values,
        ref_values,
        s=8,
        alpha=0.35,
        color="black",
        label="reference (test)",
    )
    axis.scatter(
        x_values,
        pred_values,
        s=8,
        alpha=0.35,
        color="tab:blue",
        label="MOEKAN (test)",
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(label)
    axis.set_title(rf"{label} versus $2E_{{xy}}$ (test)")
    axis.grid(True)
    axis.legend()

figure.tight_layout()
figure.savefig(
    "MOEKAN_data_labels_b_stress_vs_shear_test.pdf",
    bbox_inches="tight",
)
plt.show()

# --- Plot 4: Work vs strain (W vs E11) ---
# Predicted work: W_pred = model(moekan_inputs_test_norm). The model is trained
# on the normalized inputs, so it must be evaluated there. Its output is the
# physical strain energy W in MPa (consistent with the chain-rule stress).
W_pred_test = np.asarray(model(moekan_inputs_test_norm, params=model.params)).flatten()
W_ref_test = W_test / 1.0e6  # Convert to MPa

figure, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(Ex_test, W_ref_test, s=8, alpha=0.35, color="black", label="reference (test)")
axes[0].scatter(Ex_test, W_pred_test, s=8, alpha=0.35, color="tab:blue", label="MOEKAN (test)")
axes[0].set_xlabel(r"$E_{xx}$")
axes[0].set_ylabel(r"$W$ (MPa)")
axes[0].set_title(r"$W$ versus $E_{xx}$ (test)")
axes[0].grid(True)
axes[0].legend()

axes[1].scatter(Ey_test, W_ref_test, s=8, alpha=0.35, color="black", label="reference (test)")
axes[1].scatter(Ey_test, W_pred_test, s=8, alpha=0.35, color="tab:blue", label="MOEKAN (test)")
axes[1].set_xlabel(r"$E_{yy}$")
axes[1].set_ylabel(r"$W$ (MPa)")
axes[1].set_title(r"$W$ versus $E_{yy}$ (test)")
axes[1].grid(True)
axes[1].legend()

figure.tight_layout()
figure.savefig(
    "MOEKAN_data_labels_b_work_test.pdf",
    bbox_inches="tight",
)
plt.show()

# --- Plot 5: Predicted vs Reference (parity plots) ---
parity_components = [
    (r"$S_{11}$", S11_ref_test, S11_pred_test),
    (r"$S_{22}$", S22_ref_test, S22_pred_test),
    (r"$S_{12}$", S12_ref_test, S12_pred_test),
]

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, (label, ref_values, pred_values) in zip(axes, parity_components):
    axis.scatter(
        ref_values,
        pred_values,
        s=8,
        alpha=0.35,
        color="tab:blue",
    )
    # Diagonal line
    lims = [
        min(ref_values.min(), pred_values.min()) - 0.05 * (ref_values.max() - ref_values.min()),
        max(ref_values.max(), pred_values.max()) + 0.05 * (ref_values.max() - ref_values.min()),
    ]
    axis.plot(lims, lims, "k--", linewidth=1, label="parity")
    axis.set_xlabel(f"{label} reference (test)")
    axis.set_ylabel(f"{label} MOEKAN (test)")
    axis.set_title(f"{label} parity (test)")
    axis.grid(True)
    axis.legend()

figure.tight_layout()
figure.savefig(
    "MOEKAN_data_labels_b_parity_test.pdf",
    bbox_inches="tight",
)
plt.show()

# --- Print test metrics ---
print("\n" + "=" * 60)
print("TEST METRICS (stress, MPa)")
print("=" * 60)
for name, ref, pred in [
    ("S11", S11_ref_test, S11_pred_test),
    ("S22", S22_ref_test, S22_pred_test),
    ("S12", S12_ref_test, S12_pred_test),
]:
    mse = np.mean((pred - ref) ** 2)
    mae = np.mean(np.abs(pred - ref))
    max_err = np.max(np.abs(pred - ref))
    print(f"  {name}: MSE={mse:.6e}, MAE={mae:.6e}, MaxErr={max_err:.6e}")

print("\nDone.")
