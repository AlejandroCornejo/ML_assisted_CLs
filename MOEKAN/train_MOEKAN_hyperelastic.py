import numpy as np
import matplotlib.pyplot as plt
import torch as torch

import jax
import jax.numpy as jnp
import optax

from MOEKAN import MOEKAN

import cl_loader as cl_loader


#=============================================================================================================
"""
TRANSFORMATION FROM GREEN-LAGRANGE STRAIN TO MOEKAN INPUTS:

    strain (batches, steps, 3)  ->  moekan_inputs (batches, steps, 4)

The strain history is the 2D Green-Lagrange strain in Voigt form
[E11, E22, 2*E12] (Kratos GREEN_LAGRANGE_STRAIN_VECTOR convention).

The MOEKAN inputs are:
    [lambda_x, lambda_y, lambda_x * lambda_y, log(J)]

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
    # Reconstruct the symmetric Green-Lagrange strain tensor E.
    # NOTE: the 3rd Voigt component is the engineering shear strain 2*E12.
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
    # eigvalsh returns the eigenvalues in ascending order for symmetric matrices.
    eigvals = jnp.linalg.eigvalsh(C)
    # eigvals = jnp.clip(eigvals, a_min=1.0e-12)  # guard against non-physical values
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
    # vmap maps over the leading axis, so flatten to 2D (N, 3) first and
    # reshape the results back to the original leading shape.
    leading_shape = strain_history.shape[:-1]
    strain_flat = strain_history.reshape(-1, 3)

    outputs = _strain_to_moekan_inputs_vmap(strain_flat)
    jacobian = _d_moekan_inputs_d_strain_vmap(strain_flat)

    outputs = outputs.reshape(*leading_shape, 3)
    jacobian = jacobian.reshape(*leading_shape, 3, 3)
    return outputs, jacobian
#=============================================================================================================


#=============================================================================================================
"""
INPUT DATASET:
"""
number_of_steps = 25
ADD_NOISE = False
database = cl_loader.CustomDataset("raw_data", number_of_steps, None, ADD_NOISE)
#=============================================================================================================

ref_strain_database = torch.stack([item[0] for item in database]) # batch x steps x strain_size
ref_stress_database = torch.stack([item[1] for item in database]) # batch x steps x strain_size
ref_work_database   = torch.stack([item[2] for item in database]) # batch x steps x 1

ref_stress_database /= 1.0e6 # to MPa
ref_work_database   /= 1.0e6 # to MPa vs strain

# Transform the Green-Lagrange strain history (batch x steps x 3) into the
# MOEKAN inputs (batch x steps x 4): [lambda_x, lambda_y, lambda_x*lambda_y, log(J)]
# The MOEKAN model is JAX-based, so convert the torch tensor to a JAX array first.
moekan_inputs, d_moekan_inputs_d_strain = strain_to_moekan_inputs(
    jnp.asarray(ref_strain_database.numpy(), dtype=jnp.float32)
)


print("\nLaunching the training of a KAN...")
print("Number of training batches: ", ref_strain_database.shape[0])
print("Number of total batches: ", ref_strain_database.shape[0])
print("Number of steps  : ", ref_strain_database.shape[1])
print("Strain size      : ", ref_strain_database.shape[2])

print("\nThe MOEKAN inputs size is: ", moekan_inputs.shape)


# ==========================================================================================
"""
MOEKAN TRAINING:
    inputs  : flattened MOEKAN inputs (batches*steps, 3)
    targets : reference stress in Voigt form [S11, S22, S12] (batches*steps, 3)
    loss    : L2 norm of the stress difference, where the predicted stress
              is obtained via the chain rule:
                  dW/dE = dW/d(moekan_inputs) @ d(moekan_inputs)/dE
"""
def stress_l2_loss(model, params, inputs, d_inputs_d_strain, stress_ref):
    """
    L2 loss on the stress (dW/dE) obtained via the chain rule.

        dW/dE = dW/d(moekan_inputs) @ d(moekan_inputs)/dE

    Args:
        model: MOEKAN model.
        params: model parameters.
        inputs: MOEKAN inputs, shape (N, 3).
        d_inputs_d_strain: Jacobian d(moekan_inputs)/d(strain), shape (N, 3, 3).
        stress_ref: reference stress in Voigt form [S11, S22, S12], shape (N, 3).

    Returns:
        Mean squared error of the stress.
    """
    # dW/d(moekan_inputs): gradient of the network output W w.r.t. its inputs.
    dW_d_inputs = model.gradient(inputs, params=params)  # (N, 3)

    # Chain rule: dW/dE = dW/d(moekan_inputs) @ d(moekan_inputs)/dE.
    # dW_d_inputs: (N, 3)  ->  (N, i)
    # d_inputs_d_strain: (N, 3, 3)  ->  (N, i, j)
    # result: (N, 3)  ->  (N, j)
    dW_d_strain = jnp.einsum(
        '...i,...ij->...j', dW_d_inputs, d_inputs_d_strain
    )

    diff = dW_d_strain - stress_ref
    return jnp.mean(diff ** 2)


def train_model(
    model,
    inputs,
    d_inputs_d_strain,
    stress_ref,
    learning_rate=1e-3,
    epochs=50_000,
    patience=1e-7,
):
    device = jax.devices("cpu")[0]
    print(f"Using JAX device: {device}")

    inputs = jax.device_put(inputs, device)
    d_inputs_d_strain = jax.device_put(d_inputs_d_strain, device)
    stress_ref = jax.device_put(stress_ref, device)
    params = jax.device_put(model.params, device)

    optimizer = optax.adamw( # adamw
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


# Prepare flattened inputs and targets.
n_batches = moekan_inputs.shape[0]
n_steps = moekan_inputs.shape[1]

moekan_inputs_flat = moekan_inputs.reshape(n_batches * n_steps, 3)
d_inputs_d_strain_flat = d_moekan_inputs_d_strain.reshape(
    n_batches * n_steps, 3, 3
)
stress_ref_flat = jnp.asarray(
    ref_stress_database.reshape(n_batches * n_steps, 3).numpy(),
    dtype=jnp.float32,
)

print("Flattened MOEKAN inputs shape: ", moekan_inputs_flat.shape)
print("Flattened Jacobian shape     : ", d_inputs_d_strain_flat.shape)
print("Flattened stress shape       : ", stress_ref_flat.shape)

# Create the MOEKAN model: 3 inputs, n hidden, 1 output (W).
model = MOEKAN(
    width=(3, 8, 4, 1),
    temperature=1.0,
)

print(
    "Trainable parameters:",
    model.parameter_count(),
)

# Train the model.
model = train_model(
    model,
    moekan_inputs_flat,
    d_inputs_d_strain_flat,
    stress_ref_flat,
    learning_rate=1e-3,
    epochs=500_000,
    patience=1e-4,
)

# Final loss.
final_loss = float(
    stress_l2_loss(
        model,
        model.params,
        moekan_inputs_flat,
        d_inputs_d_strain_flat,
        stress_ref_flat,
    )
)
print(f"Final stress L2 loss: {final_loss:.6e}")

# ==========================================================================================
"""
PLOTS: evolution of each stress component against each strain component
"""
# Reference strain components (batch x steps x 3): [E11, E22, 2*E12].
strain_np = ref_strain_database.numpy()
Ex = strain_np[..., 0].reshape(-1)        # E11
Ey = strain_np[..., 1].reshape(-1)        # E22
Gamma_xy = strain_np[..., 2].reshape(-1)  # 2*E12 (engineering shear)

# Reference stress components (batch x steps x 3): [S11, S22, S12].
stress_np = ref_stress_database.numpy()
S11_ref = stress_np[..., 0].reshape(-1)
S22_ref = stress_np[..., 1].reshape(-1)
S12_ref = stress_np[..., 2].reshape(-1)

# Predicted stress via the chain rule: dW/dE = dW/d(moekan_inputs) @ d(moekan_inputs)/dE.
dW_d_inputs = model.gradient(moekan_inputs_flat, params=model.params)  # (N, 3)
dW_d_strain = jnp.einsum(
    '...i,...ij->...j', dW_d_inputs, d_inputs_d_strain_flat
)
S11_pred = np.asarray(dW_d_strain[..., 0]).reshape(-1)
S22_pred = np.asarray(dW_d_strain[..., 1]).reshape(-1)
S12_pred = np.asarray(dW_d_strain[..., 2]).reshape(-1)

stress_components = [
    (r"$S_{11}$", S11_ref, S11_pred),
    (r"$S_{22}$", S22_ref, S22_pred),
    (r"$S_{12}$", S12_ref, S12_pred),
]

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, (label, ref_values, pred_values) in zip(axes, stress_components):
    axis.scatter(
        Ex,
        ref_values,
        s=8,
        alpha=0.35,
        color="black",
        label="reference",
    )
    axis.scatter(
        Ex,
        pred_values,
        s=8,
        alpha=0.35,
        color="tab:blue",
        label="MOEKAN",
    )
    axis.set_xlabel(r"$E_{xx}$")
    axis.set_ylabel(label)
    axis.set_title(rf"{label} versus $E_{{xx}}$")
    axis.grid(True)
    axis.legend()

figure.tight_layout()
figure.savefig(
    "MOEKAN_hyperelastic_stress_vs_strain.pdf",
    bbox_inches="tight",
)
plt.show()

# ==========================================================================================

