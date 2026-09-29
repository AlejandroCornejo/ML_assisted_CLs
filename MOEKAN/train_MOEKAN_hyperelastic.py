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
def strain_to_moekan_inputs(strain_history):
    """
    Transforms a Green-Lagrange strain history into MOEKAN inputs (JAX).

    Args:
        strain_history (jnp.ndarray): array of shape (..., 3) with the
            Green-Lagrange strain in Voigt form [E11, E22, 2*E12].

    Returns:
        jnp.ndarray: array of shape (..., 4) with
            [lambda_x, lambda_y, lambda_x * lambda_y, log(J)].
    """
    # Reconstruct the symmetric Green-Lagrange strain tensor E.
    # NOTE: the 3rd Voigt component is the engineering shear strain 2*E12.
    E11 = strain_history[..., 0]
    E22 = strain_history[..., 1]
    E12 = 0.5 * strain_history[..., 2]

    # Right Cauchy-Green deformation tensor: C = I + 2E.
    C = jnp.stack(
        [
            jnp.stack([1.0 + 2.0 * E11, E12], axis=-1),
            jnp.stack([E12, 1.0 + 2.0 * E22], axis=-1),
        ],
        axis=-2,
    )

    # Principal stretches of F: square roots of the eigenvalues of C.
    # eigvalsh returns the eigenvalues in ascending order for symmetric matrices.
    eigvals = jnp.linalg.eigvalsh(C)
    # eigvals = jnp.clip(eigvals, a_min=1.0e-12)  # guard against non-physical values
    lambdas = jnp.sqrt(eigvals)

    lambda_x = lambdas[..., 0]
    lambda_y = lambdas[..., 1]

    # J = sqrt(det(C)) = product of the principal stretches.
    J = lambda_x * lambda_y
    log_J = jnp.log(J)

    return jnp.stack(
        [lambda_x, lambda_y, J, log_J], axis=-1
    )
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
moekan_inputs = strain_to_moekan_inputs(
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
    inputs  : flattened MOEKAN inputs (batches*steps, 4)
    targets : work W (batches*steps, 1)
    loss    : relative L2 norm of the squared difference of work
"""
def relative_l2_loss(model, params, inputs, targets):
    """
    Relative L2 norm of the squared difference of work.

        loss = ||(W_pred - W_ref)^2||_2 / (||W_ref^2||_2 + eps)
             = sqrt(sum((W_pred - W_ref)^2)) / (sqrt(sum(W_ref^2)) + eps)
    """
    prediction = model(inputs, params=params)
    diff = prediction - targets
    numerator = jnp.sqrt(jnp.sum(diff ** 2))
    denominator = jnp.sqrt(jnp.sum(targets ** 2)) + 1.0e-12
    return numerator / denominator


def train_model(
    model,
    inputs,
    targets,
    learning_rate=1e-3,
    epochs=50_000,
    patience=1e-7,
):
    device = jax.devices("cpu")[0]
    print(f"Using JAX device: {device}")

    inputs = jax.device_put(inputs, device)
    targets = jax.device_put(targets, device)
    params = jax.device_put(model.params, device)

    optimizer = optax.adam( # adamw
        learning_rate=learning_rate,
    )

    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state):
        def loss_fn(current_params):
            return relative_l2_loss(
                model,
                current_params,
                inputs,
                targets,
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

moekan_inputs_flat = moekan_inputs.reshape(n_batches * n_steps, 4)
work_flat = jnp.asarray(
    ref_work_database.reshape(n_batches * n_steps, 1).numpy(),
    dtype=jnp.float32,
)

print("Flattened MOEKAN inputs shape: ", moekan_inputs_flat.shape)
print("Flattened work shape         : ", work_flat.shape)

# Create the MOEKAN model: 4 inputs, n hidden, 1 output (W).
model = MOEKAN(
    width=(4, 1, 1),
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
    work_flat,
    learning_rate=1e-4,
    epochs=50_000,
    patience=1e-7,
)

# Final loss.
final_loss = float(
    relative_l2_loss(
        model,
        model.params,
        moekan_inputs_flat,
        work_flat,
    )
)
print(f"Final relative L2 loss: {final_loss:.6e}")

# ==========================================================================================
"""
PLOTS: evolution of W against each strain component (Ex, Ey, Gamma_xy)
"""
# Reference strain components (batch x steps x 3): [E11, E22, 2*E12].
strain_np = ref_strain_database.numpy()
Ex = strain_np[..., 0].reshape(-1)        # E11
Ey = strain_np[..., 1].reshape(-1)        # E22
Gamma_xy = strain_np[..., 2].reshape(-1)  # 2*E12 (engineering shear)

W_ref = np.asarray(work_flat).reshape(-1)
W_pred = np.asarray(model(moekan_inputs_flat)).reshape(-1)

strain_components = [
    (r"$E_{xx}$", Ex),
    (r"$E_{yy}$", Ey),
    (r"$\Gamma_{xy}$", Gamma_xy),
]

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, (label, strain_values) in zip(axes, strain_components):
    axis.scatter(
        strain_values,
        W_ref,
        s=8,
        alpha=0.35,
        color="black",
        label="reference",
    )
    axis.scatter(
        strain_values,
        W_pred,
        s=8,
        alpha=0.35,
        color="tab:blue",
        label="MOEKAN",
    )
    axis.set_xlabel(label)
    axis.set_ylabel(r"$W$")
    axis.set_title(rf"$W$ versus {label}")
    axis.grid(True)
    axis.legend()

figure.tight_layout()
figure.savefig(
    "MOEKAN_hyperelastic_W_vs_strain.pdf",
    bbox_inches="tight",
)
plt.show()

# ==========================================================================================

