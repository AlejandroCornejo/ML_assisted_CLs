import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from MOEKAN import MOEKAN


def relative_mse_loss(model, params, inputs, targets, ref_mse):
    prediction = model(
        inputs,
        params=params,
    )

    return jnp.mean((prediction - targets) ** 2) / ref_mse


def train_model(
    model,
    inputs,
    targets,
    learning_rate=1e-3,
    epochs=50_000,
    patience=1e-7,
    mse_ref=1.0,
):
    device = jax.devices("cpu")[0]
    print(f"Using JAX device: {device}")

    inputs = jax.device_put(inputs, device)
    targets = jax.device_put(targets, device)
    params = jax.device_put(model.params, device)

    optimizer = optax.adam( # lbfgs
        learning_rate=learning_rate,
    )

    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state):
        def loss_fn(current_params):
            return relative_mse_loss(
                model,
                current_params,
                inputs,
                targets,
                mse_ref,
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


def main():
    # Generate a two-dimensional training grid.
    x_values = np.linspace(0.05, 3.0, 80)
    y_values = np.linspace(0.05, 3.0, 80)

    x_grid, y_grid = np.meshgrid(
        x_values,
        y_values,
        indexing="ij",
    )

    # Target function: z = x * y.
    # z_grid = x_grid * y_grid # Case A
    z_grid = x_grid / y_grid # Case A
    
    mse_ref = jnp.mean(z_grid**2) + 1e-12

    # Flatten the grid into samples.
    inputs = np.column_stack(
        [
            x_grid.reshape(-1),
            y_grid.reshape(-1),
        ]
    )

    targets = z_grid.reshape(-1, 1)

    inputs_jax = jnp.asarray(
        inputs,
        dtype=jnp.float32,
    )

    targets_jax = jnp.asarray(
        targets,
        dtype=jnp.float32,
    )

    model = MOEKAN(
        # width=(2, 2, 1), # Case A
        width=(2, 3, 2, 1), # Case B
        temperature=1.0
    )

    print(
        "Trainable parameters:",
        model.parameter_count(),
    )

    model = train_model(
        model,
        inputs_jax,
        targets_jax,
        learning_rate=1e-3,
        epochs=200_000,
        patience=1e-5,
        mse_ref=mse_ref
    )

    prediction = model(inputs_jax)
    prediction = np.asarray(
        prediction
    ).reshape(x_grid.shape)

    final_mse = jnp.mean((prediction - z_grid) ** 2) / mse_ref

    print(f"Final MSE: {final_mse:.6e}")

    # Plot reference and predicted surfaces.
    figure = plt.figure(figsize=(14, 6))

    reference_axis = figure.add_subplot(
        1,
        2,
        1,
        projection="3d",
    )

    prediction_axis = figure.add_subplot(
        1,
        2,
        2,
        projection="3d",
    )

    reference_axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )

    reference_axis.set_title(
        r"Reference: $z=x y$"
    )
    reference_axis.set_xlabel("x")
    reference_axis.set_ylabel("y")
    reference_axis.set_zlabel("z")

    prediction_axis.plot_surface(
        x_grid,
        y_grid,
        prediction,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )

    prediction_axis.set_title(
        "MOEKAN prediction"
    )
    prediction_axis.set_xlabel("x")
    prediction_axis.set_ylabel("y")
    prediction_axis.set_zlabel("z")

    figure.tight_layout()
    figure.savefig(
        "MOEKAN_2d/moekan_2d_xy_surface.pdf",
        bbox_inches="tight",
    )
    plt.show()

    # Plot every learned edge function separately.
    model.plot_edge_functions(
        x=inputs_jax,
        filename="MOEKAN_2d/moekan_2d_xy_edge_functions.pdf",
        samples=300,
    )

    # Compare x versus z and y versus z.
    x_flat = inputs[:, 0]
    y_flat = inputs[:, 1]
    z_reference_flat = targets.reshape(-1)
    z_prediction_flat = prediction.reshape(-1)

    figure_2d, axes_2d = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    axes_2d[0].scatter(
        x_flat,
        z_reference_flat,
        s=8,
        alpha=0.35,
        color="black",
        label="reference",
    )

    axes_2d[0].scatter(
        x_flat,
        z_prediction_flat,
        s=8,
        alpha=0.35,
        color="tab:blue",
        label="MOEKAN",
    )

    axes_2d[0].set_xlabel("x")
    axes_2d[0].set_ylabel("z")
    axes_2d[0].set_title(r"$x$ versus $z$")
    axes_2d[0].grid(True)
    axes_2d[0].legend()

    axes_2d[1].scatter(
        y_flat,
        z_reference_flat,
        s=8,
        alpha=0.35,
        color="black",
        label="reference",
    )

    axes_2d[1].scatter(
        y_flat,
        z_prediction_flat,
        s=8,
        alpha=0.35,
        color="tab:orange",
        label="MOEKAN",
    )

    axes_2d[1].set_xlabel("y")
    axes_2d[1].set_ylabel("z")
    axes_2d[1].set_title(r"$y$ versus $z$")
    axes_2d[1].grid(True)
    axes_2d[1].legend()

    figure_2d.tight_layout()
    figure_2d.savefig(
        "MOEKAN_2d/moekan_2d_xz_yz.pdf",
        bbox_inches="tight",
    )

    model.log_expert_weights(
        filename="MOEKAN_2d/moekan_expert_weights.log"
    )


if __name__ == "__main__":
    main()