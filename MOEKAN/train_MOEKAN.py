import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from MOEKAN import MOEKAN


def relative_mse_loss(model, params, x, y):
    prediction = model(x, params=params)

    return jnp.mean((prediction - y) ** 2) / (
        jnp.mean(y**2) + 1e-12
    )


def train_model(
    model,
    x,
    y,
    learning_rate=1e-3,
    epochs=10_000,
    patience=1e-6,
):
    devices = jax.devices("cpu")
    device = devices[0]

    print(f"Using JAX device: {device}")

    x = jax.device_put(x, device)
    y = jax.device_put(y, device)

    params = jax.device_put(model.params, device)

    optimizer = optax.adamw(
        learning_rate=learning_rate
    )

    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state):
        def loss_fn(current_params):
            return relative_mse_loss(
                model,
                current_params,
                x,
                y,
            )

        loss_value, gradients = jax.value_and_grad(
            loss_fn
        )(params)

        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            params,
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
    x = np.linspace(-1.0, 1.0, 1500)
    y = np.sin(8.0 * x) + np.log(x**4 + 5.0) + 10.0

    x_jax = jnp.asarray(
        x[:, None],
        dtype=jnp.float32,
    )

    y_jax = jnp.asarray(
        y[:, None],
        dtype=jnp.float32,
    )

    model = MOEKAN(
        width=(1, 4, 2, 1),
        temperature=0.1,
        seed=42,
    )

    print(
        "Trainable parameters:",
        model.parameter_count(),
    )

    model = train_model(
        model,
        x_jax,
        y_jax,
        learning_rate=1e-4,
        epochs=1_000_000,
        patience=1e-6,
    )

    prediction = model(x_jax)
    prediction = np.asarray(prediction).squeeze()

    final_mse = np.mean((prediction - y) ** 2)
    print(f"Final MSE: {final_mse:.6e}")

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="reference")
    plt.plot(
        x,
        prediction,
        "--",
        label="MOEKAN",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        "multilayer_moekan_result.pdf"
    )
    
    model.plot_edge_tree(
        x=x_jax,
        filename="multilayer_moekan_edge_tree.pdf")

    plt.show()

if __name__ == "__main__":
    main()