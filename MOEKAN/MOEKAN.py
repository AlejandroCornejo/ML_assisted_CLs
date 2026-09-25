import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax


NUM_EXPERTS = 5


def moekan_edge(x, edge_params, temperature):
    """Evaluate one or more MOEKAN edges."""
    a = edge_params["a"]
    b = edge_params["b"]
    c = edge_params["c"]
    d = edge_params["d"]
    w = edge_params["w"]

    z = a * x + b

    probabilities = jax.nn.softmax(w / temperature, axis=-1)

    basis = jnp.stack(
        [
            z,
            z**2,
            z**3,
            jnp.tanh(z),
            jnp.sin(z),
        ],
        axis=-1,
    )

    return jnp.sum(
        probabilities * (c * basis + d),
        axis=-1,
    )


def moekan_layer(layer_params, x, temperature):
    """
    Apply a KAN-like MOEKAN layer.

    x shape:
        (..., input_width)

    Output shape:
        (..., output_width)

    Each output node receives one MOEKAN edge from every input node.
    """
    a = layer_params["a"]
    b = layer_params["b"]
    c = layer_params["c"]
    d = layer_params["d"]
    w = layer_params["w"]

    # Parameters have shape:
    # a, b, c, d: (output_width, input_width)
    # w:          (output_width, input_width, NUM_EXPERTS)

    z = (
        x[..., None, :]
        * a
        + b
    )

    probabilities = jax.nn.softmax(w / temperature, axis=-1)

    basis = jnp.stack(
        [
            z,
            z**2,
            z**3,
            jnp.tanh(z),
            jnp.sin(z),
        ],
        axis=-1,
    )

    edge_values = probabilities * (
        c[..., None] * basis + d[..., None]
    )

    # Sum over all input edges connected to each output node.
    return jnp.sum(edge_values, axis=(-1, -2))


def moekan_network(params, x, temperature):
    """Evaluate the multilayer MOEKAN network."""
    activations = x

    for layer_params in params:
        activations = moekan_layer(
            layer_params,
            activations,
            temperature,
        )

    return activations


def initialize_layer(input_width, output_width, key):
    """
    Initialize all MOEKAN edges in one layer.

    Small random perturbations are used to break symmetry between
    hidden neurons.
    """
    key_a, key_b, key_c, key_d, key_w = jax.random.split(key, 5)

    return {
        "a": (
            1.0
            + 0.01
            * jax.random.normal(
                key_a,
                (output_width, input_width),
            )
        ),
        "b": 0.01 * jax.random.normal(
            key_b,
            (output_width, input_width),
        ),
        "c": (
            1.0
            + 0.01
            * jax.random.normal(
                key_c,
                (output_width, input_width),
            )
        ),
        "d": 0.01 * jax.random.normal(
            key_d,
            (output_width, input_width),
        ),
        "w": 0.01 * jax.random.normal(
            key_w,
            (output_width, input_width, NUM_EXPERTS),
        ),
    }


def initialize_network(width, seed=42):
    """
    Initialize a KAN-like network.

    Example:
        width=[1, 8, 8, 1]
    """
    if len(width) < 2:
        raise ValueError("width must contain at least input and output sizes")

    key = jax.random.PRNGKey(seed)
    layer_keys = jax.random.split(key, len(width) - 1)

    return [
        initialize_layer(
            input_width=width[layer_index],
            output_width=width[layer_index + 1],
            key=layer_keys[layer_index],
        )
        for layer_index in range(len(width) - 1)
    ]


def relative_mse_loss(params, x, y, temperature):
    prediction = moekan_network(params, x, temperature)

    return jnp.mean((prediction - y) ** 2) / (
        jnp.mean(y**2) + 1e-12
    )


def train_model(
    x,
    y,
    width=(1, 8, 8, 1),
    temperature=1.0,
    learning_rate=1e-3,
    epochs=10_000,
    patience=1e-6,
    seed=42,
):
    """Train the multilayer MOEKAN network."""
    devices = jax.devices("cpu")

    if not devices:
        raise RuntimeError("No JAX CPU device was found.")

    device = devices[0]
    print(f"Using JAX device: {device}")

    x = jax.device_put(x, device)
    y = jax.device_put(y, device)

    params = initialize_network(width, seed=seed)
    params = jax.device_put(params, device)

    optimizer = optax.adamw(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state, x, y):
        loss_value, gradients = jax.value_and_grad(
            relative_mse_loss
        )(
            params,
            x,
            y,
            temperature,
        )

        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            params,
        )

        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss_value

    for epoch in range(1, epochs + 1):
        params, optimizer_state, loss_value = train_step(
            params,
            optimizer_state,
            x,
            y,
        )

        if epoch == 1 or epoch % 1000 == 0:
            loss_float = float(loss_value)

            print(
                f"Epoch {epoch}/{epochs} "
                f"loss={loss_float:.6e}"
            )

            if loss_float < patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"with loss={loss_float:.6e}"
                )
                break

    return params


def main():
    x = np.linspace(-1.0, 1.0, 500)
    y = np.sin(4.0 * x) + np.log(x**4 + 5.0)

    x_jax = jnp.asarray(x[:, None], dtype=jnp.float32)
    y_jax = jnp.asarray(y[:, None], dtype=jnp.float32)

    width = [1, 5, 5, 1]

    temperature = 1.0e-1
    params = train_model(
        x_jax,
        y_jax,
        width=width,
        temperature=temperature,
        learning_rate=1e-4,
        epochs=100_000,
        patience=1e-6,
    )

    prediction = moekan_network(
        params,
        x_jax,
        temperature,
    )

    prediction = np.asarray(prediction).squeeze()
    final_mse = np.mean((prediction - y) ** 2)

    print(f"Final MSE: {final_mse:.6e}")

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="reference")
    plt.plot(
        x,
        prediction,
        "--",
        label="multilayer MOEKAN",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("multilayer_moekan_result.pdf")
    plt.show()


if __name__ == "__main__":
    main()