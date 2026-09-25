import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax


class SoftMaxAnalyticalEdge:
    """
    JAX reimplementation of the original PyTorch model.

    We interpolate with five function candidates:
        x
        x**2
        x**3
        tanh(x)
        sin(x)

    f(x) = sum_i PI_i * (c_i * funct_i(a_i * x + b_i) + d_i)
    where PI is a softmax over the learned weights w_i.
    """

# -----------------------------------------------------------------
    def __init__(self, temperature=1.0):
        self.num_experts = 5
        self.temperature = float(temperature)

        self.params = {
            "a_i": jnp.array(1.0, dtype=jnp.float32),
            "b_i": jnp.array(0.0, dtype=jnp.float32),
            "c_i": jnp.array(1.0, dtype=jnp.float32),
            "d_i": jnp.array(0.0, dtype=jnp.float32),
            "w_i": jnp.ones((self.num_experts,), dtype=jnp.float32),
        }

# -----------------------------------------------------------------
    def get_expert_probabilities(self, params=None):
        params = self.params if params is None else params
        logits = params["w_i"] / self.temperature
        return jax.nn.softmax(logits, axis=0)

# -----------------------------------------------------------------
    def eval_functions(self, X, params=None):
        params = self.params if params is None else params
        PI = self.get_expert_probabilities(params)
        z = params["a_i"] * X + params["b_i"]

        return jnp.stack(
            [
                PI[0] * ((z) * params["c_i"] + params["d_i"]),          # x
                PI[1] * ((z ** 2) * params["c_i"] + params["d_i"]),     # x**2
                PI[2] * ((z ** 3) * params["c_i"] + params["d_i"]),     # x**3
                PI[3] * (jnp.tanh(z) * params["c_i"] + params["d_i"]),  # tanh(x)
                PI[4] * (jnp.sin(z) * params["c_i"] + params["d_i"]),   # sin(x)
            ], axis=0)

# -----------------------------------------------------------------
    def __call__(self, X, params=None): # forward method
        functs = self.eval_functions(X, params=params)
        return jnp.sum(functs, axis=0)

# ------------------------------ End class -----------------------------------
# ----------------------------------------------------------------------------



# -----------------------------------------------------------------
def relative_mse_loss(model, params, X, Y):
    pred = model(X, params=params)
    return jnp.mean((pred - Y) ** 2) / (jnp.mean(Y ** 2) + 1e-12)


# -----------------------------------------------------------------
def train_model(X, Y, temperature=1.0, lr=1e-3, epochs=10_000, patience=1e-6):
    gpu_devices = jax.devices("cpu") # gpu not supported in windows jax version, so using cpu for now
    if not gpu_devices:
        raise RuntimeError(
            "No JAX GPU device found. Install a CUDA-enabled JAX build "
            "and run with a supported NVIDIA CUDA environment."
        )

    device = gpu_devices[0]
    print(f"Using JAX device: {device}")

    X = jax.device_put(X, device)
    Y = jax.device_put(Y, device)

    model = SoftMaxAnalyticalEdge(temperature=temperature)
    params = jax.device_put(model.params, device)

    optimizer = optax.adamw(learning_rate=lr)
    opt_state = jax.device_put(optimizer.init(params), device)

    # -----------------------------------------------------------------
    @jax.jit
    def train_step(params, opt_state, X, Y):
        def loss_fn(current_params):
            return relative_mse_loss(model, current_params, X, Y)

        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_value

    for epoch in range(1, epochs + 1):
        params, opt_state, loss_value = train_step(
            params, opt_state, X, Y
        )

        # This synchronizes only periodically instead of every iteration.
        if epoch % 1000 == 0 or epoch == 1:
            loss_float = float(loss_value)
            print(f"Epoch {epoch}/{epochs} loss={loss_float:.6e}")

            if loss_float < patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"with loss={loss_float:.6e}"
                )
                break

    model.params = params
    return model


def main():
    X = np.linspace(-1.0, 1.0, 500)
    # Y = np.sin(3 * X) * np.log(X + 5)
    Y = np.sin(10 * X)

    X_j = jnp.asarray(X, dtype=jnp.float32)
    Y_j = jnp.asarray(Y, dtype=jnp.float32)

    model = train_model(X_j, Y_j,
                        patience=1e-4,
                        temperature=1.0e-0,
                        lr=1e-4,
                        epochs=1_000_000)

    PI = model.get_expert_probabilities()
    Y_pred = np.asarray(model(X_j))

    final_loss = float(np.mean((Y_pred - Y) ** 2))
    print("\nTrained parameters:")
    print("a_i:", np.asarray(model.params["a_i"]))
    print("b_i:", np.asarray(model.params["b_i"]))
    print("c_i:", np.asarray(model.params["c_i"]))
    print("d_i:", np.asarray(model.params["d_i"]))
    print("w_i:", np.asarray(model.params["w_i"]))
    print("Expert probability [X, X^2, x^3, tanh, sin] :", np.asarray(PI))
    print(f"\nFinal MSE (numpy): {final_loss:.6e}")

    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, label="reference")
    plt.plot(X, Y_pred, "--", label="MOEKAN prediction")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("JAX MOEKAN (1 edge)")
    out_file = "train_result_jax.pdf"
    plt.savefig(out_file)
    print(f"Saved comparison plot to {out_file}")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()