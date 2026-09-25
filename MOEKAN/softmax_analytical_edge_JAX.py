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