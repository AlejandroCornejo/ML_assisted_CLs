import numpy as np
import jax
import jax.numpy as jnp


class MOEKAN:
    def __init__(
        self,
        width,
        temperature=1.0,
        seed=42,
    ):
        if len(width) < 2:
            raise ValueError(
                "width must contain at least input and output sizes"
            )

        self.width = tuple(width)
        self.temperature = float(temperature)
        self.seed = seed
        self.num_experts = 5

        self.initialize()

    def initialize(self):
        self.params = self.initialize_network(
            self.width,
            seed=self.seed,
        )

    def initialize_layer(
        self,
        input_width,
        output_width,
        key,
    ):
        """Initialize all MOEKAN edges in one layer."""
        key_a, key_b, key_c, key_d, key_w = (
            jax.random.split(key, 5)
        )

        edge_shape = (
            output_width,
            input_width,
        )

        return {
            "a": (
                1.0
                + 0.01
                * jax.random.normal(
                    key_a,
                    edge_shape,
                )
            ),
            "b": 0.01 * jax.random.normal(
                key_b,
                edge_shape,
            ),
            "c": (
                1.0
                + 0.01
                * jax.random.normal(
                    key_c,
                    edge_shape,
                )
            ),
            "d": 0.01 * jax.random.normal(
                key_d,
                edge_shape,
            ),
            "w": 0.01 * jax.random.normal(
                key_w,
                (
                    output_width,
                    input_width,
                    self.num_experts,
                ),
            ),
        }

    def initialize_network(self, width, seed=42):
        """Initialize all layers of the MOEKAN network."""
        key = jax.random.PRNGKey(seed)

        layer_keys = jax.random.split(
            key,
            len(width) - 1,
        )

        return [
            self.initialize_layer(
                input_width=width[layer_index],
                output_width=width[layer_index + 1],
                key=layer_keys[layer_index],
            )
            for layer_index in range(len(width) - 1)
        ]

    def moekan_edge(self, x, edge_params):
        """Evaluate one or more MOEKAN edges."""
        a = edge_params["a"]
        b = edge_params["b"]
        c = edge_params["c"]
        d = edge_params["d"]
        w = edge_params["w"]

        z = a * x + b

        probabilities = jax.nn.softmax(
            w / self.temperature,
            axis=-1,
        )

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
            probabilities * (
                c * basis + d
            ),
            axis=-1,
        )

    def moekan_layer(self, layer_params, x):
        """
        Apply one MOEKAN layer.

        x shape:
            (..., input_width)

        output shape:
            (..., output_width)
        """
        a = layer_params["a"]
        b = layer_params["b"]
        c = layer_params["c"]
        d = layer_params["d"]
        w = layer_params["w"]

        z = x[..., None, :] * a + b

        probabilities = jax.nn.softmax(
            w / self.temperature,
            axis=-1,
        )

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
            c[..., None] * basis
            + d[..., None]
        )

        return jnp.sum(
            edge_values,
            axis=(-1, -2),
        )

    def moekan_network(self, x, params=None):
        """Evaluate the complete multilayer MOEKAN network."""
        if params is None:
            params = self.params

        activations = x

        for layer_params in params:
            activations = self.moekan_layer(
                layer_params,
                activations,
            )

        return activations

    def parameter_count(self):
        """Return the number of trainable scalar parameters."""
        return sum(
            value.size
            for layer_params in self.params
            for value in layer_params.values()
        )

    def __call__(self, x, params=None):
        return self.moekan_network(
            x,
            params=params,
        )