import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


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

    def plot_edge_tree(
        self,
        x,
        params=None,
        filename="moekan_edge_tree.pdf",
        samples=200,
        figsize=None,
    ):
        """
        Plot the multilayer MOEKAN architecture as a tree.

        Every connection is represented by a small inset plot showing its
        learned analytical edge function over the data range encountered by
        that edge.

        Parameters
        ----------
        x : array-like
            Input data with shape (n_samples, input_width).

        params : pytree, optional
            Network parameters. If omitted, self.params is used.

        filename : str
            Output PDF filename.

        samples : int
            Number of points used to draw each edge function.

        figsize : tuple, optional
            Matplotlib figure size.
        """
        if params is None:
            params = self.params

        x = jnp.asarray(x)

        if x.ndim == 1:
            x = x[:, None]

        if x.shape[-1] != self.width[0]:
            raise ValueError(
                f"Expected input width {self.width[0]}, "
                f"received {x.shape[-1]}"
            )

        # Store the input values seen by every layer.
        layer_inputs = []
        activations = x

        for layer_params in params:
            layer_inputs.append(activations)
            activations = self.moekan_layer(
                layer_params,
                activations,
            )

        n_layers = len(self.width) - 1
        maximum_width = max(self.width)

        if figsize is None:
            figsize = (
                max(14.0, 5.0 * n_layers),
                max(8.0, 1.8 * maximum_width),
            )

        figure = plt.figure(figsize=figsize)
        network_axis = figure.add_axes(
            [0.02, 0.02, 0.96, 0.96]
        )

        network_axis.set_xlim(-0.05, n_layers + 0.05)
        network_axis.set_ylim(-0.05, 1.05)
        network_axis.axis("off")

        def node_positions(layer_width):
            if layer_width == 1:
                return np.array([0.5])

            return np.linspace(
                0.90,
                0.10,
                layer_width,
            )

        layer_y_positions = [
            node_positions(layer_width)
            for layer_width in self.width
        ]

        layer_x_positions = np.linspace(
            0.0,
            float(n_layers),
            n_layers + 1,
        )

        # Draw the neurons.
        for layer_index, layer_width in enumerate(self.width):
            x_position = layer_x_positions[layer_index]
            y_positions = layer_y_positions[layer_index]

            for neuron_index, y_position in enumerate(y_positions):
                network_axis.scatter(
                    x_position,
                    y_position,
                    s=180,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=10,
                )

                if layer_index == 0:
                    label = rf"$x_{neuron_index + 1}$"
                elif layer_index == n_layers:
                    label = rf"$y_{neuron_index + 1}$"
                else:
                    label = rf"$h^{{({layer_index})}}_{neuron_index + 1}$"

                network_axis.text(
                    x_position,
                    y_position + 0.055,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Draw each edge and place one inset plot on it.
        for layer_index, layer_parameters in enumerate(params):
            input_width = self.width[layer_index]
            output_width = self.width[layer_index + 1]

            input_x = layer_x_positions[layer_index]
            output_x = layer_x_positions[layer_index + 1]

            input_y_positions = layer_y_positions[layer_index]
            output_y_positions = layer_y_positions[layer_index + 1]

            layer_data = np.asarray(
                layer_inputs[layer_index]
            )

            for output_index in range(output_width):
                for input_index in range(input_width):
                    source_y = input_y_positions[input_index]
                    target_y = output_y_positions[output_index]

                    # Draw the connection line.
                    network_axis.plot(
                        [input_x, output_x],
                        [source_y, target_y],
                        color="0.70",
                        linewidth=0.7,
                        zorder=1,
                    )

                    edge_parameters = {
                        name: value[output_index, input_index]
                        for name, value in layer_parameters.items()
                    }

                    edge_input_values = layer_data[:, input_index]
                    edge_input_values = np.asarray(
                        edge_input_values
                    )

                    finite_values = edge_input_values[
                        np.isfinite(edge_input_values)
                    ]

                    if finite_values.size == 0:
                        continue

                    lower_bound = float(np.min(finite_values))
                    upper_bound = float(np.max(finite_values))

                    if np.isclose(lower_bound, upper_bound):
                        padding = max(
                            1.0,
                            0.1 * abs(lower_bound),
                        )
                        lower_bound -= padding
                        upper_bound += padding

                    edge_x = jnp.linspace(
                        lower_bound,
                        upper_bound,
                        samples,
                    )

                    edge_y = self.moekan_edge(
                        edge_x,
                        edge_parameters,
                    )

                    edge_x = np.asarray(edge_x)
                    edge_y = np.asarray(edge_y)

                    # Place a small plot near the middle of the edge.
                    middle_x = (
                        input_x + output_x
                    ) / 2.0

                    middle_y = (
                        source_y + target_y
                    ) / 2.0

                    # Offset parallel edges so that their boxes do not
                    # completely overlap.
                    edge_number = (
                        output_index * input_width
                        + input_index
                    )

                    total_edges = input_width * output_width

                    if total_edges > 1:
                        offset = (
                            edge_number
                            - 0.5 * (total_edges - 1)
                        ) * 0.018
                    else:
                        offset = 0.0

                    box_width = min(
                        0.18,
                        0.75 / max(1, total_edges),
                    )

                    box_height = min(
                        0.12,
                        0.60 / max(1, total_edges),
                    )

                    box_left = (
                        middle_x
                        - 0.5 * box_width
                    )

                    box_bottom = (
                        middle_y
                        + offset
                        - 0.5 * box_height
                    )

                    edge_axis = figure.add_axes(
                        [
                            0.02 + 0.96 * box_left,
                            0.02 + 0.96 * box_bottom,
                            0.96 * box_width,
                            0.96 * box_height,
                        ]
                    )

                    edge_axis.plot(
                        edge_x,
                        edge_y,
                        color="tab:blue",
                        linewidth=0.8,
                    )

                    edge_axis.set_title(
                        rf"$\phi^{{({layer_index + 1})}}_"
                        rf"{{{output_index + 1},{input_index + 1}}}$",
                        fontsize=6,
                        pad=1,
                    )

                    edge_axis.tick_params(
                        labelsize=5,
                        length=1.5,
                        pad=1,
                    )

                    edge_axis.grid(
                        True,
                        linewidth=0.25,
                        alpha=0.5,
                    )

                    edge_axis.set_xlim(
                        lower_bound,
                        upper_bound,
                    )

        for layer_index, x_position in enumerate(
            layer_x_positions
        ):
            network_axis.text(
                x_position,
                -0.025,
                f"Layer {layer_index}",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

        network_axis.text(
            0.5 * n_layers,
            1.02,
            "MOEKAN edge-function tree",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

        figure.savefig(
            filename,
            format="pdf",
            bbox_inches="tight",
        )

        plt.close(figure)

        print(
            f"Saved MOEKAN edge tree to {filename}"
        )

    def plot_edge_functions(
        self,
        x,
        params=None,
        filename="moekan_edge_functions.pdf",
        samples=300,
    ):
        """
        Plot all MOEKAN edge functions in a grid.

        Columns correspond to network layers.
        Rows correspond to input-output edge relations.
        """
        if params is None:
            params = self.params

        x = jnp.asarray(x)

        if x.ndim == 1:
            x = x[:, None]

        if x.shape[-1] != self.width[0]:
            raise ValueError(
                f"Expected input width {self.width[0]}, "
                f"received {x.shape[-1]}"
            )

        # Store the inputs entering each layer.
        layer_inputs = []
        activations = x

        for layer_params in params:
            layer_inputs.append(activations)
            activations = self.moekan_layer(
                layer_params,
                activations,
            )

        n_layers = len(params)
        edges_per_layer = [
            self.width[layer_index]
            * self.width[layer_index + 1]
            for layer_index in range(n_layers)
        ]

        n_rows = max(edges_per_layer)
        n_cols = n_layers

        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            squeeze=False,
            figsize=(
                4.0 * n_cols,
                2.5 * n_rows,
            ),
        )

        for layer_index, layer_params in enumerate(params):
            input_width = self.width[layer_index]
            output_width = self.width[layer_index + 1]
            layer_data = np.asarray(
                layer_inputs[layer_index]
            )

            edge_number = 0

            for output_index in range(output_width):
                for input_index in range(input_width):
                    axis = axes[edge_number, layer_index]

                    edge_values = np.asarray(
                        layer_data[:, input_index]
                    )

                    finite_values = edge_values[
                        np.isfinite(edge_values)
                    ]

                    if finite_values.size == 0:
                        axis.set_visible(False)
                        edge_number += 1
                        continue

                    lower_bound = float(
                        np.min(finite_values)
                    )
                    upper_bound = float(
                        np.max(finite_values)
                    )

                    if np.isclose(
                        lower_bound,
                        upper_bound,
                    ):
                        padding = max(
                            1.0,
                            0.1 * abs(lower_bound),
                        )
                        lower_bound -= padding
                        upper_bound += padding

                    edge_x = jnp.linspace(
                        lower_bound,
                        upper_bound,
                        samples,
                    )

                    edge_parameters = {
                        name: value[
                            output_index,
                            input_index,
                        ]
                        for name, value in layer_params.items()
                    }

                    edge_y = self.moekan_edge(
                        edge_x,
                        edge_parameters,
                    )

                    edge_x = np.asarray(edge_x)
                    edge_y = np.asarray(edge_y)

                    axis.plot(
                        edge_x,
                        edge_y,
                        color="tab:blue",
                        linewidth=1.2,
                    )

                    axis.axhline(
                        0.0,
                        color="black",
                        linewidth=0.4,
                        alpha=0.5,
                    )

                    axis.grid(
                        True,
                        linewidth=0.4,
                        alpha=0.4,
                    )

                    axis.set_title(
                        rf"$\phi^{{({layer_index + 1})}}_"
                        rf"{{{output_index + 1},{input_index + 1}}}$",
                        fontsize=10,
                    )

                    axis.tick_params(
                        labelsize=7,
                    )

                    edge_number += 1

        # Hide unused rows in layers with fewer edges.
        for row_index in range(
            edge_number,
            n_rows,
        ):
            axes[row_index, layer_index].axis("off")

        axes[0, layer_index].set_title(
            f"Layer {layer_index + 1}\n"
            rf"$n_{{in}}={input_width},\ "
            rf"n_{{out}}={output_width}$",
            fontsize=12,
            fontweight="bold",
        )

        for layer_index in range(n_layers):
            axes[-1, layer_index].set_xlabel(
                "edge input",
                fontsize=9,
            )

            for row_index in range(n_rows):
                if axes[row_index, 0].get_visible():
                    axes[row_index, 0].set_ylabel(
                        "edge output",
                        fontsize=9,
                    )

            figure.suptitle(
                "MOEKAN analytical edge functions",
                fontsize=16,
                fontweight="bold",
            )

            figure.tight_layout(
                rect=[0.0, 0.0, 1.0, 0.96]
            )

            figure.savefig(
                filename,
                format="pdf",
                bbox_inches="tight",
            )

            plt.close(figure)

            print(
                f"Saved MOEKAN edge functions to {filename}"
            )