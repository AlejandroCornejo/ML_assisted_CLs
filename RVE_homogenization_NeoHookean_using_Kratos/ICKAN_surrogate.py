
import torch as torch
import torch.nn as nn
import sys

sys.path.insert(0, r'C:\ICKANs')
import ickan as KAN

class ICKAN_W_Surrogate(nn.Module):

    def __init__(self, order_stretches, grid_size, k, W_width):
        super(ICKAN_W_Surrogate, self).__init__()

        self.order_stretches = order_stretches
        self.input_size = 2 * self.order_stretches + 1  # Total inputs: 2 * reg_eigenvalues for each order + 1 * log(J)
        self.grid_size = grid_size
        self.k = k

        # Define the spline grid range for all inputs
        # grid_range = [-1.0, 1.0]

        # KAN definition for the energy density potential W
        self.KAN_W = KAN.MultKAN(
            base_fun = "zero",
            grid_eps = 0.1, # 1 grid is uniformly spaced in the range [grid_range[0], grid_range[1]]
            width=W_width,  # output of size 1: W
            grid=self.grid_size,
            k=self.k,

            # grid_range=grid_range,
            # grid_range_0=grid_range,

            # grid_range=[grid_range,grid_range,grid_range, grid_range, grid_range],
            # grid_range_0=[grid_range,grid_range,grid_range, grid_range, grid_range],

            sp_trainable = False,
            sb_trainable = False,
        )

        # self.KAN_W.speed()

        # Initialize some extra parameters
        self.ki = nn.ParameterList([
            p + 1 for p in range(self.order_stretches + 1)
            # nn.Parameter(torch.tensor(p + 1.0)) for p in range(self.order_stretches + 1)
        ])

        # The parameter multiplying the log(J) is initially set to 1.0
        self.ki[-1] = 1.0
        # self.ki[-1] = nn.Parameter(torch.tensor(1.0))
        
        # for k in self.ki:
        #     print(f"Initial ki: {k.item()}")

    # ==========================================================================================

    def print_kan_edge_grid_ranges(self):
        """
        Print the grid input ranges for each KAN layer edge.
        """
        print("KAN grid edge ranges:")
        for layer_idx, kanlayer in enumerate(self.KAN_W.act_fun):
            grid = kanlayer.grid.detach().cpu()
            if grid.ndim == 1:
                x_min, x_max = float(grid[0]), float(grid[-1])
                print(f"  layer {layer_idx}: [{x_min:.6g}, {x_max:.6g}]")
            else:
                print(f"  layer {layer_idx} (in_dim={grid.shape[0]}, out_dim={kanlayer.out_dim}):")
                for inp in range(grid.shape[0]):
                    x_min = float(grid[inp, 0])
                    x_max = float(grid[inp, -1])
                    print(f"    edge input {inp}: [{x_min:.6g}, {x_max:.6g}]")

    # ==========================================================================================

    def UpdateGridFromSamples(self, strain_database):
        kan_input = self._compute_kan_input_for_strain(strain_database)  # Shape: (batches*steps, input_size) --> lambda_1, lambda_2, log_J
        max_lambda_1 = torch.max(kan_input[:, 0]).item()
        min_lambda_1 = torch.min(kan_input[:, 0]).item()
        max_lambda_2 = torch.max(kan_input[:, 1]).item()
        min_lambda_2 = torch.min(kan_input[:, 1]).item()
        max_log_J = torch.max(kan_input[:, -1]).item()
        min_log_J = torch.min(kan_input[:, -1]).item()
        # print(f"KAN input ranges from samples:")
        # print(f"  lambda_1: [{min_lambda_1:.6g}, {max_lambda_1:.6g}]")
        # print(f"  lambda_2: [{min_lambda_2:.6g}, {max_lambda_2:.6g}]")
        # print(f"  log_J: [{min_log_J:.6g}, {max_log_J:.6g}]")

        steps = 200
        input_1 = torch.linspace(0.5*min_lambda_1, 1.5*max_lambda_1, steps)
        input_2 = torch.linspace(0.5*min_lambda_2, 1.5*max_lambda_2, steps)
        input_3 = torch.linspace(0.5*min_log_J   , 1.5*max_log_J, steps)
        kan_input_for_grid = torch.stack((input_1, input_2, input_3), dim=1)  # Shape: (steps, 3), stacked along columns
        
        # kan_input_for_grid = torch.tensor([
        #     [min_lambda_1, min_lambda_2, min_log_J],
        #     [max_lambda_1, max_lambda_2, max_log_J]
        # ])

        self.KAN_W.update_grid(kan_input_for_grid)


    # ==========================================================================================

    def _compute_kan_input_for_strain(self, strain):
        """
        Compute KAN input for a given strain tensor
        strain: Tensor of shape (batches, 3) with components [E_xx, E_yy, E_xy]
        Returns: Tensor of shape (batches, input_size) with KAN inputs
        
        This method must be called just ONCE when loading the strain database, to compute the KAN inputs for all samples.
        """
        batches = strain.shape[0]

        E = torch.zeros((batches, 2, 2))
        E[:, 0, 0] = strain[:, 0]
        E[:, 1, 1] = strain[:, 1]
        E[:, 0, 1] = 0.5 * strain[:, 2]
        E[:, 1, 0] = 0.5 * strain[:, 2]

        C = 2.0 * E + torch.eye(2)
        J = torch.linalg.det(C) ** 0.5
        # log_J = torch.log(J + 1.0e-12)
        log_J = (J - 0.9995)**2

        square_eigenvalues = torch.linalg.eigvalsh(C)
        eigenvalues = torch.sqrt(square_eigenvalues)

        reg_eigenvalues = torch.zeros_like(eigenvalues)
        aux = J ** (-1 / 3)
        reg_eigenvalues[:, 0] = eigenvalues[:, 0] * aux
        reg_eigenvalues[:, 1] = eigenvalues[:, 1] * aux

        kan_inputs = []
        for index in range(self.order_stretches):
            reg_eigenvalues_order = reg_eigenvalues ** self.ki[index]
            kan_inputs.append(reg_eigenvalues_order)

        log_J_scaled = log_J * self.ki[-1]
        log_J_expanded = log_J_scaled.unsqueeze(-1)
        kan_inputs.append(log_J_expanded)

        KAN_input = torch.cat(kan_inputs, dim=-1)
        
        viewed_KAN_input = KAN_input.view(-1, self.input_size)

        return viewed_KAN_input # Reshape to (batches*steps, input_size)


    # ==========================================================================================
    def CalculateW(self, strain_database):
        """
        Computes W for the given strain with normalization.
        """
        kan_input = self._compute_kan_input_for_strain(strain_database)  # Shape: (batches*steps, input_size)

        null_kan_input = self._compute_kan_input_for_strain(torch.zeros_like(strain_database))
        W0 = self.KAN_W.forward(null_kan_input)

        W_raw = self.KAN_W.forward(kan_input)  # Shape: (batch x steps, 1)

        return W_raw - W0
    # ==========================================================================================

    def forward(self, strain_database):
        return self.CalculateW(strain_database)

    # ==========================================================================================
    def CalculateNormalizedStress(self, strain_database):
        """
        Computes the normalized stress (derivative of W with respect to strain) for the given strain.
        """

        # Ensure strain_database requires gradient for autograd.grad computation
        strain_database = strain_database.requires_grad_(True)

        null_strain  = torch.zeros(1, 3).requires_grad_(True)
        W0 = self.CalculateW(null_strain)

        W = self.CalculateW(strain_database)  # Shape: (batches*steps, 1)

        stress_0 = torch.autograd.grad(
                    outputs=W0,
                    inputs=null_strain,
                    grad_outputs=torch.ones_like(W0),
                    create_graph=True
                    )[0]

        predicted_stress = torch.autograd.grad(
                    outputs=W,
                    inputs=strain_database,
                    grad_outputs=torch.ones_like(W),
                    create_graph=True
                    )[0]

        return predicted_stress - stress_0