
import torch as torch
import torch.nn as nn
import sys
import os
import inspect

ICKANS_REPO = "/home/kratos/ICKANs"
PYKAN_REPO = "/home/kratos/pykan"

if os.path.isdir(ICKANS_REPO) and ICKANS_REPO not in sys.path:
    sys.path.insert(0, ICKANS_REPO)

try:
    import ickan as KAN
    KAN_BACKEND = "ickan"
except ImportError:
    # Local Sebastian tests can use the baseline pykan implementation already
    # available at /home/kratos/pykan. This is not the same as Alejandro's
    # ICKAN fork; it is a fallback so we can run baseline KAN tests locally.
    if PYKAN_REPO not in sys.path:
        sys.path.insert(0, PYKAN_REPO)
    import kan as KAN
    KAN_BACKEND = "kan"

class ICKAN_W_Surrogate(nn.Module):

    def __init__(
        self,
        order_stretches,
        grid_size,
        k,
        W_width,
        input_mode="principal",
        base_fun="silu",
        noise_scale=0.0,
        grid_eps=0.01,
    ):
        super(ICKAN_W_Surrogate, self).__init__()

        if input_mode not in (
            "principal",
            "direct_strain",
            "hybrid",
            "orthotropic_invariants",
            "orthotropic_invariants_signed",
        ):
            raise ValueError(
                "input_mode must be 'principal', 'direct_strain', 'hybrid', "
                "'orthotropic_invariants', or 'orthotropic_invariants_signed'. "
                f"Got {input_mode}."
            )
        self.input_mode = input_mode
        self.base_fun = base_fun
        self.noise_scale = noise_scale
        self.grid_eps = grid_eps
        self.order_stretches = order_stretches
        self.principal_input_size = 2 * self.order_stretches + 1
        if self.input_mode == "direct_strain":
            self.input_size = 3
        elif self.input_mode == "orthotropic_invariants":
            self.input_size = 4
        elif self.input_mode == "orthotropic_invariants_signed":
            self.input_size = 5
        elif self.input_mode == "hybrid":
            self.input_size = self.principal_input_size + 3
        else:
            self.input_size = self.principal_input_size  # Total inputs: 2 * reg_eigenvalues for each order + 1 * log(J)
        self.grid_size = grid_size
        self.k = k

        first_width = W_width[0][0] if isinstance(W_width[0], (list, tuple)) else W_width[0]
        if int(first_width) != int(self.input_size):
            raise ValueError(
                f"W_width[0] must match input_size={self.input_size} for input_mode={self.input_mode}. "
                f"Got W_width={W_width}."
            )

        # Define the spline grid range for all inputs
        # grid_range = [-1.0, 1.0]

        # KAN definition for the energy density potential W
        kan_kwargs = dict(
            base_fun=base_fun,
            grid_eps=grid_eps, # 1 grid is uniform, 0 is sample-adaptive.
            width=W_width,  # output of size 1: W
            grid=self.grid_size,
            k=self.k,
            
            # affine_trainable = True,
            # sparse_init = True,

            # grid_range=grid_range,
            # grid_range_0=grid_range,

            # grid_range=[grid_range,grid_range,grid_range, grid_range, grid_range],
            # grid_range_0=[grid_range,grid_range,grid_range, grid_range, grid_range],

            # sp_trainable = False,
            # sb_trainable = False,
            
            noise_scale=noise_scale
        )
        try:
            multkan_signature = inspect.signature(KAN.MultKAN)
            if "auto_save" in multkan_signature.parameters:
                kan_kwargs["auto_save"] = False
            if "ckpt_path" in multkan_signature.parameters:
                kan_kwargs["ckpt_path"] = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "KAN_model_cache",
                )
        except (TypeError, ValueError):
            pass

        self.KAN_W = KAN.MultKAN(**kan_kwargs)

        if hasattr(self.KAN_W, "speed"):
            self.KAN_W.speed()

        # Initialize some extra parameters
        self.ki = nn.ParameterList([
            # p + 1 for p in range(self.order_stretches + 1)
            nn.Parameter(torch.tensor(p + 1.0)) for p in range(self.order_stretches + 1)
        ])

        # The parameter multiplying the log(J) is initially set to 1.0
        # self.ki[-1] = 1.0
        self.ki[-1] = nn.Parameter(torch.tensor(1.0))
        
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
        self.KAN_W.update_grid_from_samples(kan_input.detach())


    # ==========================================================================================

    def _compute_principal_kan_input_for_strain(self, strain):
        """
        Compute principal-stretch/logJ inputs for a given strain tensor.
        strain: Tensor of shape (batches, 3) with components [E_xx, E_yy, E_xy]
        Returns: Tensor of shape (batches, principal_input_size) with KAN inputs
        """
        batches = strain.shape[0]

        E = torch.zeros((batches, 2, 2), device=strain.device, dtype=strain.dtype)
        E[:, 0, 0] = strain[:, 0]
        E[:, 1, 1] = strain[:, 1]
        E[:, 0, 1] = 0.5 * strain[:, 2]
        E[:, 1, 0] = 0.5 * strain[:, 2]

        C = 2.0 * E + torch.eye(2, device=strain.device, dtype=strain.dtype)
        det_C = torch.clamp(torch.linalg.det(C), min=1.0e-12)
        J = torch.sqrt(det_C)
        log_J = torch.log(J + 1.0e-8)
        # log_J = (J - 1.0)**2

        square_eigenvalues = torch.linalg.eigvalsh(C)
        eigenvalues = torch.sqrt(torch.clamp(square_eigenvalues, min=1.0e-12))

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

        viewed_KAN_input = KAN_input.view(-1, self.principal_input_size)

        return viewed_KAN_input # Reshape to (batches*steps, input_size)

    # ==========================================================================================

    def _compute_orthotropic_invariant_input_for_strain(self, strain, include_signed_shear=False):
        """
        Compute RVE-axis-aware invariant-like inputs.

        The strain vector uses [E_xx, E_yy, gamma_xy], where gamma_xy is the
        engineering shear component stored by the FOM data. With C = I + 2E,
        this gives C_xy = gamma_xy. The 4-feature mode uses C_xy^2, which keeps
        the energy even in shear. The signed 5-feature mode also includes C_xy
        so the model can learn non-even shear effects or optimize signed shear
        sensitivities more easily.
        """
        flat_strain = strain.view(-1, 3)

        c_xx_minus_one = 2.0 * flat_strain[:, 0:1]
        c_yy_minus_one = 2.0 * flat_strain[:, 1:2]
        c_xy = flat_strain[:, 2:3]
        c_xy_squared = c_xy**2

        c_xx = 1.0 + c_xx_minus_one
        c_yy = 1.0 + c_yy_minus_one
        det_c = torch.clamp(c_xx * c_yy - c_xy_squared, min=1.0e-12)
        log_j = 0.5 * torch.log(det_c + 1.0e-8)

        if include_signed_shear:
            return torch.cat(
                [c_xx_minus_one, c_yy_minus_one, c_xy, c_xy_squared, log_j],
                dim=-1,
            )

        return torch.cat([c_xx_minus_one, c_yy_minus_one, c_xy_squared, log_j], dim=-1)

    # ==========================================================================================

    def _compute_kan_input_for_strain(self, strain):
        """
        Compute KAN input for a given strain tensor.
        strain: Tensor of shape (batches, 3) with components [E_xx, E_yy, E_xy]
        Returns: Tensor of shape (batches, input_size) with KAN inputs
        """
        flat_strain = strain.view(-1, 3)
        if self.input_mode == "direct_strain":
            return flat_strain
        if self.input_mode == "orthotropic_invariants":
            return self._compute_orthotropic_invariant_input_for_strain(flat_strain)
        if self.input_mode == "orthotropic_invariants_signed":
            return self._compute_orthotropic_invariant_input_for_strain(
                flat_strain,
                include_signed_shear=True,
            )

        principal_input = self._compute_principal_kan_input_for_strain(flat_strain)
        if self.input_mode == "principal":
            return principal_input

        return torch.cat([principal_input, flat_strain], dim=-1)


    # ==========================================================================================
    def CalculateW(self, strain_database):
        """
        Computes the raw shifted energy W(E)-W(0) for the given strain.

        This is not yet the stress-free reference energy if the raw network has
        a nonzero gradient at the reference state. For plotting/training energy
        values together with CalculateNormalizedStress, use CalculateCorrectedW.
        """
        kan_input = self._compute_kan_input_for_strain(strain_database)  # Shape: (batches*steps, input_size)

        null_strain = torch.zeros(
            1,
            3,
            device=strain_database.device,
            dtype=strain_database.dtype,
        )
        null_kan_input = self._compute_kan_input_for_strain(null_strain)
        W0 = self.KAN_W.forward(null_kan_input)

        W_raw = self.KAN_W.forward(kan_input)  # Shape: (batch x steps, 1)

        return W_raw - W0
    # ==========================================================================================

    def CalculateCorrectedW(self, strain_database):
        """
        Computes the energy consistent with the corrected stress.

        CalculateNormalizedStress returns dW/dE - dW/dE|_0 so that the reference
        state is stress-free. The matching potential is therefore

            W_c(E) = W(E) - W(0) - [dW/dE|_0] : E.
        """
        flat_strain = strain_database.view(-1, 3)
        raw_w = self.CalculateW(flat_strain)

        null_strain = torch.zeros(
            1,
            3,
            device=flat_strain.device,
            dtype=flat_strain.dtype,
        ).requires_grad_(True)
        raw_w0 = self.CalculateW(null_strain)
        stress_0 = torch.autograd.grad(
            outputs=raw_w0,
            inputs=null_strain,
            grad_outputs=torch.ones_like(raw_w0),
            create_graph=True,
        )[0]

        linear_reference_work = torch.sum(stress_0 * flat_strain, dim=1, keepdim=True)
        return raw_w - linear_reference_work

    # ==========================================================================================

    def forward(self, strain_database):
        return self.CalculateCorrectedW(strain_database)

    # ==========================================================================================
    def CalculateNormalizedStress(self, strain_database):
        """
        Computes the normalized stress (derivative of W with respect to strain) for the given strain.
        """

        # Ensure strain_database requires gradient for autograd.grad computation
        strain_database = strain_database.requires_grad_(True)

        null_strain = torch.zeros(
            1,
            3,
            device=strain_database.device,
            dtype=strain_database.dtype,
        ).requires_grad_(True)
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
