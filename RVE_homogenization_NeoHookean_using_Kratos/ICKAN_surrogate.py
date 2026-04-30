# std libs imports
import numpy as np
import scipy as sp
import torch as torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os
import sys
import random

# import pykan.kan as KAN # now the repo is local, not pip

# NOW we use input convex KANs, so we import the KAN class from the ICKANs module instead of pykan
sys.path.insert(0, r'C:\ICKANs')

import ickan as KAN
# import kan as KAN


# Define the GetColor method
def GetColor(component):
    if component == 0:
        return "r"  # Red for component 0
    elif component == 1:
        return "b"  # Blue for component 1
    elif component == 2:
        return "g"  # Green for component 2
    else:
        return "k"  # Black for any other component



#=============================================================================================================
"""
INPUT DATASET:
Load strain and stress from FOM trajectories (10 trajectories from stage_1_training_set_fom folder).
Data is loaded as [history, step, component] with shape [10, steps, 3].
"""
n_epochs = 1500
learning_rate = 1.0

# Path to FOM trajectories folder (relative to script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
fom_trajectories_dir = os.path.join(script_dir, 'stage_1_training_set_fom')

# Load all 10 trajectories
strain_trajectories = []
stress_trajectories = []

for i in range(1, 11):  # trajectory_1 to trajectory_10
    strain_file = os.path.join(fom_trajectories_dir, f'trajectory_{i}', f'trajectory_{i}_strain.npy')
    stress_file = os.path.join(fom_trajectories_dir, f'trajectory_{i}', f'trajectory_{i}_stress.npy')
    
    strain_data = np.load(strain_file)  # shape: (steps, 3)
    stress_data = np.load(stress_file)  # shape: (steps, 3)
    
    strain_trajectories.append(strain_data)
    stress_trajectories.append(stress_data)
    print(f"Loaded trajectory_{i}: strain={strain_data.shape}, stress={stress_data.shape}")

# Find minimum number of steps across all trajectories
min_steps = min(t.shape[0] for t in strain_trajectories)
print(f"\nMinimum number of steps across all trajectories: {min_steps}")

# Sample each trajectory at equally spaced intervals to cover the full path
# For trajectories longer than min_steps, select min_steps points evenly spaced
def sample_equally_spaced(data, n_samples):
    """Sample data at equally spaced intervals."""
    original_n = data.shape[0]
    if original_n == n_samples:
        return data  # No sampling needed
    # Create indices that are equally spaced
    indices = np.linspace(0, original_n - 1, n_samples, dtype=int)
    return data[indices]

strain_trajectories_truncated = [sample_equally_spaced(t, min_steps) for t in strain_trajectories]
stress_trajectories_truncated = [sample_equally_spaced(t, min_steps) for t in stress_trajectories]

print(f"Sampled trajectories to {min_steps} equally spaced points from original lengths: {[t.shape[0] for t in strain_trajectories]}")

# Stack trajectories into [history, step, component] format
ref_strain_database = torch.tensor(np.stack(strain_trajectories_truncated), dtype=torch.float32)  # [10, steps, 3]
ref_stress_database = torch.tensor(np.stack(stress_trajectories_truncated), dtype=torch.float32)  # [10, steps, 3]

# Convert stress to be base 1
ref_stress_database /= 1.0e9
ref_strain_database /= 2.0

# Use all data for training (no train/test split)
train_strain_database = ref_strain_database
train_stress_database = ref_stress_database

print("\nLaunching the training of a KAN...")
print(f"Number of training trajectories: {train_strain_database.shape[0]}")
print(f"Number of total trajectories: {ref_strain_database.shape[0]}")
print(f"Number of steps: {train_strain_database.shape[1]}")
print(f"Strain size: {train_strain_database.shape[2]}")

# ==========================================================================================

class KANStressPredictor(nn.Module):
    """
    KAN-based Stress Predictor with support for multiple orders of stretches.
    """

    def __init__(self):
        super(KANStressPredictor, self).__init__()

        # EDIT:
        self.order_stretches = 1  # Number of orders (can be set to any value)
        self.k = 2  # Degree of splines
        self.grid = 3  # Number of knots
        # -------------------------------------

        self.input_size = 2 * self.order_stretches + 1  # Total inputs: 2 * reg_eigenvalues for each order + 1 * log(J)

        # KAN definition
        self.KAN_W = KAN.MultKAN(
            width=[self.input_size, 5, 3, 1], # output of size 1: W
            grid=self.grid,
            k=self.k,
            # grid_range=[-100, 100.0]
        )

        # Initialize some extra parameters
        self.ki = nn.ParameterList([
            # nn.Parameter(torch.tensor((float(p + 1)))) for p in range(self.order_stretches + 1)
            nn.Parameter(torch.tensor(1.0)) for p in range(self.order_stretches + 1)
        ])

        # The parameter multiplying the log(J) is initially set to 1.0
        self.ki[-1] = nn.Parameter(torch.tensor(1.0))
        # self.ki[-1] = 1.0

        for i, ki in enumerate(self.ki):
            print("self.ki[i]: ", ki.data)

    def ResetParameters(self):
        # Initialize some extra parameters
        self.ki = nn.ParameterList([
            nn.Parameter(torch.tensor(float(p + 1) + random.random())) for p in range(self.order_stretches + 1)
            # nn.Parameter(torch.tensor(float(p + 1))) for p in range(self.order_stretches + 1)
        ])

        # The parameter multiplying the log(J) is initially set to 1.0
        self.ki[-1] = nn.Parameter(torch.tensor(1.0))

    # ==========================================================================================
    def CalculateWWithoutNormalization(self, strain):
        """
        Computes W for the given strain without normalization.
        """
        batches = strain.shape[0]
        steps = strain.shape[1]

        # Compute strain tensor components
        E = torch.zeros((batches, steps, 2, 2))
        E[:, :, 0, 0] = strain[:, :, 0]  # Exx
        E[:, :, 1, 1] = strain[:, :, 1]  # Eyy
        E[:, :, 0, 1] = 0.5 * strain[:, :, 2]  # Exy
        E[:, :, 1, 0] = 0.5 * strain[:, :, 2]  # Eyx

        # Left Cauchy strain tensor
        C = torch.zeros_like(E)
        C = 2.0 * E + torch.eye(2)

        J = torch.linalg.det(C) ** 0.5  # Determinant of C (Jacobian)
        log_J = torch.log(J)  # Logarithm of J

        square_eigenvalues = torch.linalg.eigvalsh(C)  # Eigenvalues: batch x steps x 2
        eigenvalues = torch.sqrt(square_eigenvalues)

        reg_eigenvalues = torch.zeros_like(eigenvalues)  # Regularized eigenvalues
        aux = J ** (-1 / 3)
        reg_eigenvalues[:, :, 0] = eigenvalues[:, :, 0] * aux
        reg_eigenvalues[:, :, 1] = eigenvalues[:, :, 1] * aux

        # Prepare inputs for KAN
        kan_inputs = []  # List to store inputs for each order

        for index in range(self.order_stretches):
            # Compute reg_eigenvalues**order
            reg_eigenvalues_order = reg_eigenvalues ** self.ki[index]
            # reg_eigenvalues_order = reg_eigenvalues

            # Append the pair of stretches for this order
            kan_inputs.append(reg_eigenvalues_order)

        # Append log(J) multiplied by the last ki factor
        log_J_scaled = log_J * self.ki[-1]  # Multiply log(J) by the last ki factor
        log_J_expanded = log_J_scaled.unsqueeze(-1)  # Add an extra dimension for concatenation
        kan_inputs.append(log_J_expanded)

        # Concatenate all inputs along the last dimension
        KAN_input = torch.cat(kan_inputs, dim=-1)  # Shape: (batches, steps, 2 * self.order_stretches + 1)

        # Flatten the input for KAN (KAN cannot read 3D tensors)
        flat_KAN_input = KAN_input.view(-1, self.input_size)  # Shape: (batch x steps, input_size)

        # Pass the input through the KAN layer
        W_flat = self.KAN_W.forward(flat_KAN_input)  # Shape: (batch x steps, 1)

        # Reshape the output back to the original shape
        W = W_flat.view(batches, steps, -1)  # Shape: (batches, steps, 1)

        return W

    # ==========================================================================================
    def CalculateW(self, strain):
        """
        Computes W for the given strain and subtracts W evaluated at strain = torch.zeros.
        """
        # Compute W at strain = torch.zeros
        zeros = torch.zeros_like(strain)
        W0 = self.CalculateWWithoutNormalization(zeros)

        # Compute W without normalization
        W = self.CalculateWWithoutNormalization(strain)

        # Subtract W0 from W
        return W - W0

    # ==========================================================================================
    def forward(self, strain):
        """
        Forward pass to compute the gradient of W with respect to strain.
        """
        strain = strain.detach().requires_grad_(True)
        zeros = torch.zeros_like(strain).requires_grad_(True)

        W0 = self.CalculateW(zeros)
        grad_at_zero = torch.autograd.grad(
            outputs=W0,
            inputs=zeros,
            grad_outputs=torch.ones_like(W0),
            create_graph=True
        )[0]

        W = self.CalculateW(strain)  # Shape: (batches, steps, 1)

        # Compute gradients
        grad = torch.autograd.grad(
            outputs=W,
            inputs=strain,
            grad_outputs=torch.ones_like(W),
            create_graph=True
        )[0]

        return grad - grad_at_zero


    # ==========================================================================================
    def ComputeHessian(self, strain):
        strain = strain.detach().requires_grad_(True)  # Ensure strain requires gradients

        # Compute the first gradient (Jacobian)
        W = self.CalculateW(strain)  # Shape: (batches, steps, 1)
        grad = torch.autograd.grad(
            outputs=W,
            inputs=strain,
            grad_outputs=torch.ones_like(W),
            create_graph=True
        )[0]  # Shape: (batches, steps, strain_size)

        # Compute the second gradient (Hessian)
        hessian = []
        for i in range(strain.shape[-1]):  # Loop over strain components
            grad_i = grad[..., i]  # Select the i-th component of the gradient
            hessian_row = torch.autograd.grad(
                outputs=grad_i,
                inputs=strain,
                grad_outputs=torch.ones_like(grad_i),
                create_graph=True
            )[0]  # Shape: (batches, steps, strain_size)
            hessian.append(hessian_row)

        # Stack the Hessian rows to form the full Hessian matrix
        hessian = torch.stack(hessian, dim=-1)  # Shape: (batches, steps, strain_size, strain_size)

        # Compute eigenvalues of the Hessian for each strain
        hessian_eigenvalues = torch.linalg.eigvalsh(hessian)  # Shape: (batches, steps, strain_size)

        # Check if all eigenvalues are positive
        is_positive_definite = (hessian_eigenvalues > 0).all(dim=-1)  # Shape: (batches, steps)
        if not is_positive_definite.all():
            print("**************************************************************")
            print("Warning: Hessian is not positive definite for some strains.")
            print("**************************************************************")
        return hessian


    # ==========================================================================================
    def plot_spline_edges(self, folder="./ICKAN_predictions", scale=0.5):
        """
        Plot the spline edges for each connection in the KAN layer.

        Args:
            folder (str): The folder to save the plots.
            scale (float): Scale factor for the plot size.
        """

        if not self.KAN_W.save_act:
            raise ValueError("Cannot plot since activations are not saved. Set `save_act=True` before the forward pass.")

        if self.KAN_W.acts is None:
            if self.KAN_W.cache_data is None:
                raise ValueError("No cached data available. Perform a forward pass with input data.")
            self.KAN_W.forward(self.KAN_W.cache_data)  # Populate activations

        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        depth = len(self.KAN_W.width) - 1  # Number of layers
        for l in range(depth):
            for i in range(self.KAN_W.width_in[l]):  # Loop over input nodes
                for j in range(self.KAN_W.width_out[l + 1]):  # Loop over output nodes
                    # Extract activations and spline outputs
                    x = self.KAN_W.acts[l][:, i].cpu().detach().numpy()  # Input activations
                    y = self.KAN_W.spline_postacts[l][:, j, i].cpu().detach().numpy()  # Spline outputs

                    # Sort the activations for a smooth plot
                    sorted_indices = x.argsort()
                    x_sorted = x[sorted_indices]
                    y_sorted = y[sorted_indices]

                    # Plot the spline edge
                    plt.figure(figsize=(6 * scale, 4 * scale))
                    plt.plot(x_sorted, y_sorted, label=f"Edge {i} -> {j}", color="blue", lw=2)
                    plt.xlabel("Activation (Input)")
                    plt.ylabel("Spline Output")
                    plt.title(f"Spline Edge: Layer {l}, Node {i} -> Node {j}")
                    plt.legend()
                    plt.grid(True)

                    # Save the plot
                    plot_path = os.path.join(folder, f"spline_edge_layer{l}_node{i}_to_node{j}.png")
                    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
                    plt.close()

        print(f"Spline edge plots saved in the folder: {folder}")
    
    def reset_splines(self):
        """
        Reset the spline parameters to their initial state.
        This is useful if you want to reinitialize the splines without creating a new model instance.
        """
        depth = len(self.KAN_W.width) - 1  # Number of layers
        for l in range(depth):
            for i in range(self.KAN_W.width_in[l]):  # Loop over input nodes
                for j in range(self.KAN_W.width_out[l + 1]):  # Loop over output nodes
                    # Extract activations and spline outputs
                    x = self.KAN_W.acts[l][:, i]  # Input activations
                    y = self.KAN_W.spline_postacts[l][:, j, i]  # Spline outputs
                    self.KAN_W.acts[l][:, j, i] = torch.zeros_like(x)  # Reset spline outputs to zero
                    self.KAN_W.spline_postacts[l][:, j, i] = torch.zeros_like(y)  # Reset spline outputs to zero
                    self.KAN_W.spline_preacts[l][:, j, i]  = torch.zeros_like(x)  # Reset spline pre-activations to zero
                    # print(x)
                    # print(y)
        print("Spline parameters have been reset to their initial state.")

# =========================================================================================
#==========================================================================================

def TRAIN_KAN(model, optimizer, train_strain_database, train_stress_database, n_epochs):
    """
    Train KAN model using mean stress difference loss.
    
    Args:
        model: KAN stress predictor model
        optimizer: Optimizer
        train_strain_database: Training strain data [batch, steps, components]
        train_stress_database: Training stress data [batch, steps, components] (in MPa)
        n_epochs: Number of training epochs
    """
    for epoch in range(n_epochs):
        def closure():
            optimizer.zero_grad()

            predicted_stress = model.forward(train_strain_database)
            
            # Use mean stress difference loss instead of work-based loss
            # Compute the mean difference between predicted and reference stress
            error = predicted_stress - train_stress_database
            loss = 0.5 * torch.mean(error ** 2)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

        if epoch == n_epochs - 1:
            print("\nTraining finished.\n")
            print("\nFinal loss: ", loss.item())


# Initialize model, optimizer, and loss function
model = KANStressPredictor()

print("\nNull strain KAN prediction initial CHECK: ", model.forward(torch.tensor([[[0.0, 0.0, 0.0]]]))) # for the order 1
print("\n")


# Initialize the optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    # max_iter=5,
    # history_size=7,
    # line_search_fn='strong_wolfe'
)


# Train the KAN model using mean stress difference loss on all data
TRAIN_KAN(
    model=model,
    optimizer=optimizer,
    train_strain_database=train_strain_database,
    train_stress_database=train_stress_database,
    n_epochs=n_epochs)

for i, ki in enumerate(model.ki):
    print("self.ki[i]: ", ki.data)

# Prune the KAN model (optional)
prune_KAN = False
if prune_KAN:
    print("Pruning the KAN...")
    model.CalculateW(train_strain_database)  # Forward pass to compute activations
    model.KAN_W = model.KAN_W.prune(node_th=0.1, edge_th=0.1)
    model.KAN_W.reset_model()  # new implementation
    model.ResetParameters()  # Reset the parameters after pruning
    model.CalculateW(train_strain_database)  # Forward pass to compute activations

    # Reinicializar completamente el modelo después del pruning
    # model = KANStressPredictor()
    print("All model parameters have been reset after pruning.")

    optimizer = optim.LBFGS(
                        model.parameters(),
                        lr=0.01,
                        max_iter=20,
                        history_size=30
                    )

    TRAIN_KAN(
        model=model,
        optimizer=optimizer,
        train_strain_database=train_strain_database,
        train_stress_database=train_stress_database,
        n_epochs=100 )

fix_symbolic = False
if fix_symbolic:
    # model.KAN_W.suggest_symbolic(0,0,0,weight_simple=0.0)
    # model.KAN_W.suggest_symbolic(0,1,0,weight_simple=0.)
    # model.KAN_W.suggest_symbolic(0,2,0,weight_simple=0.)

    model.KAN_W.fix_symbolic(0,0,0,'abs', random=False)
    model.KAN_W.fix_symbolic(0,1,0,'abs', random=False)
    # model.KAN_W.fix_symbolic(0,2,0,'x^2')

    TRAIN_KAN(
        model=model,
        optimizer=optimizer,
        train_strain_database=train_strain_database,
        train_stress_database=train_stress_database,
        n_epochs=200)

# model.ComputeHessian(ref_strain_database[:, :, :])  # Check the Hessian eigenvalues for the whole strain

print("\nNull strain KAN prediction POST CHECK: ", model.forward(torch.tensor([[[0.0, 0.0, 0.0]]]))) # for the order 1
print("\n")

# Create the folder to save the plots if it doesn't exist
output_folder = "ICKAN_predictions"
os.makedirs(output_folder, exist_ok=True)
torch.save(model.state_dict(), "ICKAN_predictions/KAN_model_weights.pth")

# Generate predictions for the full database (all data)
prediction_KAN = model.forward(ref_strain_database)

# Plot and save results for all trajectories (all data)
for elem in range(ref_strain_database.shape[0]):
    plt.figure(figsize=(8, 6))  # Create a new figure for each trajectory

    for compo in [0, 1, 2]:
        strain_for_print = ref_strain_database[elem, :, compo]
        predicted_stress_ANN = prediction_KAN[elem, :, compo].detach().numpy()

        plt.plot(
            strain_for_print,
            ref_stress_database[elem, :, compo],
            label=f"DATA_comp{compo}",
            # marker="o",
            color=GetColor(compo),
            # s=50,
            alpha=0.7,      # This should work...
            # edgecolors='none' # Try adding this to see if the borders are the issue
            linestyle="--"
        )
        plt.plot(
            strain_for_print,
            predicted_stress_ANN,
            label=f"ICKAN_comp{compo}",
            color=GetColor(compo),
            linestyle="-"
        )

    # Add plot details
    plt.title(f"Trajectory: {elem}" + f" KAN")
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [MPa]")
    plt.legend()
    plt.grid(True)

    # Save the plot in the output folder
    output_path = os.path.join(output_folder, f"batch_{elem}.png")
    plt.savefig(output_path, dpi=600)
    plt.close()  # Close the figure to free memory
print(f"Plots saved in the folder: {output_folder}")


model.CalculateW(ref_strain_database[:, :, :])  # Compute the spline activations
# print("check that W is null at zero strain: ", model.CalculateW(torch.zeros_like(ref_strain_database[:, :, :])))
model.plot_spline_edges()
model.KAN_W.plot(folder="./ICKAN_predictions")
plt.savefig("./ICKAN_predictions/KAN_splines.png")
