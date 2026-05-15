import numpy as np
import scipy as sp
import torch as torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.insert(0, r'C:\ICKANs')

import ickan as KAN

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
    

# ==========================================================================================
# ==========================================================================================

class KANStressPredictor(nn.Module):
    """
    KAN-based Stress Predictor with support for multiple orders of stretches.
    """

    def __init__(self):
        super(KANStressPredictor, self).__init__()

        # EDIT:
        self.order_stretches = 1   # Number of orders (can be set to any value)
        self.k = 2  # Degree of splines
        self.grid = 3  # Number of knots
        # -------------------------------------

        self.input_size = 2 * self.order_stretches + 1  # Total inputs: 2 * reg_eigenvalues for each order + 1 * log(J)

        grid = []
        for i in range(self.input_size):
            grid.append([-1, 1])

        # KAN definition
        self.KAN_W = KAN.MultKAN(
            width=[self.input_size, self.input_size, self.input_size, 1], # output of size 1: W
            grid=self.grid,
            k=self.k,
            grid_range_0=grid,
            grid_range=grid,
            base_fun = "identity"
        )
        self.KAN_W.speed()

        # Initialize some extra parameters
        self.ki = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for p in range(self.order_stretches + 1)
        ])

        # The parameter multiplying the log(J) is initially set to 1.0
        self.ki[-1] = nn.Parameter(torch.tensor(1.0))
        
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
        log_J = J**2
        log_J = torch.log(J + 1.0e-12)

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

        # print("KAN input in CalculateW: ", kan_input)

        W_raw = self.KAN_W.forward(kan_input)  # Shape: (batch x steps, 1)

        # print("W_raw in CalculateW: ", W_raw)
        W0 = self.KAN_W.forward(torch.zeros_like(kan_input))
        return W_raw - W0

    # ==========================================================================================
    def forward(self, strain_database):
        """
        This method returns the stress by applying the chain rule to compute dW/dE = dW/dI * dI/dE
        kan_input: Tensor of shape (batches*steps, input_size) with KAN inputs
        dI_dE: Tensor of shape (batches*steps, 3, 3) with derivatives of invariants w.r.t. strain components
        dW_dI_null: Optional precomputed gradient at null input (1, input_size) for normalization
        Returns: Tensor of shape (batches*steps, 3) with predicted stress components
        
        For that we compute the gradients of W w.r.t invariants using autograd, and then apply the chain rule to get dW/dE.
        """
        W = self.CalculateW(strain_database)  # Shape: (batches*steps, 1)

        grad_kan_input = torch.autograd.grad(
            outputs=W,
            inputs=strain_database,
            grad_outputs=torch.ones_like(W),
            create_graph=True
        )[0]

        zeros = torch.zeros_like(strain_database).requires_grad_(True)
        W0 = self.CalculateW(zeros)  # Shape: (batches*steps, 1)

        grad_kan_input_0 = torch.autograd.grad(
            outputs=W0,
            inputs=zeros,
            grad_outputs=torch.ones_like(W0),
            create_graph=True
        )[0]
        
        return grad_kan_input - grad_kan_input_0

    #=============================================================================================================
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
#=============================================================================================================

def TRAIN_KAN(model, optimizer, ref_strain_database, ref_stress_database, n_epochs):
    # Precompute dW_dI_null ONCE before training to avoid graph recreation each epoch

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass: compute predicted stress
        predicted_stress = model.forward(ref_strain_database)

        # Compute L2 loss between predicted stress and reference stress
        loss = torch.mean((predicted_stress - ref_stress_database) ** 2)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

#=============================================================================================================



"""
INPUT DATASET:
Load strain and stress from FOM trajectories (10 trajectories from stage_1_training_set_fom folder).
Data is loaded as [history, step, component] with shape [10, steps, 3].
"""
#***********************************************
min_steps = 250  # truncate all trajectories to the same length (e.g., 150 steps)
#***********************************************

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
# min_steps = min(t.shape[0] for t in strain_trajectories)
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
print(f"Strain size: {train_strain_database.shape[2]}\n")

# reshape for optimal loops
ref_strain_database = ref_strain_database.view(-1, 3) # Reshape to (batches*steps, input_size)
# print("ref_strain_database shape= ", ref_strain_database.shape)
ref_stress_database = ref_stress_database.view(-1, 3) # Reshape to (batches*steps, input_size)


ref_strain_database = ref_strain_database.detach().requires_grad_(True) # Enable gradients for strain database to compute dW/dI using autograd

# Create the torch model
model = KANStressPredictor()
# model.KAN_W.speed()

# Here we compute the KAN inputs, invariants
kan_input = model._compute_kan_input_for_strain(ref_strain_database)
# print(f"KAN input shape: {kan_input.shape}")
# print("KAN grid updated with computed KAN inputs from the strain database.")
# model.KAN_W.update_grid(kan_input) # Update KAN grid with the computed KAN inputs (invariants) from the strain database

# Check null stress at null strain
print("\nChecking null stress at null strain...")
null_strain = torch.zeros((1, 3), requires_grad=True)  # Shape
print("null stress = ", model.forward(null_strain))

# output = model.forward(ref_strain_database)
# print("output size: ", output.shape)
# print("output type: ", output[25,:])


kan_input = model._compute_kan_input_for_strain(ref_strain_database)
model.KAN_W.update_grid_from_samples(kan_input)


#*****************************
n_epochs = 1000
learning_rate = 0.001
#*****************************

optimizer_1 = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
TRAIN_KAN(model, optimizer_1, ref_strain_database, ref_stress_database, n_epochs)




# ==========================================================================================
# PLOTTING: Create batch plots showing Strain vs Stress (Reference vs Predicted)
# ==========================================================================================


# Create the folder to save the plots if it doesn't exist
output_folder = "ICKAN_predictions"
os.makedirs(output_folder, exist_ok=True)
torch.save(model.state_dict(), "ICKAN_predictions/ICKAN_model_weights.pth")


# CRITICAL: Recompute KAN inputs and update grid after training with learned parameters
# kan_input_final = model._compute_kan_input_for_strain(ref_strain_database)
# model.KAN_W.update_grid(kan_input_final)
print("Grid updated with final KAN inputs after training.")

# Now enable activation saving and compute forward pass for plotting
# model.KAN_W.save_act = True
# with torch.no_grad():
#     model.KAN_W.forward(kan_input_final)
# model.KAN_W.plot(folder="./ICKAN_predictions")
# plt.savefig("./ICKAN_predictions/KAN_splines.png")
# plt.close()





# Restore original shapes for plotting (before reshaping to batches*steps)
# Original: [10 trajectories, 150 steps, 3 components]
num_trajectories = 10
steps_per_trajectory = min_steps


predicted_stress = model.forward(ref_strain_database).detach().numpy()  # Shape: (batches*steps, 3)


plt.plot(ref_strain_database[:, 0].detach().numpy(), ref_stress_database[:, 0].detach().numpy(), 'ro', label='Reference Stress Component 0', color=GetColor(0), linestyle="")
plt.plot(ref_strain_database[:, 1].detach().numpy(), ref_stress_database[:, 1].detach().numpy(), 'bo', label='Reference Stress Component 1', color=GetColor(1), linestyle="")
plt.plot(ref_strain_database[:, 2].detach().numpy(), ref_stress_database[:, 2].detach().numpy(), 'go', label='Reference Stress Component 2', color=GetColor(2), linestyle="")
plt.plot(ref_strain_database[:, 0].detach().numpy(), predicted_stress[:, 0], 'r-', label='Predicted Stress Component 0', color=GetColor(0), linestyle="-")
plt.plot(ref_strain_database[:, 1].detach().numpy(), predicted_stress[:, 1], 'b-', label='Predicted Stress Component 1', color=GetColor(1), linestyle="-")
plt.plot(ref_strain_database[:, 2].detach().numpy(), predicted_stress[:, 2], 'g-', label='Predicted Stress Component 2', color=GetColor(2), linestyle="-")

plt.xlabel('Strain')
plt.ylabel('Stress')
plt.legend()
plt.show()



model.CalculateW(ref_strain_database)
model.plot_spline_edges()
model.KAN_W.plot(folder="./ICKAN_predictions")
plt.savefig("./ICKAN_predictions/KAN_splines.png")
print("\nAll batch plots saved to ICKAN_predictions folder.")