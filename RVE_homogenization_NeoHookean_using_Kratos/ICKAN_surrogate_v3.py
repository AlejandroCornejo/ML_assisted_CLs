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

"""
In here we now change the strategy... we transform the E,S input to E,W.
Then the KAN is trained to predict W instead of S. The loss is still computed on the S values,
but the KAN is now predicting W, which is a more direct measure of the material response 
and should be easier to learn.

The we return the S using autograd of the trained KAN, which should be more accurate than 
the direct S prediction from the previous version.

"""

class ICKAN_W_Surrogate(nn.Module):

    def __init__(self, order_stretches, grid, k):

        super(ICKAN_W_Surrogate, self).__init__()

        self.order_stretches = order_stretches
        self.input_size = 2 * self.order_stretches + 1  # Total inputs: 2 * reg_eigenvalues for each order + 1 * log(J)
        self.grid = grid
        self.k = k

        grid = []
        for i in range(self.input_size):
            grid.append([-0, 1])

        # KAN definition for the energy density potential W
        self.KAN_W = KAN.MultKAN(
            width=[self.input_size, 1], # output of size 1: W
            grid_range_0=grid,
            grid_range=grid,
            # base_fun = "identity"
        )

        self.KAN_W.speed()

        # Initialize some extra parameters
        self.ki = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for p in range(self.order_stretches + 1)
        ])

        # The parameter multiplying the log(J) is initially set to 1.0
        self.ki[-1] = nn.Parameter(torch.tensor(1.0))

    # ==========================================================================================

    def UpdateGridFromSamples(self, strain_database):
        kan_input = self._compute_kan_input_for_strain(strain_database)
        self.KAN_W.update_grid_from_samples(kan_input)

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

        W_raw = self.KAN_W.forward(kan_input)  # Shape: (batch x steps, 1)

        null_kan_input = self._compute_kan_input_for_strain(torch.zeros_like(strain_database))
        W0 = self.KAN_W.forward(null_kan_input)

        return W_raw - W0
    # ==========================================================================================

    def forward(self, strain_database):
        return self.CalculateW(strain_database)

    # ==========================================================================================


# ==========================================================================================
def TRAIN_KAN(model, optimizer, ref_strain_database, ref_W_database, n_epochs):
    # Precompute dW_dI_null ONCE before training to avoid graph recreation each epoch

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass: compute predicted W
        predicted_W = model.forward(ref_strain_database)

        # Compute L2 loss between predicted W and reference W
        loss = torch.mean((predicted_W - ref_W_database) ** 2)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
# ==========================================================================================



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
ref_stress_database /= ref_stress_database.abs().max()  # Normalize stress to have max absolute value of 1
ref_strain_database /= ref_strain_database.abs().max()  # Normalize strain to have max absolute value of 1

# Use all data for training (no train/test split)
# reshape for optimal loops
train_strain_database = ref_strain_database.view(-1, 3)
train_stress_database = ref_stress_database.view(-1, 3)

print("\nLaunching the training of a KAN...")
print(f"Number of training trajectories: {train_strain_database.shape[0]}")
print(f"Number of total trajectories: {ref_strain_database.shape[0]}")
print(f"Number of steps: {ref_strain_database.shape[1]}")
print(f"Strain size: {ref_strain_database.shape[2]}\n")

train_W_database = torch.zeros(train_strain_database.shape[0], 1)  # Placeholder for W values

train_W_database[:, 0] = 0.5 * (train_stress_database[:,0] * train_strain_database[:,0] +
                                train_stress_database[:,1] * train_strain_database[:,1] + 
                                train_stress_database[:,2] * train_strain_database[:,2])

train_W_database /= train_W_database.abs().max()  # Normalize W to have max absolute value of 1




#*****************************
#*****************************
#*****************************
n_epochs = 1000
learning_rate = 0.1


order_stretches = 1   # Number of orders (can be set to any value)
k = 2  # Degree of splines
grid = 3  # Number of knots
#*****************************
#*****************************
#*****************************



model = ICKAN_W_Surrogate(order_stretches=order_stretches, grid=grid, k=k)
# model.UpdateGridFromSamples(train_strain_database)

optimizer_1 = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
TRAIN_KAN(model, optimizer_1, train_strain_database, train_W_database, n_epochs)

print("Check null W at null strain: ", model.forward(torch.zeros(1,3)))


predicted_w = model.forward(train_strain_database)

plt.plot(train_strain_database[:,0].numpy(), train_W_database[:,0].numpy(), '--', label='Reference W')
plt.plot(train_strain_database[:,0].numpy(), predicted_w[:,0].detach().numpy(), '-', label='ICKAN W')
plt.xlabel('E_xx')
plt.ylabel('W')
plt.legend()
plt.savefig("./ICKAN_predictions/W_history.png")
plt.close()