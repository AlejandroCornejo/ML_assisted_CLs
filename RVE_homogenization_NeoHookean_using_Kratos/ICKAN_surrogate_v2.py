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
            width=[self.input_size,  1], # output of size 1: W
            grid=self.grid,
            k=self.k,
            grid_range_0=grid,
            grid_range=grid
        )

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
        return KAN_input.view(-1, self.input_size) # Reshape to (batches*steps, input_size)

    # ==========================================================================================
    def CalculateW(self, kan_input):
        """
        Computes W for the given strain with normalization.
        """
        # Pass the input through the KAN layer
        # W0 = self.KAN_W.forward(torch.zeros_like(kan_input[0, :]).unsqueeze(0)) # compute W0 at the null strain input (reg_eigenvalues=1.0, log_J=0.0) for one sample (shape: (1,1))

        # Normalization (if needed, can be implemented here)
        # For now, we return W directly without normalization
        W_raw = self.KAN_W.forward(kan_input)  # Shape: (batch x steps, 1)
        # W_raw[:, 0] = W_raw[:, 0] - W0  # Subtract W0 from all samples to normalize
        return W_raw

    # ==========================================================================================
    def forward(self, kan_input, dI_dE):
        """
        This method returns the stress by applying the chain rule to compute dW/dE = dW/dI * dI/dE
        kan_input: Tensor of shape (batches*steps, input_size) with KAN inputs
        dI_dE: Tensor of shape (batches*steps, 3, 3) with derivatives of invariants w.r.t. strain components
        dW_dI_null: Optional precomputed gradient at null input (1, input_size) for normalization
        Returns: Tensor of shape (batches*steps, 3) with predicted stress components
        
        For that we compute the gradients of W w.r.t invariants using autograd, and then apply the chain rule to get dW/dE.
        """
        W = self.CalculateW(kan_input)  # Shape: (batches*steps, 1)

        # Compute gradients of W w.r.t. KAN inputs (invariants)
        dW_dI = torch.autograd.grad(
            outputs=W,
            inputs=kan_input,
            grad_outputs=torch.ones_like(W),
            create_graph=True
        )[0]  # Shape: (batches*steps, input_size)

        # Apply chain rule to get dW/dE using normalized dW_dI
        # dW_dE = torch.zeros((kan_input.shape[0], 3))  # Initialize dW/dE tensor
        # for i in range(kan_input.shape[0]):
        #     dW_dE[i, :] = torch.matmul(dI_dE[i, :, :].T, dW_dI[i, :])  # Shape: (3,)
        
        return dW_dI # normalized dW/dE as predicted stress

#=============================================================================================================

def TRAIN_KAN(model, optimizer, kan_input_database, ref_stress_database, dI_dE_database, n_epochs):
    # Precompute dW_dI_null ONCE before training to avoid graph recreation each epoch

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass: compute predicted stress
        predicted_stress = dI_dE_database @ model.forward(kan_input_database, dI_dE_database).T   # Shape: (batches*steps, 3)

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
n_epochs = 500
learning_rate = 0.01
min_steps = 150  # truncate all trajectories to the same length (e.g., 150 steps)
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

# Create the torch model
model = KANStressPredictor()
model.KAN_W.speed()

# Here we compute the KAN inputs, invariants
kan_input = model._compute_kan_input_for_strain(ref_strain_database)
print(f"KAN input shape: {kan_input.shape}")


# Remove [0] indexing: jacobian returns a tensor directly (not a list) when input is a single tensor.
# Full Jacobian shape: (1500, 3, 1500, 3) where grads[i, a, j, b] = d(kan_input[i, a]) / d(strain[j, b])
grads = torch.autograd.functional.jacobian(
    lambda x: model._compute_kan_input_for_strain(x),
    ref_strain_database
)
# print(f"Full Jacobian shape: {grads.shape}")

# Extract per-sample 3x3 Jacobians: d(kan_input[i]) / d(strain[i]) for each sample i
# Since each output point depends only on its own strain point (no cross-batch coupling),
# the diagonal blocks grads[i, :, i, :] give the per-sample Jacobian.
per_sample_jacobian = torch.stack([grads[i, :, i, :] for i in range(ref_strain_database.shape[0])])
# print(f"Per-sample Jacobian shape: {per_sample_jacobian.shape}")  # (1500, 3, 3)

output = model.forward(kan_input, per_sample_jacobian)
print("output size: ", output.shape)

print("KAN grid updated with computed KAN inputs from the strain database.")
model.KAN_W.update_grid(kan_input) # Update KAN grid with the computed KAN inputs (invariants) from the strain database

optimizer_1 = optim.Adam(model.parameters(), lr=learning_rate)

TRAIN_KAN(model, optimizer_1, kan_input, ref_stress_database, per_sample_jacobian, n_epochs)