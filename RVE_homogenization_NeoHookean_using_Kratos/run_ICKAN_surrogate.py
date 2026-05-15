import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import ICKAN_surrogate as surrogate



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


# ==========================================================================================
def TRAIN_KAN(model, optimizer, ref_strain_database, ref_W_database, ref_stress_database, n_epochs, gradient_penalty_weight):
    """
    Training function with gradient penalization at null input.
    
    Args:
        model: ICKAN_W_Surrogate model
        optimizer: Optimizer
        ref_strain_database: Reference strain data
        ref_W_database: Reference energy data
        n_epochs: Number of training epochs
        gradient_penalty_weight: Weight for gradient penalization term at null strain
    """
    # Precompute null strain input
    null_strain = torch.zeros(1,3)
    # max_W = 1
    max_W = train_W_database.abs().max()
    
    for epoch in range(n_epochs):
        def closure():
            optimizer.zero_grad()
            
            # Compute predicted stress from model: dW/dE at each strain point
            ref_strain_pred = ref_strain_database.requires_grad_(True)
            predicted_W_ref = model.forward(ref_strain_pred)
            predicted_stress = max_W * torch.autograd.grad(
                outputs=predicted_W_ref,
                inputs=ref_strain_pred,
                grad_outputs=torch.ones_like(predicted_W_ref),
                create_graph=True
            )[0]

            grad_null_strain = null_strain.requires_grad_(True)
            predicted_W_null = model.forward(grad_null_strain)
            predicted_stress_null = max_W * torch.autograd.grad(
                outputs=predicted_W_null,
                inputs=grad_null_strain,
                grad_outputs=torch.ones_like(predicted_W_null),
                create_graph=True
            )[0]

            # Data loss: match predicted stress to reference stress
            loss = torch.mean((predicted_stress - predicted_stress_null - ref_stress_database) ** 2)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
# ==========================================================================================



"""
INPUT DATASET:
Load strain and stress from FOM trajectories (10 trajectories from stage_1_training_set_fom folder).
Data is loaded as [history, step, component] with shape [10, steps, 3].
"""
#***********************************************
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

# Compute W using trapezoidal rule for better integration accuracy
# Work increment: dW = 0.5 * (sigma_i + sigma_{i-1}) : d(epsilon_i - epsilon_{i-1})
# This averages stress at start and end of each increment (trapezoidal rule)

# Keep trajectory structure for per-trajectory cumsum: shape [batches, steps, 3]
batch_strain = ref_strain_database  # [10, steps, 3]
batch_stress = ref_stress_database  # [10, steps, 3]

# Compute strain increments per trajectory: shape [batches, steps, 3]
delta_strain_batch = torch.zeros_like(batch_strain)
delta_strain_batch[:, 1:, :] = batch_strain[:, 1:, :] - batch_strain[:, :-1, :]

# Trapezoidal rule: average stress at start and end of each increment
# stress_avg at step i uses (stress[i] + stress[i-1]) / 2
stress_avg = 0.5 * (batch_stress[:, 1:, :] + batch_stress[:, :-1, :])

# W increment per trajectory step: shape [batches, steps-1, 1]
W_increment_batch = (stress_avg * delta_strain_batch[:, 1:, :]).sum(dim=2, keepdim=True)

# Cumulative sum per trajectory, starting from zero energy at reference state
# Shape: [batches, steps-1, 1]
W_cumulative_batch = torch.cumsum(W_increment_batch, dim=1)  # [10, steps-1, 1]

# Prepend zero (W=0 at reference state) and reshape to [batches*steps, 1]
W_zero = torch.zeros(batch_strain.shape[0], 1, 1, device=W_cumulative_batch.device)  # [10, 1, 1]
W_full_batch = torch.cat([W_zero, W_cumulative_batch], dim=1)  # [10, steps, 1]
train_W_database = W_full_batch.view(-1, 1)  # [10*steps, 1]



# max_W = 1
max_W = train_W_database.abs().max()
# max_W = 1
train_W_database /= max_W  # Normalize W to have max absolute value of 1



#**********************************************************
#**********************************************************
#**********************************************************
n_epochs = 500
learning_rate = 0.1


order_stretches = 1   # Number of orders (can be set to any value)
k = 2  # Degree of splines
grid = 3  # Number of knots
input_size = 2 * order_stretches + 1

W_width = [input_size,  1] # output always 1
W_width = [input_size, input_size, input_size, 1] # output always 1
#**********************************************************
#**********************************************************
#**********************************************************



model = surrogate.ICKAN_W_Surrogate(order_stretches=order_stretches, grid=grid, k=k, W_width=W_width)
# model.UpdateGridFromSamples(train_strain_database)

optimizer_1 = optim.LBFGS(
                    model.parameters(),
                    lr=learning_rate,
                    max_iter=20,
                    history_size=30)

# optimizer_1 = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

TRAIN_KAN(model, optimizer_1, train_strain_database, train_W_database, train_stress_database, n_epochs, 0.01)

print("Check null W at null strain: ", model.forward(torch.zeros(1,3)))


predicted_w = model.forward(train_strain_database)

plt.plot(train_strain_database[:,0].detach().numpy(), train_W_database[:,0].detach().numpy(), '--', label='Reference W')
plt.plot(train_strain_database[:,0].detach().numpy(), predicted_w[:,0].detach().numpy(), '-', label='ICKAN W')
plt.xlabel('E_xx')
plt.ylabel('W')
plt.legend()
plt.savefig("./ICKAN_predictions/W_history.png")
plt.show()
plt.close()

# Enable gradients for strain database to compute dW/dE using autograd
train_strain_database = train_strain_database.requires_grad_(True)
predicted_w = model.forward(train_strain_database)

predicted_stress = torch.autograd.grad(
            outputs=predicted_w,
            inputs=train_strain_database,
            grad_outputs=torch.ones_like(predicted_w),
            create_graph=True
            )[0]

null_train_strain_database = torch.zeros_like(train_strain_database).requires_grad_(True)
predicted_w_0 = model.forward(null_train_strain_database)
stress_0 = torch.autograd.grad(
            outputs=predicted_w_0,
            inputs=null_train_strain_database,
            grad_outputs=torch.ones_like(predicted_w_0),
            create_graph=True
            )[0]

stress = max_W * (predicted_stress - stress_0)
# stress = max_W*(predicted_stress)

plt.plot(train_strain_database[:,0].detach().numpy(), train_stress_database[:,0].numpy(), '--', label='Reference S_xx')
plt.plot(train_strain_database[:,0].detach().numpy(), stress[:,0].detach().numpy(), '-', label='ICKAN S_xx')
plt.xlabel('E_xx')
plt.ylabel('S_xx')
plt.legend()
plt.savefig("./ICKAN_predictions/S_xx_history.png")
plt.show()
plt.close()

