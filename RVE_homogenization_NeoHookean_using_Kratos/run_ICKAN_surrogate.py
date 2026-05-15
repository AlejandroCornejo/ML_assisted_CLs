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



"""
INPUT DATASET:
Load strain and stress from FOM trajectories (10 trajectories from stage_1_training_set_fom folder).
Data is loaded as [history, step, component] with shape [10, steps, 3].
"""
#***********************************************
min_steps = 300  # truncate all trajectories to the same length (e.g., 150 steps)
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

# ==========================================================================================
def TRAIN_KAN(model, optimizer, ref_strain_database, ref_W_database,
                ref_stress_database, n_epochs, max_W, patience=10, reduce_lr_factor=0.5,
                is_patient=True,
                train_W = False):

    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        def closure():
            optimizer.zero_grad()

            # Data loss: match predicted stress to reference stress
            if train_W:
                predicted_w = model.CalculateW(ref_strain_database)
                loss = torch.mean((predicted_w - ref_W_database) ** 2)
            else:
                normalized_stress = model.CalculateNormalizedStress(ref_strain_database) * max_W
                loss = torch.mean((normalized_stress - ref_stress_database) ** 2)

            loss.backward()
            return loss

        loss = optimizer.step(closure)

        # Check for very low loss (absolute early stopping)
        if loss.item() < 1e-4:
            print(f"Early stopping at epoch {epoch} with loss {loss.item():.6f}")
            break

        # Track best loss and patience
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

            # Reduce learning rate when patience is exhausted
            if patience_counter >= patience and not is_patient:
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = current_lr * reduce_lr_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Reducing learning rate from {current_lr} to {new_lr} at epoch {epoch}")
                patience_counter = 0  # Reset patience counter
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.8f}, Best Loss: {best_loss:.6f}, Patience: {patience_counter}/{patience}")
# ==========================================================================================


#*****************************************************************************************************************
#*****************************************************************************************************************
#*****************************************************************************************************************
n_epochs = 5000
learning_rate = 0.01

order_stretches = 1   # Number of orders (can be set to any value)
k = 2  # Degree of splines
grid_size = 4  # Number of knots

input_size = 2 * order_stretches + 1

W_width = [input_size,
            # input_size,
            1,
            1] # output always 1
#*****************************************************************************************************************
#*****************************************************************************************************************
#*****************************************************************************************************************

model = surrogate.ICKAN_W_Surrogate(
    order_stretches=order_stretches,
    grid_size=grid_size,
    k=k,
    W_width=W_width)

# model.UpdateGridFromSamples(train_strain_database)
print("Check null W at null strain: ", model.CalculateW(torch.zeros(1,3)))
print("Check null S at null strain: ", model.CalculateNormalizedStress(torch.zeros(1,3)))

optimizer_1 = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    # weight_decay=0.1
    )

print(20*"=")
print("\nStarting W based optimization...")
print(20*"=")
TRAIN_KAN(
    model               =  model,
    optimizer           =  optimizer_1,
    ref_strain_database =  train_strain_database,
    ref_W_database      =  train_W_database,
    ref_stress_database =  train_stress_database,
    n_epochs            =  n_epochs,
    max_W               =  max_W,
    patience            =  50,
    reduce_lr_factor    =  0.9,
    is_patient          =  False,
    train_W             =  True
)

# optimizer_2 = optim.LBFGS(
#                     model.parameters(),
#                     lr=learning_rate,
#                     max_iter=10,
#                     history_size=20,
#                     # line_search_fn='strong_wolfe'
#                     )
print(20*"=")
print("\nStarting stress based optimization...")
print(20*"=")
TRAIN_KAN(
    model               =  model,
    optimizer           =  optimizer_1,
    ref_strain_database =  train_strain_database,
    ref_W_database      =  train_W_database,
    ref_stress_database =  train_stress_database,
    n_epochs            =  n_epochs,
    max_W               =  max_W,
    patience            =  30,
    reduce_lr_factor    =  0.9,
    is_patient          =  True,
    train_W             =  False
)

torch.save(model.state_dict(), "ICKAN_predictions/ICKAN_model_weights.pth")

model.KAN_W.save_act = True
kan_input = model._compute_kan_input_for_strain(train_strain_database) 
predicted_w = model.KAN_W.forward(kan_input)
model.KAN_W.plot()
# plt.show()
plt.savefig("./ICKAN_predictions/ICKAN.png")
plt.close()

predicted_w = model.CalculateW(train_strain_database)

plt.plot(train_strain_database[:,0].detach().numpy(), train_W_database[:,0].detach().numpy(), '--', label='Reference W')
plt.plot(train_strain_database[:,0].detach().numpy(), predicted_w[:,0].detach().numpy(), '-', label='ICKAN W')
plt.xlabel('E_xx')
plt.ylabel('W')
plt.legend()
plt.savefig("./ICKAN_predictions/W_history.png")
plt.show()
plt.close()

predicted_stress = max_W * model.CalculateNormalizedStress(train_strain_database)

plt.plot(train_strain_database[:,0].detach().numpy(), train_stress_database[:,0].numpy(), '--', label='Reference S_xx')
plt.plot(train_strain_database[:,0].detach().numpy(), predicted_stress[:,0].detach().numpy(), '-', label='ICKAN S_xx')
plt.xlabel('E_xx')
plt.ylabel('S_xx')
plt.legend()
plt.savefig("./ICKAN_predictions/S_xx_history.png")
plt.show()
plt.close()
