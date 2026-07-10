import numpy as np
import torch as torch
import os
import argparse

_SCRIPT_DIR_FOR_IMPORTS = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_SCRIPT_DIR_FOR_IMPORTS, ".mplconfig"))

import matplotlib.pyplot as plt
import torch.optim as optim
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
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Sebastian test runner for Alejandro's ICKAN/KAN surrogate."
    )
    p.add_argument(
        "--fom-dir",
        type=str,
        default=os.path.join(repo_root, "stage_1_training_set_fom"),
        help="Directory containing trajectory_i FOM folders.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(script_dir, "ICKAN_predictions"),
        help="Output directory. Relative paths are interpreted inside Sebastian_ICKAN_Tests.",
    )
    p.add_argument(
        "--strain-source",
        type=str,
        default="strain",
        choices=["strain", "applied_strain"],
        help="Use homogenized computed strain or imposed applied strain as KAN input.",
    )
    p.add_argument(
        "--min-steps",
        type=int,
        default=500,
        help="Number of equally spaced samples per trajectory.",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    p.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate.")
    p.add_argument(
        "--skip-kan-plot",
        action="store_true",
        help="Skip KAN spline plot generation; useful for quick smoke tests.",
    )
    return p.parse_args()


args = parse_args()

fom_trajectories_dir = os.path.abspath(args.fom_dir)
out_dir = args.out_dir
if not os.path.isabs(out_dir):
    out_dir = os.path.join(script_dir, out_dir)
out_dir = os.path.abspath(out_dir)
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "splines"), exist_ok=True)

requested_min_steps = int(args.min_steps)
strain_source = str(args.strain_source)

print(f"[SEBASTIAN-ICKAN] backend module       : {getattr(surrogate, 'KAN_BACKEND', 'unknown')}")
print(f"[SEBASTIAN-ICKAN] script dir           : {script_dir}")
print(f"[SEBASTIAN-ICKAN] FOM input dir        : {fom_trajectories_dir}")
print(f"[SEBASTIAN-ICKAN] output dir           : {out_dir}")
print(f"[SEBASTIAN-ICKAN] strain source        : {strain_source}")
print(f"[SEBASTIAN-ICKAN] requested samples/traj: {requested_min_steps}")

# Load all 10 trajectories
strain_trajectories = []
stress_trajectories = []

number_histories = 10
for i in range(1, number_histories + 1):  # trajectory_1 to trajectory_10
    strain_file = os.path.join(fom_trajectories_dir, f'trajectory_{i}', f'trajectory_{i}_{strain_source}.npy')
    stress_file = os.path.join(fom_trajectories_dir, f'trajectory_{i}', f'trajectory_{i}_stress.npy')
    
    strain_data = np.load(strain_file)  # shape: (steps, 3)
    stress_data = np.load(stress_file)  # shape: (steps, 3)

    strain_trajectories.append(strain_data)
    stress_trajectories.append(stress_data)
    print(f"Loaded trajectory_{i}: strain={strain_data.shape}, stress={stress_data.shape}")

# Choose a common sample count across all trajectories.
available_min_steps = min(t.shape[0] for t in strain_trajectories)
min_steps = min(requested_min_steps, available_min_steps)
print(f"\nMinimum available number of steps across all trajectories: {available_min_steps}")
print(f"Using {min_steps} equally spaced samples per trajectory.")

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

def L2_relative_error(pred, target):
    # return torch.sqrt(torch.mean(((pred - target) / (target + 1.0e-12)) ** 2))
    return (torch.mean((pred - target) ** 2)) / ((torch.mean(target ** 2)) + 1.0e-12)

# ==========================================================================================
def TRAIN_KAN(
    model,
    optimizer,
    ref_strain_database,
    ref_W_database,
    ref_stress_database,
    n_epochs, max_W,
    patience=10, 
    reduce_lr_factor=0.5,
    reset_model_after_patience=False,
    is_patient=True,
    train_W = False,
    mixed_sovolev_training = False,
    mixed_sovolev_W_loss_weight = 0.5,
    early_stopping_threshold = 1e-4,
    minimum_lr = 1.0e-4,
    verbose_interval = 100,
    update_grid=False,
    grid_update_interval = 500,
    initial_step_grid_update = 1000,
    final_step_grid_update = 2000
    ):

    best_loss = float('inf')
    patience_counter = 0
    grid_update_counter = 0

    for epoch in range(n_epochs):
        def closure():
            optimizer.zero_grad()

            # Data loss: match predicted stress to reference stress
            if mixed_sovolev_training:
                predicted_w = model.CalculateW(ref_strain_database)
                loss_W = L2_relative_error(predicted_w, ref_W_database)  # Relative W loss

                normalized_stress = model.CalculateNormalizedStress(ref_strain_database) * max_W
                loss_S = L2_relative_error(normalized_stress, ref_stress_database)

                loss = mixed_sovolev_W_loss_weight * (loss_W) + (1.0 - mixed_sovolev_W_loss_weight) * (loss_S)
            else:
                if train_W:
                    predicted_w = model.CalculateW(ref_strain_database)
                    loss = L2_relative_error(predicted_w, ref_W_database)  # Relative W loss
                else:
                    normalized_stress = model.CalculateNormalizedStress(ref_strain_database) * max_W
                    loss = L2_relative_error(normalized_stress, ref_stress_database)

            loss.backward()
            return loss

        loss = optimizer.step(closure)

        grid_update_counter += 1

        # Check for very low loss (absolute early stopping)
        if loss.item() < early_stopping_threshold:
            print(f"Early stopping at epoch {epoch} with loss {loss.item():.8E}")
            break

        # Track best loss and patience
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            if reset_model_after_patience:
                best_parameters = {k: v.clone() for k, v in model.state_dict().items()}  # Save best model parameters
        else:
            patience_counter += 1
            # Reduce learning rate when patience is exhausted
            if patience_counter >= patience and not is_patient:
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = current_lr * reduce_lr_factor

                if new_lr < minimum_lr:
                    new_lr = minimum_lr

                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"\tReducing learning rate from {current_lr:.7E} to {new_lr:.7E} at epoch {epoch}")

                if reset_model_after_patience:
                    model.load_state_dict(best_parameters)  # Revert to best parameters
                    print(f"\t\tReverting to best model parameters with loss {best_loss:.4E}")
                patience_counter = 0  # Reset patience counter

        if epoch % verbose_interval == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.8E}, Best Loss: {best_loss:.6E}, Patience: {patience_counter}/{patience}")
        
        if epoch == n_epochs - 1:
            print(f"Reached maximum epochs. Final Loss: {loss.item():.4E}, Best Loss: {best_loss:.4E}")
            if not is_patient:
                if loss.item() < best_loss:
                    print(f"Final model is the best model with loss {loss.item():.6E}")
                    model.load_state_dict(best_parameters)  # Revert to best parameters

        if update_grid and epoch >= initial_step_grid_update and epoch <= final_step_grid_update and grid_update_counter >= grid_update_interval:
            model.UpdateGridFromSamples(ref_strain_database)
            print(f"\t Grid updated from samples at epoch {epoch}")
            grid_update_counter = 0

# ==========================================================================================


#*****************************************************************************************************************
#*****************************************************************************************************************
#*****************************************************************************************************************
n_epochs = 100
learning_rate = 1.0e-3
if args.epochs is not None:
    n_epochs = int(args.epochs)
if args.learning_rate is not None:
    learning_rate = float(args.learning_rate)

order_stretches = 1   # Number of orders (can be set to any value)
k = 3  # Degree of splines
grid_size = 100  # Number of knots

input_size = 2 * order_stretches + 1
W_width = [input_size,
            5,
            4,
            1,
            1] # output always 1

#*****************************************************************************************************************
#*****************************************************************************************************************
#*****************************************************************************************************************

model = surrogate.ICKAN_W_Surrogate(
    order_stretches=order_stretches,
    grid_size=grid_size,
    k=k,
    W_width=W_width
)

print("\nInitial KAN grid update:")
model.UpdateGridFromSamples(train_strain_database)


# model.KAN_W.plot(tick=True)
# plt.show()

print("Check null W at null strain: ", model.CalculateW(torch.zeros(1,3)))
print("Check null S at null strain: ", model.CalculateNormalizedStress(torch.zeros(1,3)))


optimizer_1 = optim.LBFGS( #LBFGS
    model.parameters(),
    lr=learning_rate,
    line_search_fn="strong_wolfe"
)

print(20*"=")
print("\nStarting stress based optimization...")
print(20*"=")

#------------------------------------------------------------------------------------
TRAIN_KAN(
    model                       = model,
    optimizer                   = optimizer_1,
    ref_strain_database         = train_strain_database,
    ref_W_database              = train_W_database,
    ref_stress_database         = train_stress_database,
    n_epochs                    = n_epochs,
    max_W                       = max_W,

    is_patient                  = True,
    patience                    = 50,
    reduce_lr_factor            = 0.75,
    minimum_lr                  = 1.0e-6,

    train_W                     = False,
    early_stopping_threshold    = 1.0e-3,
    mixed_sovolev_training      = False,
    mixed_sovolev_W_loss_weight = 0.05, # 1 is only W loss, 0 is only S loss

    update_grid = True,
    grid_update_interval = 10,
    initial_step_grid_update = 10,
    final_step_grid_update = 50,
    verbose_interval = 1
)
#------------------------------------------------------------------------------------

weights_path = os.path.join(out_dir, "ICKAN_model_weights.pth")
torch.save(model.state_dict(), weights_path)

model.KAN_W.save_act = True
kan_input = model._compute_kan_input_for_strain(train_strain_database) 
predicted_w = model.KAN_W.forward(kan_input)

# model.KAN_W.update_grid(kan_input)

if not args.skip_kan_plot:
    model.KAN_W.plot(
                    folder=os.path.join(out_dir, "splines"),
                    tick=True,
                    scale=10.0,
                    varscale=0.05
                    )
    # plt.show()
    plt.savefig(os.path.join(out_dir, "ICKAN.png"))
    plt.close()

predicted_w = model.CalculateW(train_strain_database)

plt.plot(train_strain_database[:,0].detach().numpy(), train_W_database[:,0].detach().numpy(), '--', label='Reference W')
plt.plot(train_strain_database[:,0].detach().numpy(), predicted_w[:,0].detach().numpy(), '-', label='ICKAN W')
plt.xlabel('Strain XX')
plt.ylabel('Elastic Energy Density W')
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, "W_history_x.png"))
# plt.show()
plt.close()

plt.plot(train_strain_database[:,1].detach().numpy(), train_W_database[:,0].detach().numpy(), '--', label='Reference W')
plt.plot(train_strain_database[:,1].detach().numpy(), predicted_w[:,0].detach().numpy(), '-', label='ICKAN W')
plt.xlabel('Strain YY')
plt.ylabel('Elastic Energy Density W')
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, "W_history_y.png"))
# plt.show()
plt.close()

plt.plot(train_strain_database[:,2].detach().numpy(), train_W_database[:,0].detach().numpy(), '--', label='Reference W')
plt.plot(train_strain_database[:,2].detach().numpy(), predicted_w[:,0].detach().numpy(), '-', label='ICKAN W')
plt.xlabel('Strain XY')
plt.ylabel('Elastic Energy Density W')
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, "W_history_xy.png"))
# plt.show()
plt.close()

predicted_stress = max_W * model.CalculateNormalizedStress(train_strain_database)

plt.plot(train_strain_database[:,0].detach().numpy(), train_stress_database[:,0].numpy(), '--', label='Reference S_xx',)
plt.plot(train_strain_database[:,0].detach().numpy(), predicted_stress[:,0].detach().numpy(), '-', label='ICKAN S_xx' )
plt.xlabel('Strain XX')
plt.ylabel('Normalized Stress XX')
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, "S_xx_history.png"))
# plt.show()
plt.close()

plt.plot(train_strain_database[:,1].detach().numpy(), train_stress_database[:,1].numpy(), '--', label='Reference S_yy')
plt.plot(train_strain_database[:,1].detach().numpy(), predicted_stress[:,1].detach().numpy(), '-', label='ICKAN S_yy' )
plt.xlabel('Strain YY')
plt.ylabel('Normalized Stress YY')
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, "S_yy_history.png"))
# plt.show()
plt.close()

plt.plot(train_strain_database[:,2].detach().numpy(), train_stress_database[:,2].numpy(), '--', label='Reference S_xy')
plt.plot(train_strain_database[:,2].detach().numpy(), predicted_stress[:,2].detach().numpy(), '-', label='ICKAN S_xy' )
plt.xlabel('Strain XY')
plt.ylabel('Normalized Stress XY')
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, "S_xy_history.png"))
# plt.show()
plt.close()


print(f"\nFinished training and plotting results. Model weights saved to '{weights_path}'.")

print("*"*60)
print("*"*60)
