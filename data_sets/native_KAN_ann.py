# std libs imports
import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import kan as KAN
import random
import cl_loader as cl_loader
from sklearn.model_selection import train_test_split

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

# =============================================================================================================
"""
INPUT DATASET:
"""
n_epochs = 100
learning_rate = 0.05
number_of_steps = 25
ADD_NOISE = False
database = cl_loader.CustomDataset("neo_hookean_hyperelastic_law/raw_data", number_of_steps, None, ADD_NOISE)
# =============================================================================================================

ref_strain_database = torch.stack([item[0] for item in database])  # batch x steps x strain_size
ref_stress_database = torch.stack([item[1] for item in database])  # batch x steps x strain_size
ref_work_database = torch.stack([item[2] for item in database])  # batch x steps x 1

ref_stress_database /= 1.0e6  # to MPa
ref_work_database /= 1.0e6  # to MPa vs strain

# Split the dataset into training and testing datasets
train_indices, test_indices = train_test_split(
    range(len(database)), test_size=0.05, random_state=42  # 0.2 for 20% test size
)

train_strain_database = ref_strain_database[train_indices]
train_work_database = ref_work_database[train_indices]

test_strain_database = ref_strain_database[test_indices]
test_work_database = ref_work_database[test_indices]

print("\nLaunching the training of a KAN...")
print("Number of training batches: ", train_strain_database.shape[0])
print("Number of total batches: ", ref_strain_database.shape[0])
print("Number of steps  : ", train_strain_database.shape[1])
print("Strain size      : ", train_strain_database.shape[2])

# ==========================================================================================

class KANStressPredictor(nn.Module):
    """
    KAN-based Stress Predictor with support for multiple orders of stretches.
    """

    def __init__(self):
        super(KANStressPredictor, self).__init__()

        self.order_stretches = 1  # Number of orders (can be set to any value)
        self.k = 3  # Degree of splines
        self.grid = 3  # Number of knots

        self.input_size = 2 * self.order_stretches + 1  # Total inputs: 2 * reg_eigenvalues for each order + 1 * log(J)

        # KAN definition
        self.KAN_W = KAN.MultKAN(
            width=[self.input_size, 1, 1],
            grid=self.grid,
            k=self.k
        )

        # Initialize some extra parameters
        self.ki = nn.ParameterList([
            nn.Parameter(torch.tensor(float(p + 1) + random.random())) for p in range(self.order_stretches + 1)
        ])
        self.ki[-1] = nn.Parameter(torch.tensor(1.0))  # Parameter multiplying log(J)

    def prepare_input(self, strain):
        """
        Prepares the input tensor (stretches and log(J)) for the KAN model.
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
        kan_inputs = []
        for index in range(self.order_stretches):
            reg_eigenvalues_order = reg_eigenvalues ** self.ki[index]
            kan_inputs.append(reg_eigenvalues_order)

        log_J_scaled = log_J * self.ki[-1]
        log_J_expanded = log_J_scaled.unsqueeze(-1)
        kan_inputs.append(log_J_expanded)

        # Concatenate all inputs along the last dimension
        KAN_input = torch.cat(kan_inputs, dim=-1)  # Shape: (batches, steps, input_size)
        return KAN_input


# Initialize the model
model = KANStressPredictor()

# Prepare the input and output for training
X_train = model.prepare_input(train_strain_database)  # Input tensor (stretches and log(J))
Y_train = train_work_database  # Output tensor (work database)

# Flatten the tensors for compatibility with pykan.fit
X_train_flat = X_train.view(-1, model.input_size)  # Shape: (batch_size * steps, input_size)
Y_train_flat = Y_train.view(-1, 1)  # Shape: (batch_size * steps, 1)

# Train the KAN model using pykan's fit method
model.KAN_W.fit(X_train_flat, Y_train_flat, lr=learning_rate, steps=n_epochs)

# Save the model weights
torch.save(model.state_dict(), "KAN_model_weights.pth")

# Generate predictions for the full dataset
X_test = model.prepare_input(ref_strain_database)
X_test_flat = X_test.view(-1, model.input_size)
prediction_KAN = model.KAN_W(X_test_flat).view(ref_strain_database.shape[0], ref_strain_database.shape[1], -1)

# Create the folder to save the plots if it doesn't exist
output_folder = "KAN_predictions"
os.makedirs(output_folder, exist_ok=True)

# Plot and save results for all testing batches
for elem in test_indices:
    plt.figure(figsize=(8, 6))  # Create a new figure for each batch

    for compo in [0, 1, 2]:
        strain_for_print = ref_strain_database[elem, :, compo]
        predicted_stress_ANN = prediction_KAN[elem, :, compo].detach().numpy()
        plt.plot(
            strain_for_print,
            predicted_stress_ANN,
            label=f"KAN_comp{compo}",
            color=GetColor(compo),
            linestyle="-",
        )
        plt.scatter(
            strain_for_print,
            ref_stress_database[elem, :, compo],
            label=f"DATA_comp{compo}",
            marker="o",
            color=GetColor(compo),
            s=20,
        )

    # Add plot details
    plt.title(f"Batch: {elem}" + f" KAN")
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [MPa]")
    plt.legend()
    plt.grid(True)

    # Save the plot in the output folder
    output_path = os.path.join(output_folder, f"batch_{elem}.png")
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory
print(f"Plots saved in the folder: {output_folder}")