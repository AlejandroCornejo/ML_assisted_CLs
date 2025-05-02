# std libs imports
import numpy as np
import scipy as sp
import torch as torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os

# custom imports
import cl_loader as cl_loader

from sklearn.model_selection import train_test_split

# torch.set_num_threads(20)

"""
# Generalized Ogden ANN Script

This script implements a machine learning model to predict stress-strain relationships using a generalized Ogden hyperelastic law. The model is based on an Artificial Neural Network (ANN) and is trained on synthetic datasets generated from hyperelastic material behavior.

## Features

1. **Dataset Preparation**:
   - The script loads a custom dataset containing strain, stress, and work data.
   - The dataset is split into training and testing subsets using an 80-20 split.

2. **Model Architecture**:
   - The ANN is implemented as a PyTorch module (`StressPredictor`) with trainable parameters for the Ogden hyperelastic law.
   - The model predicts stress based on strain input using the Ogden strain energy function.

3. **Training**:
   - The model is trained using the LBFGS optimizer.
   - The loss function minimizes the squared difference between predicted and reference work values.

4. **Visualization**:
   - The script plots the predicted stress-strain curves for both training and testing datasets.
   - Training and testing data points are distinguished using different markers (`o` for training and `x` for testing) but share the same colors for each stress component.

5. **Model Saving**:
   - The trained model's weights are saved to a file (`model_weights.pth`) for future use.

---

## Requirements

The script requires the following Python libraries:

- `numpy`
- `scipy`
- `torch`
- `matplotlib`
- `scikit-learn`

To install the required libraries, run:

```bash
pip install numpy scipy torch matplotlib scikit-learn
"""

#=============================================================================================================
"""
INPUT DATASET:
"""
n_epochs = 6000
learning_rate = 0.08
number_of_steps = 25
ADD_NOISE = False
database = cl_loader.CustomDataset("neo_hookean_hyperelastic_law/raw_data", number_of_steps, None, ADD_NOISE)
#=============================================================================================================

ref_strain_database = torch.stack([item[0] for item in database]) # batch x steps x strain_size
ref_stress_database = torch.stack([item[1] for item in database]) # batch x steps x strain_size
ref_work_database   = torch.stack([item[2] for item in database]) # batch x steps x 1

ref_stress_database /= 1.0e6 # to MPa
ref_work_database   /= 1.0e6 # to MPa vs strain
strain_rate = ref_strain_database[:, 1 :, :] - ref_strain_database[:, : -1, :]

# Split the dataset into training and testing datasets
train_indices, test_indices = train_test_split(
    range(len(database)), test_size=0.2, random_state=42
)

train_strain_database = ref_strain_database[train_indices]
train_stress_database = ref_stress_database[train_indices]
train_work_database = ref_work_database[train_indices]

test_strain_database = ref_strain_database[test_indices]
test_stress_database = ref_stress_database[test_indices]
test_work_database = ref_work_database[test_indices]

print("\nLaunching the training of a ANN...")
print("Number of training batches: ", train_strain_database.shape[0])
print("Number of total batches: ", ref_strain_database.shape[0])
print("Number of steps  : ", train_strain_database.shape[1])
print("Strain size      : ", train_strain_database.shape[2])

# ==========================================================================================

class StressPredictor(nn.Module):
    """

    """
    def __init__(self):
        super(StressPredictor, self).__init__()

        self.N = 2 # number of terms in the Ogden series
        self.tol = 1.0e-12

        self.K = nn.Parameter(torch.tensor(1.0))

        self.mu_p = nn.ParameterList([
            nn.Parameter(((-1.0)**(p + 2)) * (torch.tensor(1.0 + torch.rand(1)))) for p in range(self.N)
        ])

        self.alpha_p = nn.ParameterList([
            nn.Parameter(((-1.0)**(p + 2)) * (torch.tensor(1.0 + torch.rand(1)))) for p in range(self.N)
        ])

    def forward(self, strain):
        batches = strain.shape[0]
        steps   = strain.shape[1]

        strain = strain.detach().requires_grad_(True)

        E = torch.zeros((batches, steps, 2, 2))
        E[:, :, 0, 0] = strain[:, :, 0] # Exx
        E[:, :, 1, 1] = strain[:, :, 1] # Eyy
        E[:, :, 0, 1] = 0.5 * strain[:, :, 2] # Exy
        E[:, :, 1, 0] = 0.5 * strain[:, :, 2] # Eyx

        # left cauchy strain tensor
        C = torch.zeros_like(E)
        C = 2.0 * E + torch.eye(2)

        J = torch.linalg.det(C)**0.5

        square_eigenvalues = torch.linalg.eigvalsh(C) # eigenvalues: batch x steps x 2
        eigenvalues = torch.sqrt(square_eigenvalues)

        reg_eigenvalues = torch.zeros_like(eigenvalues) # batch x steps x 2
        aux = J**(-1 / 3)
        reg_eigenvalues[:,:, 0] = eigenvalues[:,:,0] * aux
        reg_eigenvalues[:,:, 1] = eigenvalues[:,:,1] * aux

        W = 0.5 * self.K * (J - 1.0)**2.0
        # W = 0.5 * self.K * torch.log(J)**2
        for p in range(self.N):
            W += (self.mu_p[p] / (self.alpha_p[p] + self.tol)) * (reg_eigenvalues[:,:,0]**self.alpha_p[p] + reg_eigenvalues[:,:,1]**self.alpha_p[p] + (1.0 / (reg_eigenvalues[:,:,0]*reg_eigenvalues[:,:,1]))**self.alpha_p[p] - 3.0)

        grad = torch.autograd.grad(
            outputs = W,
            inputs = strain,
            grad_outputs = torch.ones_like(W),
            create_graph=True)[0]
        return grad

#==========================================================================================

# Initialize model, optimizer, and loss function
model = StressPredictor()


print("\nNull strain ANN prediction CHECK: ", model(torch.tensor([[[0.0, 0.0, 0.0]]])))

# optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=10, history_size=25)

# Training loop
# for epoch in range(n_epochs):
#     optimizer.zero_grad()

#     predicted_stress = model(ref_strain_database)

#     predicted_work = torch.sum(strain_rate * predicted_stress[:, 1 :, :], axis=2) # batch x steps-1
#     predicted_work_accum = torch.cumsum(predicted_work, dim=1) # sumation along rows, horizontally
#     error = predicted_work_accum[:, :] - ref_work_database[:, 1 :, 0]

#     loss = 0.5*torch.mean(error ** 2)  # Squared difference of work
#     # loss = torch.mean((predicted_stress - ref_stress_database) ** 2)

#     loss.backward()
#     optimizer.step()

#     if epoch == n_epochs - 1:
#         print("Final loss: ", loss.item())

#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
#         for name, param in model.named_parameters():
#             print("\t", name, param.data)

# Update the training loop to use only the training dataset
for epoch in range(n_epochs):

    def closure():
        optimizer.zero_grad()

        predicted_stress = model(train_strain_database)
        predicted_work = torch.sum(strain_rate[train_indices] * predicted_stress[:, 1 :, :], axis=2)
        predicted_work_accum = torch.cumsum(predicted_work, dim=1)
        error = predicted_work_accum - train_work_database[:, 1 :, 0]
        loss = 0.5 * torch.mean(error ** 2)
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    if epoch == n_epochs - 1:
        print("\nFinal loss: ", loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        for name, param in model.named_parameters():
            print("\t", name, param.data)


# ===============================================================
# Let's print the results of the ANN for training and testing datasets

print("\nTraining finished.")
print("\nmodel parameters:")
for name, param in model.named_parameters():
    print(name, param.data)

null_prediction_ANN = model(torch.tensor([[[0.0, 0.0, 0.0]]]))
print("\nNull strain ANN prediction: ", 1.0e6*null_prediction_ANN)

torch.save(model.state_dict(), "model_weights.pth")

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

# Create the folder to save the plots if it doesn't exist
output_folder = "generalized_ogden_predictions"
os.makedirs(output_folder, exist_ok=True)

# Generate predictions for the full dataset
prediction_ANN = model(ref_strain_database)

# Plot and save results for all testing batches
for elem in test_indices:
    plt.figure(figsize=(8, 6))  # Create a new figure for each batch

    for compo in [0, 1, 2]:
        strain_for_print = ref_strain_database[elem, :, compo]
        predicted_stress_ANN = prediction_ANN[elem, :, compo].detach().numpy()
        plt.plot(
            strain_for_print,
            predicted_stress_ANN,
            label=f"ANN_comp{compo}",
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
    plt.title(f"Batch: {elem}" + f" Ogden N= {model.N}")
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [MPa]")
    plt.legend()
    plt.grid(True)

    # Save the plot in the output folder
    output_path = os.path.join(output_folder, f"batch_{elem}.png")
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory

print(f"Plots saved in the folder: {output_folder}")

