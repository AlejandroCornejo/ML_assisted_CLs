# std libs imports
import numpy as np
import scipy as sp
import torch as torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os
import kan as KAN


# custom imports
import cl_loader as cl_loader

from sklearn.model_selection import train_test_split

# torch.set_num_threads(20)

"""

"""

#=============================================================================================================
"""
INPUT DATASET:
"""
n_epochs = 80
learning_rate = 0.02
number_of_steps = 25
ADD_NOISE = True
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

        self.order_stretches = 2  # Number of orders (can be set to any value)

        self.input_size = 3 * self.order_stretches  # Total inputs: 2 * reg_eigenvalues + 1 * (J-1) for each order
        self.k = 3  # Degree of splines
        self.grid = 3  # Number of knots

        # KAN framework layers
        self.KAN_W = KAN.MultKAN(
            width=[self.input_size,   self.order_stretches,   1],
            grid=self.grid,
            k=self.k
        )

    def forward(self, strain):
        batches = strain.shape[0]
        steps = strain.shape[1]

        strain = strain.detach().requires_grad_(True)

        # Compute strain tensor components
        E = torch.zeros((batches, steps, 2, 2))
        E[:, :, 0, 0] = strain[:, :, 0]  # Exx
        E[:, :, 1, 1] = strain[:, :, 1]  # Eyy
        E[:, :, 0, 1] = 0.5 * strain[:, :, 2]  # Exy
        E[:, :, 1, 0] = 0.5 * strain[:, :, 2]  # Eyx

        # Left Cauchy strain tensor
        C = torch.zeros_like(E)
        C = 2.0 * E + torch.eye(2)

        J = torch.linalg.det(C) ** 0.5

        square_eigenvalues = torch.linalg.eigvalsh(C)  # Eigenvalues: batch x steps x 2
        eigenvalues = torch.sqrt(square_eigenvalues)

        reg_eigenvalues = torch.zeros_like(eigenvalues)  # Regularized eigenvalues
        aux = J ** (-1 / 3)
        reg_eigenvalues[:, :, 0] = eigenvalues[:, :, 0] * aux
        reg_eigenvalues[:, :, 1] = eigenvalues[:, :, 1] * aux

        # Prepare inputs for KAN
        kan_inputs = []  # List to store inputs for each order

        for order in range(1, self.order_stretches + 1):
            # Compute reg_eigenvalues**order and (J-1)**order
            reg_eigenvalues_order = reg_eigenvalues ** order
            J_minus_1_order = (J - 1) ** order

            # Concatenate reg_eigenvalues_order and J_minus_1_order
            J_minus_1_order = J_minus_1_order.unsqueeze(-1)  # Add an extra dimension
            kan_input_order = torch.cat((reg_eigenvalues_order, J_minus_1_order), dim=-1)  # Shape: (batches, steps, 3)
            """
            example
                Original reg_eigenvalues_order:
                tensor([[[1., 2.],
                        [3., 4.]]])
                Shape: torch.Size([1, 2, 2])

                Original J_minus_1_order:
                tensor([[5., 6.]])
                Shape: torch.Size([1, 2])

                J_minus_1_order after unsqueeze:
                tensor([[[5.],
                        [6.]]])
                Shape: torch.Size([1, 2, 1])

                Concatenated kan_input_order:
                tensor([[[1., 2., 5.],
                        [3., 4., 6.]]])
                Shape: torch.Size([1, 2, 3])
            """
            kan_inputs.append(kan_input_order)

        # Concatenate all orders along the last dimension
        KAN_input = torch.cat(kan_inputs, dim=-1)  # Shape: (batches, steps, 3 * self.order_stretches)

        # Flatten the input for KAN (KAN cannot read 3D tensors)
        flat_KAN_input = KAN_input.view(-1, self.input_size)  # Shape: (batch x steps, 3 * self.order_stretches)

        # Pass the input through the KAN layer
        W_flat = self.KAN_W(flat_KAN_input)  # Shape: (batch x steps, 1)

        # Reshape the output back to the original shape
        W = W_flat.view(batches, steps, -1)  # Shape: (batches, steps, 1)

        # Compute gradients
        grad = torch.autograd.grad(
            outputs=W,
            inputs=strain,
            grad_outputs=torch.ones_like(W),
            create_graph=True
        )[0]

        return grad

#==========================================================================================

# Initialize model, optimizer, and loss function
model = KANStressPredictor()

print("\nNull strain KAN prediction initial CHECK: ", model(torch.tensor([[[0.0, 0.0, 0.0]]]))) # for the order 1

# optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20, history_size=10)
optimizer = optim.LBFGS(
    model.parameters(),
    lr=learning_rate,
    max_iter=20,
    history_size=30
)

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


    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        # for name, param in model.named_parameters():
        #     print("\t", name, param.data)

    if epoch == n_epochs - 1:
        print("\nFinal loss: ", loss.item())

# ===============================================================
# Let's print the results of the ANN for training and testing datasets

print("\nTraining finished.")
# print("\nmodel parameters:")
# for name, param in model.named_parameters():
#     print(name, param.data)

null_prediction_ANN = model(torch.tensor([[[0.0, 0.0, 0.0]]]))
print("\nNull strain post training KAN prediction: ", 1.0e6*null_prediction_ANN)

torch.save(model.state_dict(), "KAN_model_weights.pth")


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
output_folder = "KAN_predictions"
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

model.KAN_W.plot(folder="./KAN_predictions")