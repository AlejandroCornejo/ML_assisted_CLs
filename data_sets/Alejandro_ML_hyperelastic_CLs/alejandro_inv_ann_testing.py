# std libs imports
import numpy as np
import scipy as sp
import torch as torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import sklearn as sk

# custom imports
import cl_loader as cl_loader

number_of_steps = 25
ADD_NOISE = False
database = cl_loader.CustomDataset("neo_hookean_hyperelastic_law/raw_data", number_of_steps, None, ADD_NOISE)

ref_strain_database = torch.stack([item[0] for item in database]) # batch x steps x strain_size
ref_stress_database = torch.stack([item[1] for item in database]) # batch x steps x strain_size
ref_work_database   = torch.stack([item[2] for item in database]) # batch x steps x 1

print("Launching the training of a ANN...")
print("Number of batches: ", ref_strain_database.shape[0])
print("Number of steps  : ", ref_strain_database.shape[1])
print("Strain size      : ", ref_strain_database.shape[2])

# ==========================================================================================
# Define the neural network
class StressPredictor(nn.Module):
    """
    This ANN inputs the strain, the computes the I1 from the left cauchy strain tensor, and J=detF
    and then computes the stress using the neo-hookean hyperelastic law.
        W = C1*(I1 - 2) - C1*log(J) + 0.5*C2*(J-1)^2
    """
    def __init__(self):
        super(StressPredictor, self).__init__()

        self.C1 = nn.Parameter(torch.tensor(1.0e6))
        self.C2 = nn.Parameter(torch.tensor(1.0e6))

    def forward(self, strain):
        batches     = strain.shape[0]
        steps       = strain.shape[1]

        strain = strain.detach().requires_grad_(True)

        # left cauchy strain tensor
        C = torch.zeros((batches, steps, 2, 2))
        C[:, :, 0, 0] = 2.0 * strain[:, :, 0] + 1.0 # Exx
        C[:, :, 1, 1] = 2.0 * strain[:, :, 1] + 1.0 # Eyy
        C[:, :, 0, 1] = strain[:, :, 2] # Exy
        C[:, :, 1, 0] = strain[:, :, 2] # Eyx

        I1 = C[:, :, 0, 0] + C[:, :, 1, 1]

        J = torch.linalg.det(C)

        W = nn.functional.relu(self.C1) * (I1 - 2.0) - nn.functional.relu(self.C1) * torch.log(J) + \
            0.5 * nn.functional.relu(self.C2) * (J - 1.0)**2.0

        grad = torch.autograd.grad(
            outputs = W,
            inputs = strain,
            grad_outputs = torch.ones_like(W),
            create_graph=True)[0]

        return grad

# ==========================================================================================

# Initialize model, optimizer, and loss function
model = StressPredictor()
optimizer = optim.Adam(model.parameters(), lr = 10000)

# Training loop
n_epochs = 2000
for epoch in range(n_epochs):
    optimizer.zero_grad()

    predicted_stress = model(ref_strain_database)

    strain_rate = ref_strain_database[:, 1 :, :] - ref_strain_database[:, : -1, :]
    predicted_work = torch.sum(strain_rate * predicted_stress[:, 1 :, :], axis=2) # batch x steps-1
    predicted_work_accum = torch.cumsum(predicted_work, dim=1) # sumation along rows, horizontally
    error = predicted_work_accum[:, :] - ref_work_database[:, 1 :, 0]

    loss = 0.5*torch.mean(error ** 2)  # Squared difference of work
    # loss = torch.mean((predicted_stress - ref_stress_database) ** 2)  # Squared difference of work
    loss.backward()
    optimizer.step()

    if epoch == n_epochs - 1:
        print("Final loss: ", loss.item())
        # print("error; ", error[0, :])
        # print("predicted_work_accum; ", predicted_work_accum[0, :])
        # print("ref_work_database; ", ref_work_database[0, 1:, 0])
        # print("strain_rate; ", strain_rate[0, :, :])
        # print("predicted_stress; ", model(ref_strain_database)[0, :, :])
        # print("ref_strain_database; ", ref_strain_database[0, :, :])
        # print("ref_stress_database; ", ref_stress_database[0, :, :])

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ===============================================================
# Let's print the results of the ANN for one batch

print("\nTraining finished.")
print("model parameters:")
for name, param in model.named_parameters():
    print(name, param.data)

"""
Training finished.
model parameters:
C1 tensor(1793389.8750)
C2 tensor(613.7129)
"""

batch = [0, 1, 5, 10, 320]

def GetColor(component):
    if component == 0:
        return "r"
    elif component==1:
        return "b"
    else:
        return "k"

prediction_ANN = model(ref_strain_database)

for elem in batch:
    for compo in [0, 1, 2]:
        strain_for_print = ref_strain_database[elem, :, compo]
        predicted_stress_ANN = prediction_ANN[elem, :, compo].detach().numpy()
        plt.plot(strain_for_print, predicted_stress_ANN, label='ANN_' + str(compo), color=GetColor(compo), marker='x', markersize=8)
        plt.plot(strain_for_print, ref_stress_database[elem, :, compo], label='REF_' + str(compo), color=GetColor(compo), marker='o')
        plt.title("batch: " + str(elem))
        plt.xlabel("strain [-]")
        plt.ylabel("stress [Pa]")
        plt.legend()
    plt.show()

