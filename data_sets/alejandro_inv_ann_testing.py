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
database = cl_loader.CustomDataset("neo_hookean_hyperelastic_law/raw_data", number_of_steps)

ref_strain_database = torch.stack([item[0] for item in database]) # batch x steps x strain_size
ref_stress_database = torch.stack([item[1] for item in database]) # batch x steps x strain_size
ref_work_database   = torch.stack([item[2] for item in database]) # batch x steps x 1

print("Launching the training of a ANN...")
print("Number of batches: ", ref_strain_database.shape[0])
print("Number of steps  : ", ref_strain_database.shape[1])
print("Strain size      : ", ref_strain_database.shape[2])

# Why not normalizing?
"""
scaler = MinMaxScaler()
normalized_db = scaler.fit_transform(database_np)
--> only works with 2Dim tensors....
original_db = scaler.inverse_transform(normalized_db.numpy())
"""

# ==========================================================================================
# Define the neural network
class StressPredictor(nn.Module):
    """
    This ANN inputs the strain, transforms to I1 invariant and returns the Stress
    """
    def __init__(self):
        super(StressPredictor, self).__init__()

        # hidden_neurons = 1
        # input_size = 1
        # output_size = 1

        self.C1 = nn.Parameter(torch.tensor(1.0))

        # self.hidden = nn.Linear(input_size,     hidden_neurons,  bias=False)  # Input: input_size      -> Hidden: hidden_neurons
        # self.output = nn.Linear(hidden_neurons, output_size,     bias=False)  # Hidden: hidden_neurons -> Output: output_size

    def forward(self, strain):
        batches     = strain.shape[0]
        steps       = strain.shape[1]
        strain_size = strain.shape[2]

        strain = strain.detach().requires_grad_(True)

        # left cauchy strain
        C = torch.zeros((batches, steps, 3))
        C[:, :, :] = 2.0 * strain[:, :, :]
        C[:, :, 0] += 1.0
        C[:, :, 1] += 1.0

        I1 = torch.zeros((batches, steps, 1))
        I1[:, :, 0] = (C[:, :, 0] + C[:, :, 1]) - 3.0 # (I1-3)

        # W = nn.functional.softplus(self.C1) * I1
        W = (self.C1) * I1

        grad = torch.autograd.grad(
            outputs = W,
            inputs = strain,
            grad_outputs = torch.ones_like(W),
            create_graph=True)[0]
        
        # print(self.C1)
        # print(grad)

        return -grad

# ==========================================================================================





# Initialize model, optimizer, and loss function
model = StressPredictor()
optimizer = optim.Adam(model.parameters(), lr = 1.0e4)

# Training loop
n_epochs = 10000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    predicted_stress = model(ref_strain_database)

    predicted_work = torch.sum((ref_strain_database[:, 1 :, :] - ref_strain_database[:, : -1, :]) * predicted_stress[:, 1 :, :], axis=2) # batch x steps-2
    predicted_work = torch.cumsum(predicted_work, dim=1) # sumation along rows, horizontally

    loss = torch.mean((predicted_work - ref_work_database[:, 1:, 0]) ** 2)  # Squared difference of work
    # loss = torch.mean((predicted_work - ref_work_database[:, 1:, 0]) ** 2) + 1e2 * torch.mean((predicted_stress[:, :, :] - ref_stress_database[:, :, :]) ** 2)
    # loss = torch.mean((predicted_stress - ref_stress_database) ** 2)  # Squared difference of work
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")



# test_strain = torch.tensor([[[0.001, 0.001, 0.0]]])
# print("model = ", model(test_strain))

# ===============================================================
# Let's print the results of the ANN for one batch
batch = [1]

def GetColor(component):
    if component == 0:
        return "r"
    elif component==1:
        return "b"
    else:
        return "k"

prediction_ANN = model(ref_strain_database)
# print(prediction_ANN[0,10,:])
# print(ref_stress_database[0,10,:])
# a

for elem in batch:
    for compo in [0, 1, 2]:
        strain_for_print = ref_strain_database[elem, :, compo]
        plt.plot(strain_for_print, ref_stress_database[elem, :, compo], label='REF_' + str(compo), color=GetColor(compo), marker='o')
        predicted_stress_ANN = prediction_ANN[elem, :, compo].detach().numpy()
        plt.plot(strain_for_print, predicted_stress_ANN, label='ANN_' + str(compo), color=GetColor(compo), marker='x')
        plt.title("batch: " + str(elem))
        plt.legend()
    plt.show()

