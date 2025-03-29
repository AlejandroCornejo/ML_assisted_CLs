
import numpy as np
import scipy as sp
import torch as torch

import torch.nn as nn
import torch.optim as optim

import cl_loader as cl_loader

import matplotlib.pyplot as plt


number_of_steps = 25
database = cl_loader.CustomDataset("neo_hookean_hyperelastic_law/raw_data", number_of_steps)

ref_strain_database = torch.stack([item[0] for item in database]) # batch x steps x strain_size
ref_stress_database = torch.stack([item[1] for item in database]) # batch x steps x strain_size
ref_work_database   = torch.stack([item[2] for item in database]) # batch x steps x 1

# ==========================================================================================
# Define the neural network
class StressPredictor(nn.Module):
    def __init__(self):
        super(StressPredictor, self).__init__()
        hidden_neurons = 5
        self.hidden = nn.Linear(3, hidden_neurons, bias=False)  # Input: 3 -> Hidden: 5
        self.output = nn.Linear(hidden_neurons, 3, bias=False)  # Hidden: 5 -> Output: 3

    def forward(self, strain):
        x = torch.relu(self.hidden(strain))  # ReLU activation
        stress = self.output(x)              # Linear output
        return stress
# ==========================================================================================





# Initialize model, optimizer, and loss function
model = StressPredictor()
optimizer = optim.Adam(model.parameters(), lr = 10.0)

# Training loop
n_epochs = 10000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    predicted_stress = model(ref_strain_database)

    predicted_work = torch.sum((ref_strain_database[:, 1 :, :] - ref_strain_database[:, : -1, :]) * predicted_stress[:, 1 :, :], axis=2) # batch x steps-2
    predicted_work = torch.cumsum(predicted_work, dim=1) # sumation along rows, horizontally

    # loss = torch.mean((predicted_work - ref_work_database[:, 1:, 0]) ** 2)  # Squared difference of work
    loss = torch.mean((predicted_stress - ref_stress_database) ** 2)  # Squared difference of work
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")




# ===============================================================
# Let's print the results of the ANN for one batch
batch = [1, 25]
strain_component = 0
stress_component = 0


prediction_ANN = model(ref_strain_database)

for elem in batch:
    strain_for_print = ref_strain_database[elem, :, strain_component]
    plt.plot(strain_for_print, ref_stress_database[elem, :, stress_component], label='REF', color='b', marker='o')
    predicted_stress_ANN = prediction_ANN[elem, :, strain_component].detach().numpy()
    plt.plot(strain_for_print, predicted_stress_ANN, label='ANN', color='r', marker='o')
    plt.legend()
    plt.show()
