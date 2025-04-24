# std libs imports
import numpy as np
import scipy as sp
import torch as torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# custom imports
import cl_loader as cl_loader

torch.set_num_threads(20)

n_epochs = 10000
learning_rate = 0.1
number_of_steps = 25
ADD_NOISE = False
database = cl_loader.CustomDataset("neo_hookean_hyperelastic_law/raw_data", number_of_steps, None, ADD_NOISE)

ref_strain_database = torch.stack([item[0] for item in database]) # batch x steps x strain_size
ref_stress_database = torch.stack([item[1] for item in database]) # batch x steps x strain_size
ref_work_database   = torch.stack([item[2] for item in database]) # batch x steps x 1

ref_stress_database /= 1.0e6 # to MPa

print("Launching the training of a ANN...")
print("Number of batches: ", ref_strain_database.shape[0])
print("Number of steps  : ", ref_strain_database.shape[1])
print("Strain size      : ", ref_strain_database.shape[2])

# ==========================================================================================
# Define the neural network
class StressPredictor(nn.Module):
    """

    """
    def __init__(self):
        super(StressPredictor, self).__init__()

        self.N = 2 # number of terms in the Ogden series
        self.tol = 1.0e-7

        self.K = nn.Parameter(torch.tensor(1.0))

        self.mu_p = nn.ParameterList([
            nn.Parameter(((-1.0) ** (p + 2)) * torch.rand(1)) for p in range(self.N)
        ])

        self.alpha_p = nn.ParameterList([
            nn.Parameter(((-1.0) ** (p + 2)) * torch.rand(1)) for p in range(self.N)
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

        W = 0.5 * 1.0 * (J - 1.0)**2.0
        for p in range(self.N):
            W += (self.mu_p[p] / (self.alpha_p[p] + self.tol)) * (reg_eigenvalues[:,:,0]**self.alpha_p[p] + reg_eigenvalues[:,:,1]**self.alpha_p[p] + (1.0 /(reg_eigenvalues[:,:,0] * reg_eigenvalues[:,:,1]))**self.alpha_p[p] - 3.0)

        grad = torch.autograd.grad(
            outputs = W,
            inputs = strain,
            grad_outputs = torch.ones_like(W),
            create_graph=True)[0]
        return grad

# ==========================================================================================

# Initialize model, optimizer, and loss function
model = StressPredictor()

optimizer = optim.Adam(model.parameters(),
                       lr = learning_rate,
                    #    weight_decay=1e-8
                       )

# Training loop
for epoch in range(n_epochs):
    optimizer.zero_grad()

    predicted_stress = model(ref_strain_database)

    strain_rate = ref_strain_database[:, 1 :, :] - ref_strain_database[:, : -1, :]
    predicted_work = torch.sum(strain_rate * predicted_stress[:, 1 :, :], axis=2) # batch x steps-1
    predicted_work_accum = torch.cumsum(predicted_work, dim=1) # sumation along rows, horizontally
    error = predicted_work_accum[:, :] - ref_work_database[:, 1 :, 0]

    # loss = 0.5*torch.mean(error ** 2)  # Squared difference of work
    loss = torch.mean((predicted_stress - ref_stress_database) ** 2)

    loss.backward()
    optimizer.step()

    if epoch == n_epochs - 1:
        print("Final loss: ", loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        for name, param in model.named_parameters():
            print("\t", name, param.data)

# ===============================================================
# Let's print the results of the ANN for one batch

print("\nTraining finished.")
print("model parameters:")
for name, param in model.named_parameters():
    print(name, param.data)

null_prediction_ANN = model(torch.tensor([[[0.0, 0.0, 0.0]]]))
print("Null strain ANN prediction: ", 1.0e6*null_prediction_ANN)

torch.save(model.state_dict(), "model_weights.pth")


batch = [0, 1, 5]

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

