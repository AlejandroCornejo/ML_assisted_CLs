import os
import cl_loader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


directory = ".//neo_hookean_hyperelastic_law//raw_data//"
steps_to_consider = 3
dataloader = cl_loader.get_dataloader(directory,batch_size=32,steps_to_consider=steps_to_consider)

for strain_history, stress_history,work in dataloader:
    # Your training or processing code here
    print(strain_history.shape, stress_history.shape,work.shape)

_ = torch.manual_seed (2023)

# class SymmetricLayer(nn.Module):
#     def __init__(self, scale_estimate=1e6):
#         super(SymmetricLayer, self).__init__()
#         # Create a 3x3 matrix with 6 independent parameters (since it's symmetric)
#         self.C_params = nn.Parameter(torch.abs(torch.randn(6)*scale_estimate))  # Random initialization

#     def ConstructSymm(self, my_device):
#         C = torch.zeros(3, 3, device=my_device)

#         C[0, 0] = self.C_params[0]
#         C[1, 1] = self.C_params[1]
#         C[2, 2] = self.C_params[2]

#         C[0, 1] = C[1, 0] = self.C_params[3]
#         C[0, 2] = C[2, 0] = self.C_params[4]
#         C[1, 2] = C[2, 1] = self.C_params[5]

#         return C

#     def forward(self, x):
#         C = self.ConstructSymm(x.device)

#         # Matrix multiplication C @ input
#         return torch.matmul(x,C)

class SymmetricLayer(nn.Module):
    def __init__(self, scale_estimate=1e6):
        super(SymmetricLayer, self).__init__()
        # self.C = nn.Parameter(torch.abs(torch.randn(3,3)*scale_estimate))  # Random initialization

        self.C = nn.Parameter(torch.tensor([[ 7.16042643e+06,  4.27997603e+02, -2.18278728e-11],
                                    [ 4.27997603e+02,  7.14548324e+06,  7.09405867e-11],
                                    [-2.18278728e-11,  7.09405867e-11,  3.57285779e+06]], dtype=torch.float32))

    def forward(self, x):
        return 0.5*torch.matmul(x,self.C+self.C.T)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.symmetric_layer = SymmetricLayer()

    def forward(self, x):
        return self.symmetric_layer(x)

model = SimpleNet()

optimizer = optim.Adam(model.parameters(), lr=1.0e-3) #lr=1e-3)
def train(model, dataloader, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for strain_history, stress_history, target_work in dataloader:
            #TODO: this should be a tensor of order 3 not 2 totake into account batches
            strain_rate = strain_history.detach().clone()
            strain_rate[:,1:,:] = strain_history[:,1:,:] - strain_history[:,0:-1,:]

            # Forward pass
            predicted_stress_history = model(strain_history)
            # print("strain",strain_history)
            # print("stress",predicted_stress_history)
            # print("target stress",stress_history)
            # err

            err_stress = stress_history - predicted_stress_history
            err_work_aux = torch.sum(err_stress*strain_rate,axis=2)
            #err_work[:,:] = torch.tensor(np.cumsum(err_work_aux[:,:]), dtype=torch.float32)
            err_work = err_work_aux
            #TODO: missing the scansum
            loss = torch.norm(err_work)**2
            #loss = torch.norm(stress_history - predicted_stress_history)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
        if epoch%10:
            print(0.5*(model.symmetric_layer.C + model.symmetric_layer.C.T))

# Example usage
train(model, dataloader, optimizer, epochs=200)
