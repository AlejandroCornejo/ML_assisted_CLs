import os
import cl_loader
import torch
import torch.nn as nn
import torch.optim as optim


directory = ".//neo_hookean_hyperelastic_law//raw_data//"
dataloader = cl_loader.get_dataloader(directory,batch_size=15)

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
        self.C = nn.Parameter(torch.abs(torch.randn(3,3)*scale_estimate))  # Random initialization

    def forward(self, x):
        return 0.5*torch.matmul(x,self.C+self.C.T)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.symmetric_layer = SymmetricLayer()

    def forward(self, x):
        return self.symmetric_layer(x)

model = SimpleNet()

optimizer = optim.Adam(model.parameters(), lr=1e3) #lr=1e-3)
def train(model, dataloader, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for strain_history, stress_history, target_work in dataloader:
            # Concatenate strain_history and stress_history if needed
            #inputs = torch.cat((strain_history, stress_history), dim=1)
            inputs = strain_history

            strain_rate = strain_history.detach().clone()
            strain_rate[1:,:] = strain_rate[1:,:] - strain_rate[0:-1,:]

            # Forward pass
            predicted_stress_history = model(inputs)

            err_stress = stress_history - predicted_stress_history
            err_work = torch.sum(err_stress*strain_rate,axis=2)
            loss = torch.norm(err_work)**2
            #loss = torch.norm(stress_history - predicted_stress_history)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
        print(model.symmetric_layer.C)

# Example usage
train(model, dataloader, optimizer, epochs=200)
