import os
import cl_loader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import linear_network
import nn_network
import matplotlib.pyplot as plt

##############here we define transformation to apply to input strains and stresses
Clinear = np.array([[ 7.16042643e+06,  4.27997603e+02, -2.18278728e-11],
                    [ 4.27997603e+02,  7.14548324e+06,  7.09405867e-11],
                    [-2.18278728e-11,  7.09405867e-11,  3.57285779e+06]])
c_transform = cl_loader.ApplyCTransform(Clinear)

############## loading data and applying transform
directory = ".//neo_hookean_hyperelastic_law//raw_data//"
steps_to_consider = -1 #3
dataloader = cl_loader.get_dataloader(directory,batch_size=32,steps_to_consider=steps_to_consider,transform=c_transform)

for strain_history, stress_history,work in dataloader:
    # Your training or processing code here
    print(strain_history.shape, stress_history.shape,work.shape)

_ = torch.manual_seed (2023)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#model = linear_network.SimpleNet(identity_init=True)
model = nn_network.StrainEnergyPotential(identity_init=True)
model.to(device)

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
#optimizer = optim.LBFGS(model.parameters(), lr=1.0e4, max_iter=20)
print(model.parameters())
#optimizer = optim.Adagrad(model.parameters(), lr=1.0e-6) #lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1.0e-4) #lr=1e-3)
def train(model, dataloader, optimizer, epochs):
    model.train()

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    losses = []
    plot_epochs = []

    initial_loss = None
    for epoch in range(epochs):
        epoch_loss = 0.0
        for strain_history, stress_history, target_work in dataloader:
            strain_rate = strain_history.detach().clone()
            strain_rate[:,1:,:] = strain_history[:,1:,:] - strain_history[:,0:-1,:]

            strain_history = strain_history.requires_grad_(True)


            # Forward pass
            predicted_stress_history = model(strain_history)
            err_stress = stress_history - predicted_stress_history
            err_work_aux = torch.sum(err_stress*strain_rate,axis=2)
            #err_work = err_work_aux #TODO uncomment the one below
            err_work = torch.cumsum(err_work_aux, dim=1) #ensure that we accumulate the error over time

            #TODO: missing the scansum
            loss = torch.norm(err_work)**2
            #loss = torch.norm(stress_history - predicted_stress_history)
            if initial_loss==None:
                initial_loss=loss.item()
            # print("initial_loss",initial_loss)
            # print("loss",loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print("raw",model.convex_nn.raw_weight1,model.convex_nn.raw_weight2,model.convex_nn.raw_weight3)

            epoch_loss += loss.item()

            # Store loss
            losses.append(loss.item()/initial_loss)
            plot_epochs.append(epoch)

        # Update plot
        if epoch%20 == 0:
            ax.clear()
            ax.plot(plot_epochs, losses, label="Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.pause(0.1)  # Pause to update the plot
        # print("epoch_loss",epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()/(initial_loss)}")
        # if epoch%10==0:
        #     print(0.5*(model.symmetric_layer.C + model.symmetric_layer.C.T))

        # if epoch%20 == 0:
        #     print("raw",model.convex_nn.raw_weight1,model.convex_nn.raw_weight2,model.convex_nn.raw_weight3)

    plt.ioff()  # Turn off interactive mode
    plt.show()

# Example usage
train(model, dataloader, optimizer, epochs=2000)
