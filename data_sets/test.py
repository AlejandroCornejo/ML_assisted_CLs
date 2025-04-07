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
dataloader = cl_loader.get_dataloader(directory,batch_size=8,steps_to_consider=steps_to_consider,transform=c_transform)

for strain_history, stress_history,work in dataloader:
    # Your training or processing code here
    print(strain_history.shape, stress_history.shape,work.shape)

_ = torch.manual_seed (2023)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

#model = linear_network.SimpleNet(identity_init=True)
model = nn_network.StrainEnergyPotential(identity_init=True)
# model.to(device)

################ DEBUGGING
# E_voigt = torch.rand(1, 1, 3, requires_grad=True)*1e-3  # Example input (batch_size=1)
# E_voigt[0,0,0] = 1e-3
# E_voigt[0,0,1] = 2e-3
# E_voigt[0,0,2] = 3e-3
# stress = model(E_voigt)
# print("strain=",E_voigt)
# print("stress",stress)
# print("psi ", model.EvaluatePsi(E_voigt))
# err

print("Model parameters:")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

##############################
E_voigt = torch.rand(1, 1, 3, requires_grad=True)  # Dummy input

##############################
# Check gradients


#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
#optimizer = optim.LBFGS(model.parameters(), lr=1.0e4, max_iter=20)
print(model.parameters())
#optimizer = optim.Adagrad(model.parameters(), lr=1.0e-4) #lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=3e-4) #lr=1e-3)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

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
            # strain_history = strain_history.to(device)
            # stress_history = stress_history.to(device)
            #target_work = target_work.to(device)

            strain_rate = strain_history.detach().clone()
            strain_rate[:,1:,:] = strain_history[:,1:,:] - strain_history[:,0:-1,:]
            # strain_rate = strain_rate.to(device)

            strain_history = strain_history.requires_grad_(True)
            # strain_history = strain_history.to(device)


            # Forward pass
            predicted_stress_history = model(strain_history)
            err_stress = stress_history - predicted_stress_history
            err_work_aux = torch.sum(err_stress*strain_rate,axis=2)
            #err_work = err_work_aux
            err_work = torch.cumsum(err_work_aux, dim=1) #ensure that we accumulate the error over time
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
            # scheduler.step()
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

            print("W",model.convex_nn.Wdiag)
            print("a",model.convex_nn.a_b)
            print("d",model.convex_nn.d_e)

        # print("epoch_loss",epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()/(initial_loss)}")
        # if epoch%10==0:
        #     print(0.5*(model.symmetric_layer.C + model.symmetric_layer.C.T))

        # if epoch%20 == 0:
        #     print("raw",model.convex_nn.raw_weight1,model.convex_nn.raw_weight2,model.convex_nn.raw_weight3)

    plt.ioff()  # Turn off interactive mode
    plt.show()

# Example usage
train(model, dataloader, optimizer, epochs=500)

# ===============================================================
# Let's print the results of the ANN for one batch
validation_dataloader = cl_loader.get_dataloader(directory,batch_size=1,steps_to_consider=-1,transform=c_transform)
batch = [1, 2, 3, 4, 6, 16, 350]

def GetColor(component):
    if component == 0:
        return "r"
    elif component==1:
        return "b"
    else:
        return "k"

visualization_strain = []
visualization_stress = []

for strain_history, stress_history, target_work in validation_dataloader:
    strain_history = strain_history.requires_grad_(True) #cannot remove this
    predicted_stress_history = model(strain_history)
    index_in_batch = 0
    strain_for_print = strain_history[index_in_batch].detach().numpy()
    visualization_stress = predicted_stress_history[index_in_batch].detach().numpy()
    visualization_stress_ref = stress_history[index_in_batch].detach().numpy()
    print(strain_for_print.shape)
    print(visualization_stress)
    for compo in [0, 1, 2]:
        strain = strain_for_print[:, compo].ravel()
        print("strain_for_print.shape",strain_for_print.shape)
        print("visualization_stress_ref[:, compo].shape",visualization_stress_ref[:, compo].shape)
        plt.plot(strain_for_print[:, compo].ravel(), visualization_stress_ref[:, compo].ravel(), label='REF_' + str(compo), color=GetColor(compo), marker='o')
        predicted_stress_ANN = visualization_stress[:, compo].ravel()
        print("predicted_stress_ANN.shape",predicted_stress_ANN.shape)
        plt.plot(strain_for_print[:, compo].ravel(), visualization_stress[:, compo].ravel(), label='ANN_' + str(compo), color=GetColor(compo), marker='x')
        #plt.title("batch: " + str(elem))
        plt.legend()
    plt.show()
err
