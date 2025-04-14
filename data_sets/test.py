import os
import cl_loader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import linear_network
import nn_network
import matplotlib.pyplot as plt
import work_based_loss
#from work_based_loss import train
import c_based_loss
from torch.utils.data import Dataset, DataLoader, TensorDataset

##############here we define transformation to apply to input strains and stresses
# Clinear = np.array([[ 7.16042643e+06,  4.27997603e+02, -2.18278728e-11],
#                     [ 4.27997603e+02,  7.14548324e+06,  7.09405867e-11],
#                     [-2.18278728e-11,  7.09405867e-11,  3.57285779e+06]])
Clinear = np.eye(3,3)
c_transform = cl_loader.ApplyCTransform(Clinear)

############## loading data and applying transform

#


_ = torch.manual_seed (2023)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

model = nn_network.StrainEnergyPotential(identity_init=True)
# model.to(device)

################ DEBUGGING
#E_voigt = torch.rand(1, 1, 3, requires_grad=True)*1e-3  # Example input (batch_size=1)

##stress = model(E_voigt)
# def psi_fn(E_voigt):
#     psi = model.EvaluatePsi(E_voigt)
#     return psi.squeeze()
# with torch.set_grad_enabled(True):
#     H = torch.autograd.functional.hessian(psi_fn, E_voigt)
#     print("C calculated = ",H)
#     err

print("Model parameters:")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

##############################


##first of all train with random small strains, so to fit the linear C
E_voigt = torch.zeros(1000, 1, 3, requires_grad=True)*1e-5  #small random strain!
random_dataset = TensorDataset(E_voigt)
dataloader = DataLoader(random_dataset, batch_size=32, shuffle=True, num_workers=0)
optimizer = optim.Adam(model.parameters(), lr=1. ) #lr=3e-4) #lr=1e-3)
c_based_loss.train(model, dataloader, optimizer, epochs=5)

#now load the real data
directory = ".//neo_hookean_hyperelastic_law//raw_data//"
steps_to_consider = -1 #3
dataset = cl_loader.CustomDataset(directory,steps_to_consider,transform=c_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
optimizer = optim.Adam(model.parameters(), lr=1.e-4 ) #lr=3e-4) #lr=1e-3)
work_based_loss.train(model, dataloader, optimizer, epochs=10)


# ===============================================================
# Let's print the results of the ANN for one batch
validation_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

def GetColor(component):
    if component == 0:
        return "r"
    elif component==1:
        return "b"
    else:
        return "k"

visualization_strain = []
visualization_stress = []

counter = 0
for strain_history, stress_history, target_work in validation_dataloader:
    strain_history = strain_history.requires_grad_(True) #cannot remove this
    predicted_stress_history = model(strain_history)
    index_in_batch = 0
    strain_for_print = strain_history[index_in_batch].detach().numpy()
    visualization_stress = predicted_stress_history[index_in_batch].detach().numpy()
    visualization_stress_ref = stress_history[index_in_batch].detach().numpy()

    for compo in [0, 1, 2]:
        strain = strain_for_print[:, compo].ravel()

        plt.plot(strain_for_print[:, compo].ravel(), visualization_stress_ref[:, compo].ravel(), label='REF_' + str(compo), color=GetColor(compo), marker='o')
        predicted_stress_ANN = visualization_stress[:, compo].ravel()
        plt.plot(strain_for_print[:, compo].ravel(), visualization_stress[:, compo].ravel(), label='ANN_' + str(compo), color=GetColor(compo), marker='x')
        #plt.title("batch: " + str(elem))
        plt.legend()
    plt.savefig(f"data_{counter}_stress_strain_component_{compo}.png", dpi=300, bbox_inches="tight")
    plt.close()
    counter += 1
    if(counter == 10):
        break

err
