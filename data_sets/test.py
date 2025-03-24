import os
import cl_loader
import torch


directory = ".//neo_hookean_hyperelastic_law//raw_data//"
dataloader = cl_loader.get_dataloader(directory,batch_size=15)

for strain_history, stress_history,work in dataloader:
    # Your training or processing code here
    print(strain_history.shape, stress_history.shape,work.shape)