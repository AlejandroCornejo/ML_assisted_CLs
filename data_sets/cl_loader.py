import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, directory,steps_to_consider):
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.npz')]
        self.steps_to_consider = steps_to_consider

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        data = np.load(file_path)

        all_strains = data["strain_history"]
        all_stresses = data["stress_history"]

        strain_history = all_strains[1:self.steps_to_consider,:]
        stress_history = all_stresses[1:self.steps_to_consider,:]
        strain_rate = all_strains[1:self.steps_to_consider,:] - all_strains[0:self.steps_to_consider-1,:]
        #print(strain)
        aux = np.sum(stress_history*strain_rate,axis=1)

        strain_history = torch.tensor(strain_history, dtype=torch.float32)
        stress_history = torch.tensor(stress_history, dtype=torch.float32)
        work = torch.tensor(np.cumsum(aux), dtype=torch.float32)

        return strain_history, stress_history,work


def get_dataloader(directory, batch_size=32, shuffle=True, num_workers=0,steps_to_consider=-1):
    dataset = CustomDataset(directory,steps_to_consider)
    print("number of files", len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader