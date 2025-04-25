import os
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, directory, steps_to_consider, transform=None, add_noise=False):
        self.steps_to_consider = steps_to_consider
        self.transform = transform
        self.data_list = []  # Store preloaded data

        # Load all .npz files into memory
        print("=========== START READING ====")
        for file in os.listdir(directory):
            if file.endswith('.npz'):
                file_path = os.path.join(directory, file)
                data = np.load(file_path)

                # Extract strain and stress history
                all_strains = data["strain_history"]
                all_stresses = data["stress_history"]

                strain_history = all_strains[1:self.steps_to_consider, :]
                stress_history = all_stresses[1:self.steps_to_consider, :]

                # Add noise if necessary
                if add_noise:
                    noise_level = 0.05
                    # strain_noise = np.random.normal(loc=0.0, scale=noise_level * np.std(strain_history), size=strain_history.shape)
                    stress_noise = np.random.normal(loc=0.0, scale=noise_level * np.std(stress_history), size=stress_history.shape)
                    # strain_history += strain_noise
                    stress_history += stress_noise


                strain_rate = strain_history - all_strains[:self.steps_to_consider - 1, :]

                # Compute work and ensure correct shape
                aux = np.sum(stress_history * strain_rate, axis=1)
                work = np.cumsum(aux)[:, None]  # Reshape to (steps_to_consider - 1, 1)

                # Convert to PyTorch tensors
                strain_history = torch.tensor(strain_history, dtype=torch.float32)
                stress_history = torch.tensor(stress_history, dtype=torch.float32)
                work = torch.tensor(work, dtype=torch.float32)

                self.data_list.append((strain_history, stress_history, work))  # Store preloaded data
        print("=========== READING COMPLETED ====")

        print(f"Loaded {len(self.data_list)} files into memory.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        strain_history, stress_history, work = self.data_list[idx]

        if self.transform:
            strain_history, stress_history = self.transform(strain_history, stress_history)

        return strain_history, stress_history, work

def get_dataloader(directory, batch_size = 32, shuffle = True, num_workers = 0, steps_to_consider = -1, transform = None):

    dataset = CustomDataset(directory,steps_to_consider,transform=transform)
    print("number of files", len(dataset))
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return dataloader

class ApplyCTransform:
    def __init__(self, C):
        """
        Initializes the transformation with the given C matrix.
        Args:
            C (torch.Tensor): A 3x3 numpy matrix to apply to the stress and strain history.
        """
        self.Cinv = torch.tensor(np.linalg.inv(C), dtype=torch.float32)
        print(self.Cinv)

    def __call__(self, strain_history, stress_history):
        """
        Applies the transformation to the stress and strain history.
        Args:
            sample (dict): A dictionary containing 'stress_history' and 'strain_history'.
        Returns:
            dict: Transformed sample with the applied Csqrt multiplication.

        Comment: This maping implies that the relationship between the
        stress transformed and the strain transformed is the identity matrix.
        """
        stress_transformed =  stress_history @ self.Cinv
        strain_transformed =  strain_history
        #print(stress_transformed.shape, stress_history.shape)

        return strain_transformed,stress_transformed