import numpy as np
import os

path = "RVE_homogenization_NeoHookean_using_Kratos/stage_1_training_set_fom/trajectory_1/"
files = [f for f in os.listdir(path) if f.endswith(".npy")]

for f in sorted(files):
    data = np.load(os.path.join(path, f))
    print(f"{f}: shape {data.shape}")
