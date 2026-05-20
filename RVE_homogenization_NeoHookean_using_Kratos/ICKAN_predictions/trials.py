

import numpy as np

def L2_relative_error(pred, target):
    return np.sqrt(np.sum((pred - target) ** 2) / np.sum(target**2))


W_database = np.linspace(0.0, 1050.0, 15)

W_pred = 0.5 * W_database + 5.0

stress_database = np.linspace(0.0, 1.0, W_database.shape[0])

s_pred = 0.5 * stress_database + 0.1


strain = np.linspace(0.0, 1e-3, len(W_database))

print(L2_relative_error(W_pred, W_database))
print(L2_relative_error(s_pred, stress_database))