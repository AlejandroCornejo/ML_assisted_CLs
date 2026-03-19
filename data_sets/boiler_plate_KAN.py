import numpy as np
import torch
import matplotlib.pyplot as plt

import pykan.kan as KAN

# -----------------------------
# Data
# -----------------------------
x = np.linspace(0, 1, 500)
y = np.exp(-0.5 * ((x-0.25) / 0.1 )**2)

x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # (100,1)
y_torch = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (100,1)

# -----------------------------
# Model
# -----------------------------
model = KAN.MultKAN( # x--[]-->y
    width=[1, 1],
    grid=6,
    k=3,
    grid_range=[0, 1]
)
model.speed()


# -----------------------------
# Optimizer
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Training loop (L2 minimization)
# -----------------------------
n_epochs = 4000

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # y_pred_list = []
    # for i in range(len(x)):
    #     x_i = x_torch[i].view(1, 1)  # already tensor, no need to recreate
    #     y_pred_i = model(x_i)        # keep tensor with grad
    #     y_pred_list.append(y_pred_i)

    y_pred = model(x_torch)  # shape (100,1)

    # Stack into a tensor
    # y_pred = torch.vstack(y_pred_list)  # shape (100,1)

    # L2 loss
    loss = torch.mean((y_pred - y_torch)**2)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# -----------------------------
# Evaluation
# -----------------------------
y_KAN = model(x_torch).detach().numpy().flatten()

# -----------------------------
# Plot
# -----------------------------
plt.plot(x, y, label='REF')
plt.plot(x, y_KAN, '--', label='KAN (trained)')
plt.legend()
plt.show()