import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

class GaussianFixedOriginNetwork(nn.Module):
    def __init__(self, init_params=None):
        super().__init__()
        self.num_funcs = 18

        # Linear parameters
        if init_params is None:
            init_params = {
                "a": torch.tensor(1.0),
                "b": torch.tensor(0.0),
                "c": torch.tensor(1.0),
                "d": torch.tensor(0.0),
            }
        self.a = nn.Parameter(init_params["a"].float())
        self.b = nn.Parameter(init_params["b"].float())
        self.c = nn.Parameter(init_params["c"].float())
        self.d = nn.Parameter(init_params["d"].float())

        # Fixed Gaussian center
        self.mu = torch.tensor([0.5, 0.5])

        # Trainable sigma
        self.sigma = nn.Parameter(torch.tensor(0.5))

        # Trainable positions for each function
        self.positions = nn.Parameter(torch.rand(self.num_funcs, 2))  # random [0,1]

        self.symbolic_function_labels = {
            0: "0",
            1: "x",
            2: "x^2",
            3: "x^3",
            4: "x^4",
            5: "exp(x)",
            6: "log(|x|+1)",
            7: "tan(x)",
            8: "sin(x)",
            9: "sqrt(|x|)",
            10: "cos(x)",
            11: "sinh(x)",
            12: "cosh(x)",
            13: "tanh(x)",
            14: "1/(x+1)",
            15: "1/(x^2+1)",
            16: "|x|",
            17: "ReLU(x)"
        }


    def EvaluateNormalDistribution2D(self):
        diff = self.positions - self.mu  # [num_funcs, 2]
        dist_sq = (diff**2).sum(dim=1)
        weights = torch.exp(-0.5 * dist_sq / (self.sigma**2))
        return weights

    def EvalFunction(self, x):
        funcs = [
            lambda x: torch.zeros_like(x), # 0
            lambda x: x, # 1
            lambda x: x**2, # 2
            lambda x: x**3, # 3
            lambda x: x**4, #4
            lambda x: torch.exp(x), #5
            lambda x: torch.log(torch.abs(x)+1), #6 avoid log(0)
            lambda x: torch.tan(x), #7
            lambda x: torch.sin(x), #8
            lambda x: torch.sqrt(torch.abs(x)+1e-12), #9 avoid sqrt(0)
            lambda x: torch.cos(x), #10
            lambda x: torch.sinh(x),    #11
            lambda x: torch.cosh(x),  #12
            lambda x: torch.tanh(x), #13
            lambda x: 1.0 / (x + 1.0), #14 avoid div by zero
            lambda x: 1.0 / (x**2 + 1.0), #15 avoid div by zero
            lambda x: torch.abs(x), #16
            lambda x: torch.relu(x) #17
            # 18 in total
        ]
        return torch.stack([funcs[i](x) for i in range(self.num_funcs)], dim=0)

    def forward(self, x):
        input_x = self.a * x + self.b
        weights = self.EvaluateNormalDistribution2D() # 18x1
        f_x = self.EvalFunction(input_x) # 18xN
        F = f_x.T @ weights  # weighted sum
        return self.c * F + self.d


if __name__ == "__main__":
    # Generate synthetic data
    torch.manual_seed(42)
    X_ref = torch.linspace(0.0, 1.0, 100)
    Y_ref = torch.sin(X_ref-8.0) # torch.sin(X_ref) torch.log(X_ref+5) torch.exp(X_ref)
    Y_ref = Y_ref / Y_ref.max()

    model = GaussianFixedOriginNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    max_iter = 20000
    for it in range(max_iter + 1):
        optimizer.zero_grad()
        Y_pred = model(X_ref)
        loss = 0.5 * torch.mean((Y_pred - Y_ref) ** 2)
        loss += 1.0e-2 * torch.abs(model.sigma)**2  # sigma penalty
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if it % 500 == 0:
            print(f"Iter {it}, Loss {loss.item():.6e}")

    # Print parameters
    print(f"\nTrained parameters:")
    print(f"a={model.a.item():.4f}, b={model.b.item():.4f}, c={model.c.item():.4f}, d={model.d.item():.4f}")
    print(f"sigma={model.sigma.item():.4f}")

    # Plot fitted function
    plt.figure(figsize=(8,4))
    plt.scatter(X_ref.numpy(), Y_ref.numpy(), label="Target", color="blue")
    with torch.no_grad():
        plt.plot(X_ref.numpy(), model(X_ref).numpy(), "r-", label="Fitted")
    plt.legend()
    plt.show()

    # 3D plot: Gaussian interpolator (light grey) + points with labels
    weights = model.EvaluateNormalDistribution2D().detach()
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a smooth Gaussian surface over [0,1]^2
    minx = model.positions[:,0].min().item()
    maxx = model.positions[:,0].max().item()
    miny = model.positions[:,1].min().item()
    maxy = model.positions[:,1].max().item()
    grid_x = torch.linspace(minx, maxx, 2000)
    grid_y = torch.linspace(miny, maxy, 2000)
    X_grid, Y_grid = torch.meshgrid(grid_x, grid_y, indexing="ij")
    Z_grid = torch.exp(-0.5 * (((X_grid-0.5)**2 + (Y_grid-0.5)**2)) / (model.sigma**2).item())
    ax.plot_surface(X_grid.numpy(), Y_grid.numpy(), Z_grid.numpy(), color='lightgrey', alpha=0.4)

    # Plot symbolic function positions
    for i in range(model.num_funcs):
        x, y, z = model.positions[i,0].item(), model.positions[i,1].item(), weights[i].item() # weights[i].item() torch.zeros_like(model.positions[i,0]
        ax.scatter(x, y, z, color='blue', alpha=0.4)
        ax.text(x, y, z, model.symbolic_function_labels[i], color='black', fontsize=5)

    # Gaussian center
    ax.scatter(0.5,0.5,0, color='red', label='Gaussian center (0.5,0.5)', alpha=0.4)

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Weight')
    ax.set_title('Gaussian at origin with symbolic function positions')
    ax.legend()
    plt.show()
