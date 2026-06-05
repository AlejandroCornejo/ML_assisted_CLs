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

        # Shared global Linear parameters (single set of a,b,c,d for all symbolic functions)
        if init_params is None:
            init_params = {
                "a": torch.ones(1),
                "b": torch.zeros(1),
                "c": torch.ones(1),
                "d": torch.zeros(1),
            }
        self.a = nn.Parameter(init_params["a"].float())   # [1]
        self.b = nn.Parameter(init_params["b"].float())   # [1]
        self.c = nn.Parameter(init_params["c"].float())   # [1]
        self.d = nn.Parameter(init_params["d"].float())   # [1]

        # Fixed Gaussian center
        self.mu = torch.tensor([0.5, 0.5])

        # Trainable sigma
        # self.sigma = nn.Parameter(torch.tensor(0.7))
        self.sigma = torch.tensor(0.4)

        # Trainable positions for each function - initialized on a circle centered at mu=(0.5,0.5)
        # so all functions are equidistant from the center and equally spaced angularly
        angles = 2.0 * np.pi * torch.arange(self.num_funcs, dtype=torch.float32) / self.num_funcs
        radius = 0.4  # distance from center
        self.positions = nn.Parameter(
            torch.stack([
                self.mu[0] + radius * torch.cos(angles) + torch.randn(1),
                self.mu[1] + radius * torch.sin(angles) + torch.randn(1)
            ], dim=1)
        )  # [num_funcs, 2]

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
        # Shared global input transform: single a, b applied to all functions
        # a: [1], x: [N] -> broadcast to [num_funcs, N]
        input_x = self.a * x.unsqueeze(0) + self.b  # [num_funcs, N]

        weights = self.EvaluateNormalDistribution2D()  # [num_funcs]
        f_x = self.EvalFunction(input_x)  # [num_funcs, N]

        # Shared global output scale/shift: single c, d applied to all functions
        # c: [1], f_x: [num_funcs, N] -> [num_funcs, N]
        F_per_func = self.c * f_x + self.d  # [num_funcs, N]

        # Weighted sum across functions
        F = (F_per_func * weights.unsqueeze(1)).sum(dim=0)  # [N]

        return F


if __name__ == "__main__":
    # Generate synthetic data
    # torch.manual_seed(42)
    X_ref = torch.linspace(0.0, 1.0, 150)
    Y_ref = torch.sin(25*X_ref-6.0) # torch.sin(X_ref) torch.log(X_ref+5) torch.exp(X_ref)
    Y_ref = Y_ref / Y_ref.max()

    model = GaussianFixedOriginNetwork()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    losses = []
    alpha = 0.01

    max_iter = 10_000
    iter = 0
    tol = 1e-5
    # for it in range(max_iter + 1):
    while iter < max_iter:
        iter += 1
        optimizer.zero_grad()
        Y_pred = model(X_ref)
        loss = torch.sqrt(torch.sum((Y_pred - Y_ref) ** 2) / torch.sum((Y_ref) ** 2))
        # loss += alpha * (model.sigma)  # sigma penalty
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if iter % 500 == 0:
            print(f"Iter {iter}, Loss {loss.item():.6e}")
        if loss.item() < tol:
            print(f"Converged at iter {iter}, Loss {loss.item():.6e}")
            break

    # Print parameters with Gaussian relevance
    weights = model.EvaluateNormalDistribution2D().detach()
    print(f"\nTrained parameters (shared global abcd) with Gaussian relevance:")
    print(f"  Shared global parameters: a={model.a.item():.4f}, b={model.b.item():.4f}, c={model.c.item():.4f}, d={model.d.item():.4f}")
    print(f"  {'Func':>6s} | {'Label':>20s} | {'pos_x':>8s} | {'pos_y':>8s} | {'weight':>8s}")
    print(f"  {'-'*6}-|-{'-'*20}-|-{'-'*8}-|-{'-'*8}-|-{'-'*8}")
    for i in range(model.num_funcs):
        print(f"  {i:>6d} | {model.symbolic_function_labels[i]:>20s} | {model.positions[i,0].item():>8.4f} | {model.positions[i,1].item():>8.4f} | {weights[i].item():>8.4e}")
    print(f"\n  sigma={model.sigma.item():.4f}, Gaussian center mu={model.mu.tolist()}")
    
    # Summary: functions with significant contribution (weight > 1% of max)
    threshold = weights.max() * 0.01
    active = [i for i in range(model.num_funcs) if weights[i] > threshold]
    print(f"\n  Active functions (weight > 1% of max, threshold={weights.max()*0.01:.4e}): {active}")
    for i in active:
        print(f"    func {i} ({model.symbolic_function_labels[i]:>20s}): weight={weights[i].item():.4e}, c={model.c.item():.4f}")

    # Plot fitted function
    plt.figure(figsize=(8,4))
    plt.scatter(X_ref.numpy(), Y_ref.numpy(), label="Target", color="blue")
    with torch.no_grad():
        Y_pred = model(X_ref)
        print(f"[DEBUG] X_ref shape: {X_ref.shape}")
        print(f"[DEBUG] Y_pred shape: {Y_pred.shape}")
        print(f"[DEBUG] Y_ref shape: {Y_ref.shape}")
        # Ensure Y_pred is 1D for plotting
        if Y_pred.dim() == 2:
            print("[DEBUG] Y_pred is 2D - taking first row for plot")
            Y_pred_plot = Y_pred[0]
        else:
            Y_pred_plot = Y_pred
        plt.plot(X_ref.numpy(), Y_pred_plot.numpy(), "r-", label="Fitted")
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
