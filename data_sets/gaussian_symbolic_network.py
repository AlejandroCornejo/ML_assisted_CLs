import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class GaussianSymbolicNetwork(nn.Module):
    def __init__(self, init_params=None):
        """
        Initialize the linear transformation module

        Args:
            init_params: Dictionary with initial values for parameters.
                         If None, uses random initialization.
        """
        super(GaussianSymbolicNetwork, self).__init__()

        # Initialize parameters
        if init_params is None:
            init_params = {
                "a": torch.tensor(1.0),
                "b": torch.tensor(0.0),
                "c": torch.tensor(1.0),
                "d": torch.tensor(0.0),
            }

        # Convert to trainable parameters
        self.a = nn.Parameter(init_params["a"].float())
        self.b = nn.Parameter(init_params["b"].float())
        self.c = nn.Parameter(init_params["c"].float())
        self.d = nn.Parameter(init_params["d"].float())

        # Parameters for normal distribution
        self.norm_factor = 9.0  # number of functions
        self.mu = nn.Parameter(torch.tensor(1.0 / self.norm_factor))  # scaled down for stability
        self.sigma = nn.Parameter(torch.tensor(0.01))
        # self.sigma = torch.tensor(0.1)

    def EvaluateNormalDistribution(self, i):
        """
        Evaluate the normal distribution function

        Args:
            x: Input tensor

        Returns:
            Normal distribution evaluated at x
        """
        # return (1.0 / torch.sqrt(2 * np.pi * self.sigma**2)) * torch.exp(-0.5 * (i-self.mu)**2 / (self.sigma**2))
        real_mu = self.mu * self.norm_factor
        return torch.exp(-0.5 * (i - real_mu) ** 2 / (self.sigma**2))  # normalized

    def EvalFunction(self, i, x):
        if i == 0:
            return torch.zeros_like(x)
        elif i == 1:
            return x
        elif i == 2:
            return x**2
        elif i == 3:
            return x**3
        elif i == 4:
            return x**4
        elif i == 5:
            return torch.exp(x)
        elif i == 6:
            return torch.log(torch.abs(x) + 1e-6)  # avoid log(0)
        elif i == 7:
            return torch.tanh(x)
        elif i == 8:
            return torch.sin(x)
        elif i == 9:
            return torch.sqrt(torch.abs(x))
        else:
            raise ValueError("Index i must be between 0 and 9")

    def forward(self, x):
        """
        Forward pass: f(x) = c*F(a*x + b) + d

            being F:
                0 -> 0.0
                1 -> x
                2 -> x^2
                ...
        Args:
            x: Input tensor
        """
        input_x = self.a * x + self.b
        F = 0.0
        for i in range(int(self.norm_factor) + 1):
            F += self.EvalFunction(i, input_x) * self.EvaluateNormalDistribution(i)
        return self.c * F + self.d


def fit_model(X_ref, Y_ref, max_iter, lr, init_params=None, verbose=True):
    """
    Fit the model to reference data using BFGS optimizer

    Args:
        X_ref: Reference input points (torch tensor)
        Y_ref: Reference output points (torch tensor)
        max_iter: Maximum number of iterations
        lr: Learning rate for BFGS
        init_params: Initial parameter values
        verbose: Whether to print progress

    Returns:
        model: Trained model
        losses: List of loss values during training
    """
    # Create model
    model = GaussianSymbolicNetwork(init_params)

    # Define loss function (L2 loss)
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.LBFGS(model.parameters(), lr=lr)

    # Store losses
    losses = []

    # Closure function for BFGS
    def closure():
        optimizer.zero_grad()
        output = model(X_ref)
        loss = criterion(output, Y_ref)
        loss.backward()
        return loss

    # Training loop
    if verbose:
        print("Starting training...")

    for i in range(max_iter):
        loss = optimizer.step(closure)
        losses.append(loss.item())

        if verbose and (i % 100 == 0 or i == max_iter - 1):
            print(f"Iteration {i+1}/{max_iter}, Loss: {loss.item():.6f}")

    return model, losses

def plot_results(X_ref, Y_ref, model, losses):
    """
    Plot the real data vs predicted data and training loss

    Args:
        X_ref: Reference input points
        Y_ref: Reference output points
        model: Trained model
        losses: List of loss values during training
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Real vs Predicted data
    X_plot = torch.linspace(X_ref.min(), X_ref.max(), 300)
    with torch.no_grad():
        Y_pred = model(X_plot)

    ax1.scatter(
        X_ref.numpy(), Y_ref.numpy(), alpha=0.7, label="Real Data", color="blue", s=20
    )
    ax1.plot(X_plot.numpy(), Y_pred.numpy(), "r-", linewidth=2, label="Fitted Function")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Real Data vs Fitted Function")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add parameter information to the plot
    param_text = (
        f"Parameters:\n"
        f"a = {model.a.item():.3f}\n"
        f"b = {model.b.item():.3f}\n"
        f"c = {model.c.item():.3f}\n"
        f"d = {model.d.item():.3f}"
    )
    ax1.text(
        0.02,
        0.98,
        param_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Plot 2: Training loss
    ax2.semilogy(losses, linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss (MSE)")
    ax2.set_title("Training Loss Convergence")
    ax2.grid(True, alpha=0.3)

    # Add final loss to the plot
    final_loss = losses[-1]
    ax2.text(
        0.02,
        0.98,
        f"Final Loss: {final_loss:.6f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate some sample data
    torch.manual_seed(42)
    X_ref = torch.linspace(-1, 1, 100)

    # True parameters for generating data
    true_a = 2.0
    true_b = 1.0
    true_c = 0.5
    true_d = -1.0

    # Generate reference Y values with some noise
    order_poly = 1.0
    Y_ref = (
        true_c * (true_a * torch.exp(X_ref) + true_b) + true_d
    )   # + 0.2 * torch.randn_like(X_ref)
    Y_ref = Y_ref / Y_ref.max()  # normalize

    # Fit the model
    model, losses = fit_model(X_ref, Y_ref, max_iter=5000, lr=0.01, verbose=True)

    # Print final parameters
    print("\nFinal parameters:")
    print(f"a: {model.a.item():.4f} (true: {true_a})")
    print(f"b: {model.b.item():.4f} (true: {true_b})")
    print(f"c: {model.c.item():.4f} (true: {true_c})")
    print(f"d: {model.d.item():.4f} (true: {true_d})")
    print(f"mu: {model.norm_factor * model.mu.item():.4f}")
    print(f"sigma: {model.sigma.item():.4f}")

    # Plot the results
    plot_results(X_ref, Y_ref, model, losses)
