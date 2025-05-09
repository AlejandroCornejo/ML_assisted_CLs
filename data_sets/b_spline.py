import torch
import numpy as np
import matplotlib.pyplot as plt

class BSpline:
    # https://en.wikipedia.org/wiki/B-spline https://rohangautam.github.io/blog/b_spline_intro/
    def __init__(self, num_knots, order, x_data, coefficients):
        """
        Parameters:
            num_knots (int): Number of knots for the B-spline.
            order (int): Polynomial order (e.g., 3 for cubic).
            x_data (array-like): Full x-domain to evaluate the spline (not just a min/max).
            coefficients (torch.tensor): Coefficients for the spline basis functions (to be optimized).
        """
        self.num_knots = num_knots
        self.order = order
        self.x_data = np.asarray(x_data)
        self.x_min, self.x_max = np.min(x_data), np.max(x_data)
        self.coefficients = coefficients  # torch.tensor, differentiable

        # Uniform open knot vector in parametric space [0, 1]
        self.knot_vector = np.concatenate((
            np.zeros(order),
            np.linspace(0, 1, num_knots - order + 1),
            np.ones(order)
        ))

    def _cox_de_boor(self, x, i, k, t):
        """
        Recursive Cox-de Boor formula to compute B-spline basis functions.
        B_{i,k}(x) = (x - t_i) / (t_{i+k} - t_i) * B_{i,k-1}(x)
                   + (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
        """
        # Number of B-spline basis functions is determined by:
        #     num_basis = num_knots - order - 1
        # where:
        #     num_knots = total number of knots in the knot vector
        #     order     = polynomial order (e.g., 4 for cubic, since degree = order - 1)
        #
        # Each basis function is defined over (order + 1) consecutive knots,
        # so the total number of basis functions is reduced accordingly.
        #
        # Example:
        #     num_knots = 10, order = 4 (cubic) ⇒ num_basis = 10 - 4 - 1 = 5

        if k == 0:
            if i == len(t) - 2:  # Special case for the last knot
                return 1.0 if t[i] <= x <= t[i + 1] else 0.0
            return 1.0 if t[i] <= x < t[i + 1] else 0.0
        denom1 = t[i + k] - t[i]
        denom2 = t[i + k + 1] - t[i + 1]
        term1 = 0.0 if denom1 == 0 else (x - t[i]) / denom1 * self._cox_de_boor(x, i, k - 1, t)
        term2 = 0.0 if denom2 == 0 else (t[i + k + 1] - x) / denom2 * self._cox_de_boor(x, i + 1, k - 1, t)
        return term1 + term2

    def evaluate(self, x_real):
        """
        Evaluate the spline at a real-world x by mapping it to [0, 1],
        and summing weighted basis functions.
        """
        x_param = (x_real - self.x_min) / (self.x_max - self.x_min)
        val = 0.0
        for i in range(len(self.coefficients)):
            b = self._cox_de_boor(x_param, i, self.order - 1, self.knot_vector)
            val += self.coefficients[i].item() * b
        return val

    def plot(self, num_points=200):
        x_vals = np.linspace(self.x_min, self.x_max, num_points)
        y_vals = [self.evaluate(x) for x in x_vals]

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label="B-spline curve")
        # ctrl_x = np.linspace(self.x_min, self.x_max, len(self.coefficients))
        # plt.plot(ctrl_x, self.coefficients.detach().numpy(), 'ro--', label="Coefficients")
        plt.xlabel("x")
        plt.ylabel("Spline Value")
        plt.title("B-spline Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_basis_functions(self, num_points=300):
        """
        Plot all B-spline basis functions over the domain.
        """
        x_vals = np.linspace(self.x_min, self.x_max, num_points)
        x_param = (x_vals - self.x_min) / (self.x_max - self.x_min)

        plt.figure(figsize=(10, 6))
        for i in range(len(self.coefficients)):
            y_vals = [self._cox_de_boor(x, i, self.order - 1, self.knot_vector) for x in x_param]
            plt.plot(x_vals, y_vals, label=f"B{i}(x)")

        plt.title(f"B-spline Basis Functions (order={self.order})")
        plt.xlabel("x")
        plt.ylabel("Basis Value")
        plt.legend()
        plt.grid(True)
        plt.show()




# Suppose this is your full history of x-coordinates
x_history = np.linspace(-1, 6, 300)
coeffs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], requires_grad=True)

bspline = BSpline(num_knots=9, order=4, x_data=x_history, coefficients=coeffs)
bspline.plot_basis_functions()
bspline.plot()
