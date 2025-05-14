import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


def evaluate_bspline(t, c, k, x):
    """
    Evaluate the B-spline at given x values.

    Parameters:
        t (list): Knot vector.
        c (list): Coefficients of the spline.
        k (int): Degree of the spline.
        x (array-like): Points at which to evaluate the spline.

    Returns:
        array: Evaluated spline values.
    """
    spl = BSpline(t, c, k)
    return spl(x)


def plot_bspline(t, c, k, num_points=200):
    """
    Plot the B-spline and its basis functions.

    Parameters:
        t (list): Knot vector.
        c (list): Coefficients of the spline.
        k (int): Degree of the spline.
        num_points (int): Number of points for plotting.
    """
    # Create x values for evaluation
    x = np.linspace(min(t), max(t), num_points, endpoint=True)

    # Evaluate the B-spline
    spl = BSpline(t, c, k)
    y = spl(x)

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Top subplot: B-spline curve
    axs[0].plot(x, y, label="B-spline curve", color="blue")

    axs[0].set_title("B-spline Curve with Coefficients")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("Spline Value")
    axs[0].legend()
    axs[0].grid(True)

    # Bottom subplot: Basis functions
    for i in range(len(c)):
        basis = BSpline.basis_element(t[i:i + k + 2], extrapolate=False)
        axs[1].plot(x, basis(x), label=f"B{i}(x)")
    axs[1].set_title(f"B-spline Basis Functions (degree={k})")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Basis Value")
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    k = 2  # Degree of the spline
    t = [0, 0, 0, 1, 2, 2, 2]  # Knot vector
    c = [0, 1.1, 2.8, 6.8]  # Coefficients

    # Evaluate the spline at a specific point
    x_val = 2.5
    spl_val = evaluate_bspline(t, c, k, x_val)
    print(f"Spline value at x={x_val}: {spl_val}")

    # Plot the spline and its basis functions
    plot_bspline(t, c, k)