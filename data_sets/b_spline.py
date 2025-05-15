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

        uses scipy.interpolate.BSpline to create the spline object and evaluate it.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html

    Returns:
        array: Evaluated spline values.
    """
    spl = BSpline(t, c, k)
    return spl(x)


def check_convexity_and_monotonicity(c):
    """
    Check if the coefficients satisfy the convexity and monotonicity condition:
    coef_{i+2} - coef_{i+1} >= coef_{i+1} - coef_{i} >= 0.0 for all coefficients.

    Parameters:
        c (list): Coefficients of the spline.

    Returns:
        bool: True if the condition is satisfied, False otherwise.
    """
    for i in range(len(c) - 2):
        diff1 = c[i + 1] - c[i]
        diff2 = c[i + 2] - c[i + 1]
        if not (diff2 >= diff1 >= 0.0):
            return False
    return True


def plot_bspline(t, c, k, x_range, num_points=200):
    """
    Plot the B-spline and its basis functions.

    Parameters:
        t (list): Knot vector.
        c (list): Coefficients of the spline.
        k (int): Degree of the spline.
        num_points (int): Number of points for plotting.
    """
    # Check convexity and monotonicity
    if not check_convexity_and_monotonicity(c):
        print("The coefficients do not satisfy the convexity and monotonicity condition.")

    # Create x values for evaluation
    # x = np.linspace(min(x_range), max(x_range), num_points, endpoint=True)
    x = x_range

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
    t = [0, 0, 0, 0.5, 1, 1, 1]  # Knot vector
    c = [0, 1.1, 3, 4]  # Coefficients

    x = np.linspace(0.0, 15.0, 500, endpoint=True)

    # Evaluate the spline at a specific point
    x_val = 2.5
    spl_val = evaluate_bspline(t, c, k, x_val)
    print(f"Spline value at x={x_val}: {spl_val}")

    plot_bspline(t, c, k, x)
