
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """
    A coupled quadratic function to serve as the objective function (ELBO).
    f(x, y) = -(x^2 + 2y^2 - 1.5xy)
    We want to maximize this function. The contour lines will be ellipses.
    """
    return -(x**2 + 2*y**2 - 1.5*x*y)

def coordinate_ascent(start_point, iterations=5):
    """
    Perform coordinate ascent optimization.
    Maximize f(x, y) iteratively.
    """
    x, y = start_point
    path = [(x, y)]

    for _ in range(iterations):
        # 1. Maximize over x (Fix y)
        # d/dx (-(x^2 + 2y^2 - 1.5xy)) = -2x + 1.5y = 0 => x = 0.75y
        x_new = 0.75 * y
        path.append((x_new, y))
        x = x_new

        # 2. Maximize over y (Fix x)
        # d/dy (-(x^2 + 2y^2 - 1.5xy)) = -4y + 1.5x = 0 => y = (1.5/4)x = 0.375x
        y_new = 0.375 * x
        path.append((x, y_new))
        y = y_new

    return np.array(path)

def main():
    # Setup grid for contour plot
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Perform Coordinate Ascent
    start_point = (-2.5, 2.0)
    path = coordinate_ascent(start_point, iterations=6)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Contour plot
    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)

    # Plot path
    # Draw arrows for each step
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Mark start and end
    ax.scatter(path[0, 0], path[0, 1], color='blue', label='Start')
    ax.scatter(0, 0, color='green', marker='*', s=200, label='Optimum')

    # Annotate steps
    ax.text(path[0, 0], path[0, 1] + 0.2, 'Start', color='blue', fontsize=10, ha='center')

    # Label a vertical move as E-step (maximizing q)
    # Label a horizontal move as M-step (maximizing theta)
    # Note: In our example, x and y are abstract. Let's assume x=theta, y=q for illustration.
    # Move 1: x updates (M-step), Move 2: y updates (E-step).

    # Just generic labels "Step 1", "Step 2" to show the zigzag
    ax.text(path[1, 0], path[1, 1] + 0.1, '1', color='red', fontsize=10)
    ax.text(path[2, 0] + 0.1, path[2, 1], '2', color='red', fontsize=10)

    ax.set_title("Coordinate Ascent (EM Algorithm Concept)")
    ax.set_xlabel("Parameter $\\theta$")
    ax.set_ylabel("Distribution $q(Z)$ (Abstract)")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Save
    output_path = "notes/chapters/assets/ch10_em_coordinate_ascent.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
