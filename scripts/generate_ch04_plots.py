
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
output_dir = "notes/chapters/assets"
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def plot_linear_boundary():
    """Generates a plot showing a linear decision boundary."""
    np.random.seed(42)
    # Generate data
    n_samples = 20
    X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    X2 = np.random.randn(n_samples, 2) + np.array([-2, -2])

    # Decision boundary: x2 = -x1 (w=[1,1], b=0 for simplicity in concept)
    x_line = np.linspace(-4, 4, 100)
    y_line = -x_line

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot data
    ax.scatter(X1[:, 0], X1[:, 1], c='blue', marker='o', label='Class +1')
    ax.scatter(X2[:, 0], X2[:, 1], c='red', marker='x', label='Class -1')

    # Plot boundary
    ax.plot(x_line, y_line, 'k-', linewidth=2, label=r'$w^T x + b = 0$')

    # Plot normal vector w
    origin = np.array([0, 0])
    w = np.array([1, 1])
    ax.arrow(0, 0, 1, 1, head_width=0.3, head_length=0.3, fc='black', ec='black')
    ax.text(1.2, 1.2, r'$w$', fontsize=12, fontweight='bold')

    # Annotations
    ax.text(2, 2, 'Positive Region\n$w^T x > 0$', ha='center', fontsize=10, color='blue')
    ax.text(-2, -2, 'Negative Region\n$w^T x < 0$', ha='center', fontsize=10, color='red')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend()
    ax.set_title("Linear Decision Boundary")

    # Remove ticks/spines for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ch04_linear_boundary.png"), dpi=300)
    plt.close()
    print("Saved ch04_linear_boundary.png")

def plot_perceptron_update():
    """Generates a plot showing a Perceptron update step."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Old weight w_t
    w_t = np.array([1.5, 0.5])

    # Misclassified point x_i (y_i = +1)
    x_i = np.array([0.5, 1.5])

    # New weight w_{t+1} = w_t + x_i
    w_next = w_t + x_i

    # Plot vectors
    ax.arrow(0, 0, w_t[0], w_t[1], head_width=0.1, head_length=0.1, fc='gray', ec='gray', length_includes_head=True, label=r'$w_t$')
    ax.arrow(0, 0, x_i[0], x_i[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', length_includes_head=True, label=r'$x_i (y_i=+1)$')
    ax.arrow(0, 0, w_next[0], w_next[1], head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True, label=r'$w_{t+1}$')

    # Show the addition with dashed line
    ax.plot([w_t[0], w_next[0]], [w_t[1], w_next[1]], 'b--', alpha=0.6)

    # Text
    ax.text(w_t[0]+0.1, w_t[1], r'$w_t$', fontsize=12)
    ax.text(x_i[0]-0.2, x_i[1]+0.1, r'$x_i$', fontsize=12, color='blue')
    ax.text(w_next[0]+0.1, w_next[1], r'$w_{t+1} = w_t + \eta x_i$', fontsize=12, fontweight='bold')

    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect('equal')
    ax.set_title("Perceptron Weight Update")
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ch04_perceptron_update.png"), dpi=300)
    plt.close()
    print("Saved ch04_perceptron_update.png")

def plot_lda_projection():
    """Generates a plot for LDA projection concept."""
    np.random.seed(1)

    # Two classes with covariance
    cov = [[1, 0.8], [0.8, 1]]
    X1 = np.random.multivariate_normal([2, 4], cov, 30)
    X2 = np.random.multivariate_normal([4, 2], cov, 30)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(X1[:, 0], X1[:, 1], c='blue', alpha=0.6, label='Class 1')
    ax.scatter(X2[:, 0], X2[:, 1], c='red', alpha=0.6, label='Class 2')

    # Bad projection (project onto horizontal axis) - high overlap
    # Good projection (LDA direction - diagonal roughly)

    # Approximate LDA direction for visualization (y = -x + c)
    x_proj = np.linspace(-1, 7, 100)
    y_proj = -x_proj + 6  # Perpendicular to the mean difference vector roughly
    # Actually LDA projects onto the line connecting means (roughly, if isotropic),
    # but here we want to find w that maximizes separation.
    # With this covariance, the optimal w is roughly [1, -1].

    # Let's show the projection line w
    w = np.array([1, -1]) # Optimal direction roughly
    w = w / np.linalg.norm(w)

    # Draw the line defined by w passing through origin (or offset for visibility)
    # Line equation: point p = t * w.
    t_vals = np.linspace(-2, 8, 100)
    line_pts = np.outer(t_vals, w) + np.array([3, 3]) # Shift to center

    ax.plot(line_pts[:, 0], line_pts[:, 1], 'k-', linewidth=1, alpha=0.5, label='Projection Line')

    # Project points onto this line
    # p_proj = (p . w) * w
    # We'll just draw lines from a few points to the line

    # Add title and remove axis
    ax.set_title("LDA: Projection maximizing class separation")
    ax.legend()
    # plt.axis('off') # Keep grid maybe?

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ch04_lda_projection.png"), dpi=300)
    plt.close()
    print("Saved ch04_lda_projection.png")

def plot_sigmoid():
    """Generates Sigmoid function plot."""
    z = np.linspace(-6, 6, 100)
    sigma = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(z, sigma, 'b-', linewidth=2, label=r'$\sigma(z) = \frac{1}{1+e^{-z}}$')
    ax.plot([-6, 6], [0.5, 0.5], 'k--', alpha=0.5)
    ax.plot([0, 0], [0, 1], 'k--', alpha=0.5)

    ax.set_xlabel('z')
    ax.set_ylabel(r'$\sigma(z)$')
    ax.set_title("Sigmoid Function")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ch04_sigmoid.png"), dpi=300)
    plt.close()
    print("Saved ch04_sigmoid.png")

if __name__ == "__main__":
    plot_linear_boundary()
    plot_perceptron_update()
    plot_lda_projection()
    plot_sigmoid()
