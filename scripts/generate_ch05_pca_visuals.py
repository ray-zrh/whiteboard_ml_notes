
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Set random seed
np.random.seed(42)

def generate_data(n_samples=100):
    # Mean and Covariance
    mean = [0, 0]
    cov = [[2, 1.2], [1.2, 1]]  # Correlated data
    X = np.random.multivariate_normal(mean, cov, n_samples)
    # Center data explicitly
    X = X - np.mean(X, axis=0)
    return X, cov

def plot_pca_concept(X, cov, output_path):
    # Calculate Eigenvalues and Eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    # Sort by eigenvalue in descending order
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot centered data
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Centered Data $x_i - \overline{x}$', color='#1f77b4')

    # Scale for vectors
    scale = 2.5

    # Plot Principal Components (Eigenvectors)
    colors = ['#d62728', '#2ca02c'] # Red for PC1, Green for PC2
    labels = ['$u_1$ (PC1: Max Variance)', '$u_2$ (PC2)']

    for i in range(2):
        vector = vecs[:, i]
        length = np.sqrt(vals[i]) * scale

        # Draw vector
        arrow = FancyArrowPatch((0, 0), (vector[0]*length, vector[1]*length),
                                arrowstyle='-|>', mutation_scale=20,
                                color=colors[i], lw=2, label=labels[i])
        ax.add_patch(arrow)

        # Draw axis line (dashed) extended
        line_x = np.array([-vector[0]*4, vector[0]*4])
        line_y = np.array([-vector[1]*4, vector[1]*4])
        ax.plot(line_x, line_y, color=colors[i], linestyle='--', alpha=0.4, lw=1)

    # Illustrate Projection for a few points onto PC1
    u1 = vecs[:, 0]
    n_projections = 5
    # Pick a few representative points (e.g., furthest)
    distances = np.linalg.norm(X, axis=1)
    idx = np.argsort(distances)[-n_projections:]

    for i in idx:
        point = X[i]
        # Projection scalar z = x^T u1
        proj_scalar = np.dot(point, u1)
        # Projection vector p = z * u1
        proj_vec = proj_scalar * u1

        # Draw projection line (from point to axis)
        ax.plot([point[0], proj_vec[0]], [point[1], proj_vec[1]],
                color='gray', linestyle=':', alpha=0.6)
        # Draw projected point
        ax.scatter(proj_vec[0], proj_vec[1], color='#d62728', s=20, alpha=0.8)

    # Add text annotation for variance
    ax.text(-3, 3, f'$\lambda_1 (Variance) = {vals[0]:.2f}$', color=colors[0], fontsize=12)
    ax.text(-3, 2.5, f'$\lambda_2 (Variance) = {vals[1]:.2f}$', color=colors[1], fontsize=12)

    # Formatting
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('PCA: Geometric Interpretation\nDirection of Maximum Variance & Projection', fontsize=15)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    X, cov = generate_data()
    output_path = 'notes/chapters/assets/ch05_pca_concept.png'
    plot_pca_concept(X, cov, output_path)
