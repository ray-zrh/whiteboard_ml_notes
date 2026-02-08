
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Ensure covariance is numpy array
    covariance = np.array(covariance)

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle=angle, **kwargs))

def main():
    np.random.seed(42)

    # 1. Generate Data from 2 Gaussians (Bimodal)
    mean1 = [2, 2]
    cov1 = [[1, 0.5], [0.5, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, 200)

    mean2 = [7, 7]
    cov2 = [[1.5, -0.8], [-0.8, 1.5]]
    data2 = np.random.multivariate_normal(mean2, cov2, 200)

    X = np.vstack([data1, data2])

    # 2. Fit Single Gaussian
    mu_single = np.mean(X, axis=0)
    cov_single = np.cov(X, rowvar=False)

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Data
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, s=10, label='Data Points')

    # Plot Single Gaussian (Bad Fit) - Red dashed
    draw_ellipse(mu_single, cov_single, ax=ax, edgecolor='red', linestyle='--', facecolor='none', alpha=0.3, label='Single Gaussian Fit (Bad)')

    # Plot True Gaussians (GMM Idea) - Green solid
    draw_ellipse(mean1, np.array(cov1), ax=ax, edgecolor='green', linestyle='-', facecolor='none', alpha=0.9, label='True/GMM Fit (Good)')
    draw_ellipse(mean2, np.array(cov2), ax=ax, edgecolor='green', linestyle='-', facecolor='none', alpha=0.9)

    # Add text labels
    ax.text(mu_single[0], mu_single[1], 'Single Gaussian\n(Unimodal Assumption)',
            color='red', ha='center', va='center', fontweight='bold', alpha=0.7)

    ax.text(mean1[0], mean1[1], 'Component 1', color='green', ha='center', va='center', fontweight='bold')
    ax.text(mean2[0], mean2[1], 'Component 2', color='green', ha='center', va='center', fontweight='bold')

    ax.set_title("Limitation of Single Gaussian vs. GMM Concept")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # Create custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', linestyle='--'),
                    Line2D([0], [0], color='green', linestyle='-')]
    ax.legend(custom_lines, ['Single Gaussian (Poor Fit)', 'Mixture of Gaussians (Good Fit)'])

    plt.grid(True, alpha=0.3)

    # Save
    output_path = "notes/chapters/assets/ch02_gmm_concept.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
