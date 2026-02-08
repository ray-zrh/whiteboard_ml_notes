
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle=angle, **kwargs))

def main():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mean = [0, 0]

    # 1. Full Covariance
    # Correlated data, rotated ellipse
    cov_full = [[1, 0.8], [0.8, 1]]
    data_full = np.random.multivariate_normal(mean, cov_full, 200)

    axes[0].scatter(data_full[:, 0], data_full[:, 1], alpha=0.3, s=10)
    draw_ellipse(mean, cov_full, ax=axes[0], edgecolor='red', facecolor='none', alpha=0.7)
    axes[0].set_title("Full Covariance\n(General Ellipse)")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].text(0, -3.5, r"$\Sigma_{full}$ (Non-zero off-diagonals)",
                 ha='center', fontsize=10)

    # 2. Diagonal Covariance
    # Uncorrelated data, axis-aligned ellipse
    cov_diag = [[1, 0], [0, 3]]
    data_diag = np.random.multivariate_normal(mean, cov_diag, 200)

    axes[1].scatter(data_diag[:, 0], data_diag[:, 1], alpha=0.3, s=10)
    draw_ellipse(mean, cov_diag, ax=axes[1], edgecolor='green', facecolor='none', alpha=0.7)
    axes[1].set_title("Diagonal Covariance\n(Axis-Aligned)")
    axes[1].set_xlabel("x1")
    # axes[1].set_ylabel("x2")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].text(0, -3.5, r"$\Sigma = \mathrm{diag}(\sigma_1^2, \sigma_2^2)$",
                 ha='center', fontsize=10)

    # 3. Spherical / Isotropic Covariance
    # Circle
    cov_sphere = [[1, 0], [0, 1]]
    data_sphere = np.random.multivariate_normal(mean, cov_sphere, 200)

    axes[2].scatter(data_sphere[:, 0], data_sphere[:, 1], alpha=0.3, s=10)
    draw_ellipse(mean, cov_sphere, ax=axes[2], edgecolor='blue', facecolor='none', alpha=0.7)
    axes[2].set_title("Spherical Covariance\n(Circle)")
    axes[2].set_xlabel("x1")
    # axes[2].set_ylabel("x2")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    axes[2].text(0, -3.5, r"$\Sigma = \sigma^2 I$",
                 ha='center', fontsize=12)

    plt.tight_layout()

    # Save
    output_path = "notes/chapters/assets/ch02_covariance_types.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
