import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import matplotlib.patches as patches
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

def plot_1d_gmm():
    """Generates a 1D Gaussian Mixture Model plot."""
    x = np.linspace(-5, 12, 1000)

    # Parameters for two Gaussian components
    mu1, sigma1, w1 = 0, 1.5, 0.4
    mu2, sigma2, w2 = 6, 2.0, 0.6

    # Distributions
    pdf1 = w1 * norm.pdf(x, mu1, sigma1)
    pdf2 = w2 * norm.pdf(x, mu2, sigma2)
    pdf_total = pdf1 + pdf2

    plt.figure(figsize=(10, 6))

    # Plot individual components
    plt.plot(x, pdf1, 'r--', label=r'Component 1: $N(\mu={mu1}, \sigma={sigma1})$', alpha=0.7)
    plt.plot(x, pdf2, 'b--', label=r'Component 2: $N(\mu={mu2}, \sigma={sigma2})$', alpha=0.7)

    # Plot mixture
    plt.plot(x, pdf_total, 'k-', linewidth=2, label=r'GMM: $p(x)$')

    # Stylize
    plt.title('1D Gaussian Mixture Model', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '../notes/chapters/assets')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ch11_gmm_1d.png'), dpi=300)
    print(f"Generated 1D GMM plot: {os.path.join(output_dir, 'ch11_gmm_1d.png')}")
    plt.close()

def plot_2d_gmm():
    """Generates a 2D Gaussian Mixture Model plot with contours and samples."""
    np.random.seed(42)

    # Parameters for two 2D Gaussian components
    mu1 = np.array([2, 5])
    cov1 = np.array([[1.0, 0.3], [0.3, 1.0]])
    n1 = 150

    mu2 = np.array([7, 3])
    cov2 = np.array([[1.5, -0.7], [-0.7, 1.5]])
    n2 = 200

    # Generate samples
    X1 = np.random.multivariate_normal(mu1, cov1, n1)
    X2 = np.random.multivariate_normal(mu2, cov2, n2)
    X = np.vstack([X1, X2])

    # Create grid for contours
    x_min, x_max = -2, 12
    y_min, y_max = -2, 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    pos = np.dstack((xx, yy))

    # Calculate PDF for contours
    rv1 = multivariate_normal(mu1, cov1)
    rv2 = multivariate_normal(mu2, cov2)
    z1 = rv1.pdf(pos)
    z2 = rv2.pdf(pos)

    plt.figure(figsize=(10, 8))

    # Plot samples
    plt.scatter(X1[:, 0], X1[:, 1], alpha=0.6, marker='x', color='red', label='Cluster 1 Samples')
    plt.scatter(X2[:, 0], X2[:, 1], alpha=0.6, marker='x', color='blue', label='Cluster 2 Samples')

    # Plot contours
    plt.contour(xx, yy, z1, levels=5, colors='red', alpha=0.5)
    plt.contour(xx, yy, z2, levels=5, colors='blue', alpha=0.5)

    # Annotate
    plt.annotate(r'$N(\mu_1, \Sigma_1)$', xy=mu1, xytext=(mu1[0]-2, mu1[1]+2),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
    plt.annotate(r'$N(\mu_2, \Sigma_2)$', xy=mu2, xytext=(mu2[0]+2, mu2[1]+2),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

    plt.title('2D Gaussian Mixture Model with Contours', fontsize=16)
    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.legend()
    plt.tight_layout()

    # Save
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '../notes/chapters/assets')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ch11_gmm_2d.png'), dpi=300)
    print(f"Generated 2D GMM plot: {os.path.join(output_dir, 'ch11_gmm_2d.png')}")
    plt.close()

if __name__ == "__main__":
    plot_1d_gmm()
    plot_2d_gmm()
