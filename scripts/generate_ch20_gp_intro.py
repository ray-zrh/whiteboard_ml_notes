import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """
    Squared Exponential (RBF) Kernel.
    """
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

def plot_gp_intro():
    # 1. Domain
    x_min, x_max = 0, 10
    n_points = 200
    X = np.linspace(x_min, x_max, n_points).reshape(-1, 1)

    # 2. Kernel and Covariance
    K = rbf_kernel(X, X, length_scale=1.5, sigma_f=1.0)
    mu = np.zeros(len(X)) # Mean function m(x) = 0

    # 3. Sample paths
    n_samples = 3
    # Add small noise for numerical stability
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X)))
    samples = mu[:, np.newaxis] + np.dot(L, np.random.normal(size=(len(X), n_samples)))

    # 4. Plotting
    plt.figure(figsize=(10, 6))

    # Plot samples
    for i in range(n_samples):
        plt.plot(X, samples[:, i], lw=2, alpha=0.8, label=f'Sample {i+1}')

    # Plot Mean
    plt.plot(X, mu, 'k--', label='Mean m(t)', lw=2)

    # Plot 2 sigma confidence interval
    std = np.sqrt(np.diag(K))
    plt.fill_between(X.flatten(), mu - 2*std, mu + 2*std, color='gray', alpha=0.1, label='95% Confidence')

    # 5. Visualize Marginal Distributions (Slices)
    slice_points = [2.0, 5.0, 8.0]
    scale_factor = 0.5 # Scale the gaussian height for visualization

    for t_slice in slice_points:
        # Find index close to t_slice
        idx = np.abs(X - t_slice).argmin()
        x_val = X.flatten()[idx]
        y_mean = mu[idx]
        y_std = std[idx]

        # Create vertical gaussian
        y_vals = np.linspace(y_mean - 3*y_std, y_mean + 3*y_std, 100)
        # Flip x and y for vertical plotting, centered at x_val
        x_probs = norm.pdf(y_vals, y_mean, y_std)

        # Plot the "bell curve" vertically
        # We add the probability (scaled) to the x-position
        # Use fill_betweenx to shade
        plt.fill_betweenx(y_vals, x_val, x_val + x_probs * scale_factor, color='purple', alpha=0.3)
        plt.plot(x_val + x_probs * scale_factor, y_vals, color='purple', lw=1, alpha=0.6)

        # Plot the slice line
        plt.vlines(x_val, y_mean - 3*y_std, y_mean + 3*y_std, colors='purple', linestyles=':', alpha=0.5)
        plt.plot([x_val], [y_mean], 'ko', markersize=4)

        plt.text(x_val, y_mean - 3.5*y_std, f't={t_slice}', ha='center', va='top', color='purple')

    plt.title('Gaussian Process: Infinite Collection of Random Variables', fontsize=14)
    plt.xlabel('Input Space (e.g., Time t)', fontsize=12)
    plt.ylabel('Output Space (f(t))', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    import os
    os.makedirs('notes/chapters/assets', exist_ok=True)
    plt.savefig('notes/chapters/assets/ch20_gp_intro.png', dpi=150)
    print("Image saved to notes/chapters/assets/ch20_gp_intro.png")

if __name__ == "__main__":
    np.random.seed(42) # Reproducibility
    plot_gp_intro()
