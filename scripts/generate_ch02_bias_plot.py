
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the proper Gaussian
mu_true = 0
sigma_true = 2
x = np.linspace(-6, 6, 200)
pdf = stats.norm.pdf(x, mu_true, sigma_true)

# Generate a small sample that clearly has a different mean
# We pick points manually to ensure sample mean is distinct from true mean for visualization
data_points = np.array([-1.5, 0.5, 3.5])
mu_mle = np.mean(data_points) # 0.833

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'k-', linewidth=2, label='True Distribution $\mathcal{N}(\mu, \sigma^2)$')

# Plot data points
plt.scatter(data_points, np.zeros_like(data_points), color='red', s=100, zorder=5, marker='x', label='Data Points ($x_i$)')

# Plot True Mean line
plt.axvline(mu_true, color='green', linestyle='--', linewidth=2, label=r'True Mean $\mu$')
plt.text(mu_true - 0.2, 0.02, r'$\mu$', color='green', fontsize=14, ha='right')

# Plot MLE Mean (Sample Mean) line
plt.axvline(mu_mle, color='blue', linestyle='--', linewidth=2, label=r'Sample Mean $\mu_{MLE}$')
plt.text(mu_mle + 0.2, 0.02, r'$\mu_{MLE}$', color='blue', fontsize=14, ha='left')

# Add interpretation text
# Illustration of distances
y_height = 0.05
for point in data_points:
    # Distance to True Mean
    # plt.plot([point, mu_true], [y_height, y_height], color='green', alpha=0.3)
    pass

plt.title(r'Bias Intuition: $\sum(x_i - \mu_{MLE})^2 < \sum(x_i - \mu)^2$', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.yticks([]) # Hide y axis values for cleaner look
plt.legend(loc='upper right')

# Annotate logic
# Moved lower to avoid legend overlap
plt.annotate(
    "Data 'pulls' the estimated mean\ntowards itself.",
    xy=(mu_mle, 0.15), xytext=(mu_mle + 1.5, 0.12),
    arrowprops=dict(facecolor='black', shrink=0.05),
    fontsize=12
)

plt.annotate(
    "Variance calculated from $\mu_{MLE}$\nis smaller than from true $\mu$",
    xy=(0, 0.1), xytext=(-5, 0.12),
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
output_path = 'notes/chapters/assets/ch02_bias_intuition.png'
plt.savefig(output_path, dpi=300)
print(f"Image saved to {output_path}")
