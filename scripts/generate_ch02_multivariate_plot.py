
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set random seed
np.random.seed(42)

# Grid setup
x, y = np.mgrid[-4:4:.01, -4:4:.01]
pos = np.dstack((x, y))

# Parameters
mu = np.array([0, 0])
sigma = np.array([[1.5, 0.8], [0.8, 1.5]]) # Correlated
rv = multivariate_normal(mu, sigma)
pdf = rv.pdf(pos)

plt.figure(figsize=(8, 6))

# Contour plot
contour = plt.contourf(x, y, pdf, levels=15, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label='Probability Density')

# Add visual elements
# Mean
plt.scatter([0], [0], color='red', s=100, marker='x', label='Mean $\mu$', zorder=10)

# Annotate eigenvectors (principal axes)
vals, vecs = np.linalg.eigh(sigma)
order = vals.argsort()[::-1]
vals, vecs = vals[order], vecs[:, order]
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

# Ellipse for 2std
from matplotlib.patches import Ellipse
ell = Ellipse(xy=(0, 0),
              width=2 * np.sqrt(vals[0]) * 2,
              height=2 * np.sqrt(vals[1]) * 2,
              angle=theta, color='white', fill=False, linestyle='--', linewidth=2, label='Confidence Ellipse ($2\sigma$)')

plt.gca().add_patch(ell)

# Vectors
plt.arrow(0, 0, vecs[0, 0]*np.sqrt(vals[0]), vecs[1, 0]*np.sqrt(vals[0]),
          head_width=0.2, head_length=0.2, fc='white', ec='white', linewidth=2)
plt.arrow(0, 0, vecs[0, 1]*np.sqrt(vals[1]), vecs[1, 1]*np.sqrt(vals[1]),
          head_width=0.2, head_length=0.2, fc='white', ec='white', linewidth=2)


plt.title('Bivariate Gaussian Distribution $\mathcal{N}(\mu, \Sigma)$', fontsize=16)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Save
output_path = 'notes/chapters/assets/ch02_multivariate_gaussian.png'
plt.savefig(output_path, dpi=300)
print(f"Image saved to {output_path}")
