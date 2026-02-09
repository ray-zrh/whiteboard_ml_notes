import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data for two classes
# Class 1: Centered at (2, 4)
mean1 = [2, 4]
cov1 = [[1, 0.5], [0.5, 1]]
X1 = np.random.multivariate_normal(mean1, cov1, 50)

# Class 2: Centered at (4, 2)
mean2 = [4, 2]
cov2 = [[1, 0.5], [0.5, 1]]
X2 = np.random.multivariate_normal(mean2, cov2, 50)

# Create figure
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(X1[:, 0], X1[:, 1], c='red', marker='x', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], c='blue', marker='o', label='Class 2')

# Good Projection (Fisher's LDA direction roughly perpendicular to the dividing line)
# The vector connecting means is (2, -2). A perp vector is (1, 1).
# Actually LDA direction w is proportional to S_w^-1 * (m1 - m2).
# Here visually, the line y = -x + 6 separates them.
# Projection onto y = x (direction [1, 1]) separates means well.
w_good = np.array([1, 1])
w_good = w_good / np.linalg.norm(w_good)
# Draw projection line passing through origin, but shifted for visual clarity
# line: p = t * w_good + offset
offset_good = np.array([0, 0])
ts = np.linspace(-3, 9, 100)
line_good = np.outer(ts, w_good) + offset_good
plt.plot(line_good[:, 0], line_good[:, 1], 'g-', linewidth=3, alpha=0.6, label='Good Projection Line (Subspace)')

# Project points onto Good Line
for p in X1:
    proj = np.dot(p, w_good) * w_good
    plt.plot([p[0], proj[0]], [p[1], proj[1]], 'r:', alpha=0.3)
for p in X2:
    proj = np.dot(p, w_good) * w_good
    plt.plot([p[0], proj[0]], [p[1], proj[1]], 'b:', alpha=0.3)

# Bad Projection (e.g., [1, -1], projecting onto y = -x)
# This will mix the classes because they are separated along [1, 1]
w_bad = np.array([1, -1])
w_bad = w_bad / np.linalg.norm(w_bad)
offset_bad = np.array([0, 6]) # shifted up
line_bad = np.outer(ts, w_bad) + offset_bad
plt.plot(line_bad[:, 0], line_bad[:, 1], 'k-', linewidth=1, alpha=0.4, label='Bad Projection Line')

# Project points onto Bad Line
for p in X1:
    proj = np.dot(p - offset_bad, w_bad) * w_bad + offset_bad
    plt.plot([p[0], proj[0]], [p[1], proj[1]], 'r:', alpha=0.1)
for p in X2:
    proj = np.dot(p - offset_bad, w_bad) * w_bad + offset_bad
    plt.plot([p[0], proj[0]], [p[1], proj[1]], 'b:', alpha=0.1)

plt.title('LDA Concept: Good vs Bad Projection')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Save
import os

# ... (data generation code) ...

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "ch04_lda_concept.png")

plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {output_path}")
