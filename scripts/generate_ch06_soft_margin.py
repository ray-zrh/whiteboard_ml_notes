import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import os

# Ensure the output directory exists
output_dir = "notes/chapters/assets"
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic data (linearly non-separable)
np.random.seed(42)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# Add some noise/overlap
X[18] = [1, 1]  # Class 0 point in Class 1 region
X[22] = [-1, -1] # Class 1 point in Class 0 region

# Fit the model with Soft Margin (C=1.0)
# C is the penalty parameter. Smaller C -> wider margin, more violations allowed.
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, Y)

# Create the figure
plt.figure(figsize=(8, 6))

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k', s=80, label='Data Points')

# Get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# Plot the parallel lines to the separating hyperplane for the margins
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

# Plot the lines
plt.plot(xx, yy, 'k-', label='Decision Boundary')
plt.plot(xx, yy_down, 'k--', label='Margin')
plt.plot(xx, yy_up, 'k--')

# Highlight support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=200,
            linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')

plt.title('Soft Margin SVM (C=1.0)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='lower right')
plt.axis('tight')
plt.grid(True, linestyle='--', alpha=0.6)

# Save the figure
output_path = os.path.join(output_dir, "ch06_soft_margin.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
