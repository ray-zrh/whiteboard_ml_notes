import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import os

# Ensure the output directory exists
output_dir = "notes/chapters/assets"
os.makedirs(output_dir, exist_ok=True)

def plot_decision_boundary(X, y, model, ax, title):
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict class for each point in mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Put the result into a color plot
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=50)

    # Highlight support vectors
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')

    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(loc='upper right')

# Generate non-linear data (concentric circles)
# factor=0.5 means inner circle is half size of outer
# noise=0.1 adds some jitter
X, y = datasets.make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Linear Kernel (Fails on this data)
clf_linear = svm.SVC(kernel='linear', C=1.0)
clf_linear.fit(X, y)
plot_decision_boundary(X, y, clf_linear, ax1, "Linear Kernel (Underfitting)")

# 2. RBF Kernel (Succeeds)
# Gamma controls the shape of the decision boundary
# C controls regularization
clf_rbf = svm.SVC(kernel='rbf', C=1.0, gamma=1.0)
clf_rbf.fit(X, y)
plot_decision_boundary(X, y, clf_rbf, ax2, "RBF Kernel (Gaussian)\nMapping to Infinite Dim Space")

plt.tight_layout()

# Save the figure
output_path = os.path.join(output_dir, "ch06_kernel_svm.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
