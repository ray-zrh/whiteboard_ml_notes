import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
import os

# Ensure the output directory exists
output_dir = "notes/chapters/assets"
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Compactness (Convex clusters) ---
# Two concentric circles: K-means / GMM will fail here
X_circles, y_circles = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

ax1 = axes[0]
ax1.scatter(X_circles[y_circles == 0, 0], X_circles[y_circles == 0, 1],
            c='#E74C3C', marker='o', s=30, edgecolors='k', linewidths=0.5,
            alpha=0.7, label='Cluster A')
ax1.scatter(X_circles[y_circles == 1, 0], X_circles[y_circles == 1, 1],
            c='#3498DB', marker='s', s=30, edgecolors='k', linewidths=0.5,
            alpha=0.7, label='Cluster B')
ax1.set_title("Compactness (e.g., K-means, GMM)", fontsize=14, fontweight='bold')
ax1.set_xlabel("$x_1$", fontsize=12)
ax1.set_ylabel("$x_2$", fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.text(0, 0, 'K-means fails\non non-convex shapes',
         ha='center', va='center', fontsize=9, color='gray',
         style='italic', alpha=0.8)

# --- Plot 2: Connectivity (Non-convex clusters) ---
# Two interleaving half-moons: spectral clustering handles this well
X_moons, y_moons = make_moons(n_samples=300, noise=0.06, random_state=42)

ax2 = axes[1]
ax2.scatter(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1],
            c='#E74C3C', marker='o', s=30, edgecolors='k', linewidths=0.5,
            alpha=0.7, label='Cluster A')
ax2.scatter(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1],
            c='#3498DB', marker='s', s=30, edgecolors='k', linewidths=0.5,
            alpha=0.7, label='Cluster B')
ax2.set_title("Connectivity (e.g., Spectral Clustering)", fontsize=14, fontweight='bold')
ax2.set_xlabel("$x_1$", fontsize=12)
ax2.set_ylabel("$x_2$", fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.text(0.5, 0.25, 'Spectral clustering\nhandles arbitrary shapes',
         ha='center', va='center', fontsize=9, color='gray',
         style='italic', alpha=0.8)

plt.tight_layout()

# Save the figure
output_path = os.path.join(output_dir, "ch22_clustering_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
