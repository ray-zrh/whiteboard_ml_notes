import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define subspace vectors (feature vectors)
# Use a cleaner "floor" projection (xy-planeish)
# User wants x1 and x2 to be orthogonal (vertical to each other)
# Let's align them with X and Y axes for maximum clarity
x1 = np.array([2.5, 0, 0])
x2 = np.array([0, 2.5, 0])

# Define Y vector
# Y = y_hat + resid
# y_hat is in the span of x1, x2
beta = np.array([0.6, 0.5]) # Coefficients relative to x1, x2 length
y_hat = beta[0] * x1 + beta[1] * x2

# Residual vector - perpendicular to the floor (z-axis)
resid = np.array([0, 0, 2.0])
y = y_hat + resid

# Origin
o = np.array([0, 0, 0])

# Plot the vectors
# Feature x1
ax.quiver(0, 0, 0, x1[0], x1[1], x1[2], color='blue', alpha=0.4, arrow_length_ratio=0.08, linewidth=1.5)
ax.text(x1[0], x1[1]-0.2, x1[2], r'$x_1$', color='blue', fontsize=12, ha='center')

# Feature x2
ax.quiver(0, 0, 0, x2[0], x2[1], x2[2], color='blue', alpha=0.4, arrow_length_ratio=0.08, linewidth=1.5)
ax.text(x2[0]-0.2, x2[1], x2[2], r'$x_2$', color='blue', fontsize=12, va='center')

# Prediction y_hat
ax.quiver(0, 0, 0, y_hat[0], y_hat[1], y_hat[2], color='red', alpha=1.0, arrow_length_ratio=0.08, linewidth=2.5)
# Label y_hat slightly offset to avoid clutter
ax.text(y_hat[0] + 0.2, y_hat[1], y_hat[2], r'$\hat{Y}$', color='red', fontsize=15, fontweight='bold')

# Observed Y
ax.quiver(0, 0, 0, y[0], y[1], y[2], color='black', arrow_length_ratio=0.08, linewidth=2.5)
ax.text(y[0], y[1], y[2]+0.2, r'$Y$', color='black', fontsize=15, fontweight='bold')

# Residual Vector (from y_hat to Y)
ax.plot([y_hat[0], y[0]], [y_hat[1], y[1]], [y_hat[2], y[2]], 'g--', linewidth=1.5)
# Label e in the middle of the dashed line
ax.text(y_hat[0] - 0.2, y_hat[1], y_hat[2] + resid[2]/2, r'$e$', color='green', fontsize=14, ha='right', va='center')

# Draw the plane (Subspace) - The XY plane
xx, yy = np.meshgrid(np.linspace(-0.5, 3.0, 10), np.linspace(-0.5, 3.0, 10))
z = np.zeros_like(xx) # Plane is z=0 since x1, x2 are in z=0

ax.plot_surface(xx, yy, z, alpha=0.1, color='gray')
ax.text(3.0, 0, 0, r'Span$(X)$', fontsize=12, color='gray')

# Draw perpendicular symbol between x1 and x2 at origin (optional, but emphasizes orthogonality)
# Small square at origin
sq_len = 0.2
ax.plot([sq_len, sq_len, 0], [0, sq_len, sq_len], [0, 0, 0], 'k-', linewidth=0.5, alpha=0.3)

# Add right angle symbol for projection
# At y_hat, looking up to Y
# Vectors defining the corner: unit vector towards -x1 and -x2 (just some direction in plane) ?
# actually just pick a direction in the plane. -y_hat is fine.
v_plane = -y_hat / np.linalg.norm(y_hat) * 0.3
v_up = resid / np.linalg.norm(resid) * 0.3
p = y_hat

ax.plot([p[0] + v_plane[0], p[0] + v_plane[0] + v_up[0]],
        [p[1] + v_plane[1], p[1] + v_plane[1] + v_up[1]],
        [p[2] + v_plane[2], p[2] + v_plane[2] + v_up[2]], 'k-', linewidth=0.8)
ax.plot([p[0] + v_up[0], p[0] + v_plane[0] + v_up[0]],
        [p[1] + v_up[1], p[1] + v_plane[1] + v_up[1]],
        [p[2] + v_up[2], p[2] + v_plane[2] + v_up[2]], 'k-', linewidth=0.8)


# View settings
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_zlim(0, 2.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.axis('off') # Turn off all axis stuff for a cleaner "whiteboard" look

# Adjust view angle for "easier to understand" perspective
ax.view_init(elev=25, azim=45)

plt.tight_layout()
output_path = "notes/chapters/assets/ch03_least_squares_projection.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Image saved to {output_path}")
