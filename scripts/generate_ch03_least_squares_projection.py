import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def draw_vector(ax, v, origin=np.array([0, 0, 0]), color='b', label=None, arrow_length_ratio=0.1, linewidth=2):
    ax.quiver(origin[0], origin[1], origin[2],
              v[0], v[1], v[2],
              color=color, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth,
              label=label)

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinite
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Define subspace vectors (feature vectors)
# Make them orthogonal for simplicity in visualization, though not required mathematically
x1 = np.array([3.0, 0, 0])
x2 = np.array([0, 3.0, 0])

# Define Y vector
# Y = y_hat + resid
beta = np.array([0.7, 0.5])
y_hat = beta[0] * x1 + beta[1] * x2
resid = np.array([0, 0, 2.5]) # Perpendicular to the plane (z-axis)
y = y_hat + resid

origin = np.array([0, 0, 0])

# --- Draw the Subspace Plane (Span of X) ---
# Create a grid for the plane
xx, yy = np.meshgrid(np.linspace(-1, 4, 10), np.linspace(-1, 4, 10))
z = np.zeros_like(xx)

# Plot the surface
ax.plot_surface(xx, yy, z, alpha=0.1, color='#add8e6', rstride=100, cstride=100, edgecolor='none')
# Add a grid on the floor manually for better look
for i in np.linspace(-1, 4, 6):
    ax.plot([i, i], [-1, 4], [0, 0], color='gray', alpha=0.2, linewidth=0.5)
    ax.plot([-1, 4], [i, i], [0, 0], color='gray', alpha=0.2, linewidth=0.5)

ax.text(3.5, 3.5, 0, r'$\mathcal{S} = \text{span}(X)$', fontsize=14, color='gray')

# --- Draw Vectors ---

# Feature vectors x1, x2
draw_vector(ax, x1, color='blue', label='$x_1$', linewidth=1.5, arrow_length_ratio=0.05)
ax.text(x1[0], x1[1]-0.3, x1[2], r'$x_1$', color='blue', fontsize=12, fontweight='bold')

draw_vector(ax, x2, color='blue', label='$x_2$', linewidth=1.5, arrow_length_ratio=0.05)
ax.text(x2[0]-0.3, x2[1], x2[2], r'$x_2$', color='blue', fontsize=12, fontweight='bold')

# Prediction y_hat
draw_vector(ax, y_hat, color='#e74c3c', label=r'$\hat{Y} = X\hat{w}$', linewidth=3, arrow_length_ratio=0.08)
ax.text(y_hat[0] + 0.2, y_hat[1], y_hat[2], r'$\hat{Y}$', color='#e74c3c', fontsize=16, fontweight='bold')

# Observed Y
draw_vector(ax, y, color='black', label='$Y$', linewidth=3, arrow_length_ratio=0.05)
ax.text(y[0], y[1], y[2]+0.2, r'$Y$', color='black', fontsize=16, fontweight='bold')

# Residual e (Dashed line)
ax.plot([y_hat[0], y[0]], [y_hat[1], y[1]], [y_hat[2], y[2]],
        color='#2ecc71', linestyle='--', linewidth=2, label=r'$e = Y - \hat{Y}$')
ax.text(y_hat[0], y_hat[1], y_hat[2] + resid[2]/2, r'$e$', color='#2ecc71', fontsize=14, ha='right', va='center', fontweight='bold')


# --- Annotations & Details ---

# Orthogonality Symbol (Right Angle)
# Between e (vertical) and the plane (horizontal line from origin to y_hat)
# We can draw little square at y_hat
len_sq = 0.3
# Vector along y_hat
v_plane = -y_hat / np.linalg.norm(y_hat) * len_sq
# Vector up along resid
v_up = resid / np.linalg.norm(resid) * len_sq

p = y_hat
# Draw the square corner
corner_x = [p[0] + v_up[0], p[0] + v_up[0] + v_plane[0], p[0] + v_plane[0]]
corner_y = [p[1] + v_up[1], p[1] + v_up[1] + v_plane[1], p[1] + v_plane[1]]
corner_z = [p[2] + v_up[2], p[2] + v_up[2] + v_plane[2], p[2] + v_plane[2]]
ax.plot(corner_x, corner_y, corner_z, color='black', linewidth=1)


# Plot Origin
ax.scatter([0], [0], [0], color='black', s=20)
ax.text(0, -0.3, 0, r'$0$', fontsize=12)

# --- View & Layout ---

# Remove axes ticks for clean look
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.axis('off')

# Set equal aspect ratio
set_axes_equal(ax)

# Adjust view angle
ax.view_init(elev=20, azim=20)

# Add Legend
# Create dummy artists for legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', lw=3, label='Observed Data $Y$'),
    Line2D([0], [0], color='#e74c3c', lw=3, label=r'Prediction $\hat{Y}$ (Projection)'),
    Line2D([0], [0], color='#2ecc71', lw=2, linestyle='--', label=r'Residual $e$ (Error)'),
    Line2D([0], [0], color='blue', lw=1.5, label=r'Basis Vectors $x_1, x_2$'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#add8e6', markersize=10, label=r'Column Space $\mathcal{S}$'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)

plt.tight_layout()
output_path = "notes/chapters/assets/ch03_least_squares_projection.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Image saved to {output_path}")
