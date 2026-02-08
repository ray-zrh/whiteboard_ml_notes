
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines

# Set up the figure
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Parameters
mu = np.array([2.0, 2.0])
lambda1 = 2.0
lambda2 = 0.5
theta_deg = 30
theta_rad = np.radians(theta_deg)

# Rotation matrix (Eigenvectors)
c, s = np.cos(theta_rad), np.sin(theta_rad)
R = np.array([[c, -s], [s, c]])
u1 = R[:, 0]  # Major axis direction
u2 = R[:, 1]  # Minor axis direction

# Draw Ellipse (Delta^2 = 1)
# Width/Height for Ellipse are full lengths (diameters), so 2 * semi-axis
width = 2 * np.sqrt(lambda1)
height = 2 * np.sqrt(lambda2)

ellipse = Ellipse(xy=mu, width=width, height=height, angle=theta_deg,
                  edgecolor='black', facecolor='#F8F8F8', alpha=0.3, linewidth=2, label='Ellipse $\Delta^2 = 1$')
ax.add_patch(ellipse)

# Draw Principal Axes (Infinite lines)
scale = 4.0
# Major Axis Line
ax.plot([mu[0] - u1[0]*scale, mu[0] + u1[0]*scale],
        [mu[1] - u1[1]*scale, mu[1] + u1[1]*scale], '--', color='gray', alpha=0.4, linewidth=1)
# Minor Axis Line
ax.plot([mu[0] - u2[0]*scale, mu[0] + u2[0]*scale],
        [mu[1] - u2[1]*scale, mu[1] + u2[1]*scale], '--', color='gray', alpha=0.4, linewidth=1)

# Draw Unit Vectors u1 and u2 (Basis) at origin
vec_scale = 1.0
plt.arrow(mu[0], mu[1], u1[0]*vec_scale, u1[1]*vec_scale, head_width=0.1, head_length=0.1, fc='blue', ec='blue', width=0.02, zorder=5)
plt.arrow(mu[0], mu[1], u2[0]*vec_scale, u2[1]*vec_scale, head_width=0.1, head_length=0.1, fc='blue', ec='blue', width=0.02, zorder=5)

plt.text(mu[0] + u1[0]*1.1, mu[1] + u1[1]*1.1, '$u_1$', fontsize=12, color='blue', fontweight='bold')
plt.text(mu[0] + u2[0]*1.1, mu[1] + u2[1]*1.1, '$u_2$', fontsize=12, color='blue', fontweight='bold')

# Center point
plt.scatter(*mu, color='black', zorder=10, label='Mean $\mu$')
plt.text(mu[0]-0.3, mu[1]-0.15, '$\mu$', fontsize=12)

# Pick a point x on the ellipse
# Use phi = 50 degrees (Quadrant I relative to axes)
phi = np.radians(50)
y1_val = np.sqrt(lambda1) * np.cos(phi)
y2_val = np.sqrt(lambda2) * np.sin(phi)
x_point = mu + y1_val * u1 + y2_val * u2

plt.scatter(*x_point, color='red', zorder=10, label='Sample $x$')
plt.text(x_point[0]+0.1, x_point[1]+0.1, '$x$', fontsize=12, color='red', fontweight='bold')

# Draw projections
proj_u1 = mu + y1_val * u1
proj_u2 = mu + y2_val * u2

# Dotted lines from x to axes
plt.plot([x_point[0], proj_u1[0]], [x_point[1], proj_u1[1]], ':', color='red', alpha=0.7)
plt.plot([x_point[0], proj_u2[0]], [x_point[1], proj_u2[1]], ':', color='red', alpha=0.7)

# Annotate y1 (coordinate length along u1 axis)
# Use an arrow <-> to show the length from mu to proj_u1
plt.annotate('', xy=mu, xytext=proj_u1, arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
# Label y1
plt.text((mu[0] + proj_u1[0])/2 + 0.15*u2[0], (mu[1] + proj_u1[1])/2 + 0.15*u2[1], '$y_1$', color='red', fontsize=12, fontweight='bold')

# Annotate y2 (coordinate length along u2 axis)
# Use an arrow <-> to show the length from mu to proj_u2
plt.annotate('', xy=mu, xytext=proj_u2, arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
# Label y2
plt.text((mu[0] + proj_u2[0])/2 + 0.15*u1[0], (mu[1] + proj_u2[1])/2 + 0.15*u1[1], '$y_2$', color='red', fontsize=12, fontweight='bold')


# Semi-axes annotations (sqrt(lambda))

# Major semi-axis (Negative u1 direction) - Keep this on negative side
v1_neg = mu - np.sqrt(lambda1) * u1
# Offset dimension line "below" (negative u2)
offset_vec = -0.6 * u2
start = mu + offset_vec
end = v1_neg + offset_vec

# Draw dimension line <--->
plt.annotate('', xy=start, xytext=end, arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
# Dotted extension lines
plt.plot([mu[0], start[0]], [mu[1], start[1]], ':', color='gray', alpha=0.5)
plt.plot([v1_neg[0], end[0]], [v1_neg[1], end[1]], ':', color='gray', alpha=0.5)
# Label
plt.text((start[0]+end[0])/2 - 0.1, (start[1]+end[1])/2 - 0.3, '$\sqrt{\lambda_1}$', color='green', fontsize=14, ha='center')

# Minor semi-axis (POSITIVE u2 direction) - Moved as requested
v2_pos = mu + np.sqrt(lambda2) * u2
# Offset dimension line "left" (negative u1) to avoid overlap with x (which is in +u1 direction)
offset_vec_2 = -0.6 * u1
start_2 = mu + offset_vec_2
end_2 = v2_pos + offset_vec_2

# Draw dimension line <--->
plt.annotate('', xy=start_2, xytext=end_2, arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
# Dotted extension lines
plt.plot([mu[0], start_2[0]], [mu[1], start_2[1]], ':', color='gray', alpha=0.5)
plt.plot([v2_pos[0], end_2[0]], [v2_pos[1], end_2[1]], ':', color='gray', alpha=0.5)
# Label
plt.text((start_2[0]+end_2[0])/2 - 0.5, (start_2[1]+end_2[1])/2 + 0.1, '$\sqrt{\lambda_2}$', color='purple', fontsize=14, ha='right')


# Setup limits with extra space for legend
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Mahalanobis Coordinates ($y_i$) vs Axes Lengths ($\sqrt{\lambda_i}$)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Custom legend
legend_elements = [
    mlines.Line2D([], [], color='blue', marker='>', markersize=10, linestyle='-', label='Eigenvectors $u_1, u_2$'),
    mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='Sample $x$'),
    mlines.Line2D([], [], color='red', linestyle='-', linewidth=2, label='Coordinates $y_1, y_2$'),
    mlines.Line2D([], [], color='green', linestyle='-', label='Semi-major axis $\sqrt{\lambda_1}$'),
    mlines.Line2D([], [], color='purple', linestyle='-', label='Semi-minor axis $\sqrt{\lambda_2}$'),
    mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, alpha=0.5, label='Ellipse $\Delta^2 = 1$')
]
plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

output_path = 'notes/chapters/assets/ch02_mahalanobis_geometry.png'
plt.savefig(output_path, dpi=300)
print(f"Image saved to {output_path}")
