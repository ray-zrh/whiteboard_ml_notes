import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Data generation
np.random.seed(42)
x = np.linspace(1, 9, 8) # Fewer points for clearer visualization
true_w = 0.8
noise = np.random.randn(8) * 1.5
y = true_w * x + 2 + noise

# Fit line
w_hat = np.polyfit(x, y, 1)
f = np.poly1d(w_hat)
x_line = np.linspace(0, 10, 100)
y_line = f(x_line)

# Setup plot
fig, ax = plt.subplots(figsize=(10, 8))

# Remove frames
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Add explicit arrows for axes
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# Plot Data points
ax.scatter(x, y, marker='o', color='black', s=80, label='Data $y_i$', zorder=5)

# Plot Regression Line
ax.plot(x_line, y_line, color='#e74c3c', linewidth=2.5, label=r'Model $f(x) = w^T x$')

# Draw Residuals and Squares
for i in range(len(x)):
    xi = x[i]
    yi = y[i]
    y_hat_i = f(xi)
    resid = yi - y_hat_i

    # 1. Draw Residual Line (Green Dashed)
    ax.plot([xi, xi], [yi, y_hat_i], color='#2ecc71', linestyle='--', linewidth=1.5)

    # 2. Draw Square representing (residual)^2
    # Determine direction of square (left or right) to avoid overlap if possible
    # Default to right
    direction = 1
    if i % 2 == 0: direction = -1 # Alternate sides

    # Square corners
    # Bottom-left (or Bottom-right) corresponds to min(yi, y_hat_i)
    base_y = min(yi, y_hat_i)
    height = abs(resid)

    # Create Rectangle patch
    # xy is bottom-left corner
    fixed_x = xi if direction == 1 else xi - height

    rect = Rectangle((fixed_x, base_y), height, height,
                     linewidth=0, edgecolor='none', facecolor='#2ecc71', alpha=0.2)
    ax.add_patch(rect)


# Annotate a representative point (e.g., index 2)
idx = 2
xi = x[idx]
yi = y[idx]
y_hat_i = f(xi)
resid = yi - y_hat_i

# Label x_i
ax.text(xi, -0.8, r'$x_i$', ha='center', fontsize=12)
# Label y_i
ax.text(xi + 0.2, yi, r'$y_i$', va='center', fontsize=12, fontweight='bold')
# Label residual e_i (put text next to the line)
ax.text(xi - 0.2, (yi + y_hat_i)/2, r'$e_i$', color='#2ecc71', ha='right', va='center', fontsize=12, fontweight='bold')
# Label square area
ax.text(xi + (0.5 if i%2!=0 else -0.5)*abs(resid), (yi + y_hat_i)/2, r'$e_i^2$', color='#2ecc71', ha='center', va='center', fontsize=10, alpha=0.8)


# Equation Label
label_x = 9.0
label_y = f(label_x)
ax.text(label_x, label_y - 1.0, r'$f(w) = w^T x$', fontsize=14, color='#e74c3c', ha='center')

# Title
ax.set_title('Geometric Interpretation: Minimizing Sum of Squared Residuals', fontsize=14, pad=20)

# Adjust limits
ax.set_xlim(-1, 12)
ax.set_ylim(-2, 14)
ax.set_xticks([])
ax.set_yticks([])

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='Observed Data $y$', markerfacecolor='black', markersize=8, linestyle='None'),
    Line2D([0], [0], color='#e74c3c', lw=2.5, label='Regression Line $f(w)$'),
    Line2D([0], [0], color='#2ecc71', lw=1.5, linestyle='--', label=r'Residual $e = y - \hat{y}$'),
    Patch(facecolor='#2ecc71', alpha=0.2, label='Squared Error $e^2$'),
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=11)

plt.tight_layout()
output_path = "notes/chapters/assets/ch03_linear_regression_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Image saved to {output_path}")
