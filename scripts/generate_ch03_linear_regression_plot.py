import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.xkcd()  # Use xkcd style for a hand-drawn look, or just standard with clean lines.
# actually, for a "notes" style, a clean plot is usually better than xkcd, but the user gave a sketch.
# Let's stick to a clean professional style but with the specific annotations requested.
plt.rcdefaults()
plt.style.use('seaborn-v0_8-whitegrid')

# Data generation
np.random.seed(42)
x = np.linspace(1, 9, 10)
true_w = 0.8
noise = np.random.randn(10) * 1.5
y = true_w * x + 2 + noise

# Fit line (just for drawing a nice line)
w_hat = np.polyfit(x, y, 1)
f = np.poly1d(w_hat)
x_line = np.linspace(0, 10, 100)
y_line = f(x_line)

# Setup plot
fig, ax = plt.subplots(figsize=(8, 6))

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Move left and bottom spines to zero position
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
# Add arrows to spines (hacky way in matplotlib)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# Plot data points
ax.scatter(x, y, marker='x', color='black', s=80, label='Data points')

# Plot regression line
ax.plot(x_line, y_line, 'k-', linewidth=2, label=r'$f(w) = w^T x$')

# Highlight a specific point (e.g., index 3)
idx = 3
x_i = x[idx]
y_i = y[idx]
y_hat_i = f(x_i)

# Draw residual line
ax.plot([x_i, x_i], [y_i, y_hat_i], 'k-', linewidth=1.5)

# Dashed lines to axes
# To x-axis
ax.plot([x_i, x_i], [0, y_hat_i], 'k--', linewidth=1, alpha=0.6)
# To y-axis (true y)
ax.plot([0, x_i], [y_i, y_i], 'k--', linewidth=1, alpha=0.6)
# To y-axis (predicted y)
ax.plot([0, x_i], [y_hat_i, y_hat_i], 'k--', linewidth=1, alpha=0.6)

# Labels
# x_i on x-axis
ax.text(x_i, -1, r'$x_i$', ha='center', fontsize=12)
# y_i on y-axis
ax.text(-0.8, y_i, r'$y_i$', va='center', fontsize=12)
# y_hat on y-axis
ax.text(-1.5, y_hat_i, r'$w^T x_i$', va='center', fontsize=12)
# Function label - aligned with the line
# Calculate angle for rotation (handling aspect ratio roughly)
# Data aspect ratio: y_range/x_range = 14/12 ~ 1.16
# Figure aspect ratio: 6/8 = 0.75
# This is an approximation. For exact rotation, one would transform points to display coords.
# But for this simple plot, manual tuning or a helper is fine.
p1 = ax.transData.transform((0, f(0)))
p2 = ax.transData.transform((10, f(10)))
dy = p2[1] - p1[1]
dx = p2[0] - p1[0]
angle = np.degrees(np.arctan2(dy, dx))

# Position the text on the line, slightly offset
# User requested parallel to x-axis (rotation=0) and no overlap.
# Putting it at x=9.5 (data ends at 9.0) ensures no overlap with data points.
label_x = 9.5
label_y = f(label_x)
# Add a small vertical offset in data coordinates? No, better to use verticalalignment='top' to put it below
# We also add a small negative offset to label_y just to clear the line thickness
ax.text(label_x, label_y - 0.5, r'$f(w) = w^T x$', fontsize=14, rotation=0,
        ha='center', va='top', color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

# Residual arrows/lines for other points to mimic the sketch
for i in range(len(x)):
    if i == idx: continue
    ax.plot([x[i], x[i]], [y[i], f(x[i])], 'k-', linewidth=0.8, alpha=0.7)

# Adjust limits
ax.set_xlim(-1, 11)
ax.set_ylim(-2, 12)
ax.set_xticks([])
ax.set_yticks([])

# Save
output_path = "notes/chapters/assets/ch03_linear_regression_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Image saved to {output_path}")
