import numpy as np
import matplotlib.pyplot as plt
import os

# Create assets directory if it doesn't exist
os.makedirs('notes/chapters/assets', exist_ok=True)

# Generate synthetic data
x = np.linspace(-5, 15, 1000)

# Define P_data (three sharp peaks to represent empirical data points)
def p_data(x):
    return 0.4 * np.exp(-((x - 0) ** 2) / 0.5) + \
           0.3 * np.exp(-((x - 6) ** 2) / 0.5) + \
           0.3 * np.exp(-((x - 10) ** 2) / 0.5)

# Define P_model (smoother, wider distribution currently attempting to fit P_data)
def p_model(x):
    return 0.5 * np.exp(-((x - 2) ** 2) / 8) + \
           0.5 * np.exp(-((x - 8) ** 2) / 8)

pd = p_data(x)
pm = p_model(x)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the distributions
ax.plot(x, pd, color='green', label='$P_{data}$', linewidth=2)
ax.plot(x, pm, color='blue', label='$P_{model}$', linewidth=2)

# Sample some "data particles" for the positive phase (at the peaks of P_data)
data_particles = [0, 6, 10]
for dp in data_particles:
    # Add crosses at the bottom for data particles
    ax.scatter(dp, 0, marker='x', color='black', s=50, zorder=5)

    # Calculate height to draw arrow
    y_val = p_model(dp)

    # Positive phase arrows (pushing P_model up where data is)
    ax.annotate('', xy=(dp, y_val + 0.1), xytext=(dp, y_val),
                arrowprops=dict(arrowstyle="->", color='red', lw=2))

# Sample some "fantasy particles" from P_model where it shouldn't be high (negative phase)
# Let's pick points where P_model is high but P_data is low
fantasy_particles = [3, 4, 8]
for fp in fantasy_particles:
    # Add crosses at the bottom for fantasy particles
    ax.scatter(fp, 0, marker='x', color='black', s=50, zorder=5)

    # Calculate height to draw arrow
    y_val = p_model(fp)

    # Negative phase arrows (pushing P_model down where data isn't)
    ax.annotate('', xy=(fp, y_val - 0.1), xytext=(fp, y_val),
                arrowprops=dict(arrowstyle="->", color='black', lw=2))

# Add a text label pointing to fantasy particles
ax.annotate('fantasy particles', xy=(8, 0.02), xytext=(8.5, 0.2),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red'),
            fontsize=12, color='red')

# Add labels and legend
ax.set_title('Stochastic Maximum Likelihood: Positive vs Negative Phase', fontsize=16)
ax.legend(fontsize=14)
ax.set_ylim(-0.05, max(pd.max(), pm.max()) + 0.1)

# Remove y-axis and top/right spines for a cleaner "whiteboard" look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_ticks([])

# Save the plot
plt.tight_layout()
plt.savefig('notes/chapters/assets/ch24_sml_distribution.png', dpi=300)
print('Saved visualization to notes/chapters/assets/ch24_sml_distribution.png')
