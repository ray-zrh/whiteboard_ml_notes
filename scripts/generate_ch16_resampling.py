
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_resampling_cdf_diagram():
    # Style settings
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 1.5
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    # Particle Weights
    weights = np.array([0.1, 0.2, 0.7])
    cdf = np.cumsum(weights)
    cdf = np.insert(cdf, 0, 0) # Start from 0

    particle_indices = [1, 2, 3]
    particle_labels = [r"$z^{(1)}$", r"$z^{(2)}$", r"$z^{(3)}$"]

    # Draw CDF
    # X-axis: Particle Index (0 to 3)
    # Y-axis: Cumulative Weight (0 to 1)

    # Increase X-limit to make room for annotations on the right
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.1, 1.2)

    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(particle_labels, fontsize=16)
    ax.set_yticks([0, 0.1, 0.3, 1.0])
    ax.set_ylabel("Cumulative Weight (CDF)", fontsize=14)
    ax.set_xlabel("Particles Index", fontsize=14)

    # Plot stairs
    ax.step(range(4), cdf, where='post', color='blue', lw=3, label='CDF')

    # Fill areas under curve to represent weight blocks
    colors = ['#ffcc99', '#99ccff', '#99ff99']
    for i in range(3):
        ax.fill_between([i, i+1], [cdf[i], cdf[i]], [cdf[i+1], cdf[i+1]], color=colors[i], alpha=0.3)
        # Label weight value
        mid_y = (cdf[i] + cdf[i+1]) / 2
        ax.text(i + 0.5, mid_y, f"w={weights[i]:.1f}", ha='center', va='center', fontsize=12, fontweight='bold', color='#444444')

    # Random Samples u ~ U(0,1)
    u_samples = [0.05, 0.6, 0.9]
    u_labels = [r"$u_1 \approx 0.05$", r"$u_2 \approx 0.6$", r"$u_3 \approx 0.9$"]

    for i, u in enumerate(u_samples):
        # Determine which particle it hits
        hit_idx = np.searchsorted(cdf, u) - 1

        # Draw horizontal line from u
        ax.hlines(u, -0.5, hit_idx + 1, colors='red', linestyles=':', lw=1.5)
        # Draw vertical line down
        ax.vlines(hit_idx + 1, 0, u, colors='red', linestyles=':', lw=1.5)

        # Dot on Y-axis
        ax.plot(-0.5, u, 'ro', markersize=6)
        ax.text(-0.6, u, u_labels[i], ha='right', va='center', color='#d62728', fontsize=12)

        # Dot on Stair
        ax.plot(hit_idx + 1, u, 'ro', markersize=6)

        # Result annotation on the right side
        # Use annotation with arrow to point to the intersection, text further right
        ax.annotate(r"Select " + particle_labels[hit_idx],
                    xy=(hit_idx + 1, u),
                    xytext=(3.5, u),
                    arrowprops=dict(arrowstyle="->", color='black', shrinkA=5, shrinkB=5),
                    color='black', fontsize=12, va='center', backgroundcolor='white')

    # Title
    ax.set_title("Resampling via Inverse Transform Sampling (CDF Method)", fontsize=18, fontweight='bold', pad=20)

    # Explanation
    # Move to top left, ensure it fits
    ax.text(0.1, 1.15, r"Note: Particle $z^{(3)}$ spans 0.7 of the y-axis" + "\n" + r"$\rightarrow$ High probability of being selected (Duplication)",
            ha='left', va='top', fontsize=12, color='#333333',
            bbox=dict(facecolor='#f9f9f9', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.5'))

    output_path = os.path.join(output_dir, 'ch16_resampling_cdf.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_resampling_cdf_diagram()
