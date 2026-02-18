
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_importance_sampling_plot():
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
        "axes.linewidth": 0
    })

    # Data
    x = np.linspace(-4, 8, 500)

    # Target distribution p(z): mixture of two Gaussians (complex, hard to sample directly)
    p_z = 0.4 * stats.norm.pdf(x, loc=0, scale=1) + 0.6 * stats.norm.pdf(x, loc=4, scale=1)

    # Proposal distribution q(z): single Gaussian (easy to sample) covering p(z)
    q_z = stats.norm.pdf(x, loc=2, scale=2.5)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot distributions
    ax.plot(x, p_z, 'b-', lw=2.5, label=r'Target Distribution $p(z)$')
    ax.plot(x, q_z, 'r--', lw=2.5, label=r'Proposal Distribution $q(z)$')

    # Fill areas
    ax.fill_between(x, p_z, alpha=0.1, color='blue')
    ax.fill_between(x, q_z, alpha=0.1, color='red')

    # Draw samples and weights
    np.random.seed(42)
    samples = np.random.normal(loc=2, scale=2.5, size=15)
    # Filter samples to be within range for plot clarity
    samples = samples[(samples > -3) & (samples < 7)]
    samples = samples[:8] # Take fewer samples for clarity

    for idx, s in enumerate(samples):
        p_val = 0.4 * stats.norm.pdf(s, 0, 1) + 0.6 * stats.norm.pdf(s, 4, 1)
        q_val = stats.norm.pdf(s, 2, 2.5)
        weight = p_val / q_val

        # Draw sample line
        ax.vlines(s, 0, q_val, colors='k', linestyles=':', alpha=0.6)

        # Draw weight visualization (how much we scale the sample)
        # If weight > 1, circle is large (green); if weight < 1, circle is small (orange)
        color = 'green' if weight > 1 else 'orange'
        size = 100 * weight
        ax.scatter(s, 0, s=size, color=color, alpha=0.8, zorder=10, edgecolor='black', linewidth=1)

        # Annotate one example
        # Annotate one example (the first sample, which is around x=3.24)
        if idx == 0:
            # Place annotation in top-left empty space
            ax.annotate(r'Sample $z^{(i)} \sim q(z)$' + '\n' + r'Weight $w^{(i)} = \frac{p(z^{(i)})}{q(z^{(i)})}$',
                        xy=(s, 0.02), xytext=(-3.5, 0.3),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black', lw=1.5),
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'),
                        fontsize=12, zorder=20)
            # Remove separate text call as it's now merged into annotate for better positioning


    # Legends & Labels
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    ax.set_title("Importance Sampling Concept", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel("Probability Density")
    ax.set_ylim(0, 0.6)

    # Explanation (moved to Top-Left to avoid overlap with Legend at Top-Right)
    ax.text(-3.5, 0.55, "Green: High Weight\nOrange: Low Weight",
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    output_path = os.path.join(output_dir, 'ch16_importance_sampling.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_importance_sampling_plot()
