
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_ppca_graphical_model():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Nodes
    # z (latent)
    z_circle = patches.Circle((3, 2.5), 0.6, fill=False, ec='black', lw=2)
    ax.add_patch(z_circle)
    ax.text(3, 2.5, '$z$', ha='center', va='center', fontsize=16)
    ax.text(3, 1.5, 'Latent\n$N(0, I)$', ha='center', va='center', fontsize=10, color='gray')

    # x (observed)
    x_circle = patches.Circle((7, 2.5), 0.6, fill=True, fc='lightgray', ec='black', lw=2)
    ax.add_patch(x_circle)
    ax.text(7, 2.5, '$x$', ha='center', va='center', fontsize=16)
    ax.text(7, 1.5, 'Observed\n$N(\mu, C)$', ha='center', va='center', fontsize=10, color='gray')

    # Parameters
    # W, mu, sigma
    ax.text(5, 3.5, '$W, \mu, \sigma^2$', ha='center', va='center', fontsize=12)

    # Arrow
    ax.arrow(3.7, 2.5, 2.6, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')

    # Plate notation (optional, strictly P-PCA usually applies to single point then i.i.d)
    # Draw a rectangle for N data points
    rect = patches.Rectangle((1, 0.5), 8, 4, fill=False, ec='gray', linestyle='--', lw=1)
    ax.add_patch(rect)
    ax.text(8.5, 0.8, '$N$', fontsize=12)

    plt.title("Probabilistic PCA: Linear Gaussian Model", fontsize=14)
    plt.tight_layout()

    output_path = "notes/chapters/assets/ch05_ppca_pgm.png"
    plt.savefig(output_path)
    print(f"Saved PGM to {output_path}")

if __name__ == "__main__":
    plot_ppca_graphical_model()
