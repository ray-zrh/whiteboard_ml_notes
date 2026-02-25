import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
from matplotlib.patches import FancyArrowPatch
import os

def draw_vae_pgm():
    """Draws the VAE Encoder-Decoder Probabilistic Graphical Model."""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Style constants
    node_radius = 0.5
    node_lw = 2.0
    font_size = 20
    text_color = '#333333'
    node_edge = '#444466'  # dark bluish
    node_face = '#ffffff'

    # Encoder / Decoder colors (like the sketch)
    decoder_color = '#2c5985'  # Blue-ish for p_theta
    encoder_color = '#b04a4a'  # Red-ish for q_phi

    pos_z = (3.0, 4.5)
    pos_x = (3.0, 1.5)

    def add_node(pos, label):
        circle = patches.Circle(pos, node_radius,
                                edgecolor=node_edge, facecolor=node_face,
                                linewidth=node_lw, zorder=10)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label,
                ha='center', va='center', fontsize=font_size+4,
                color=text_color, fontweight='bold', zorder=11)
        return circle

    # 1. Nodes
    add_node(pos_z, r'$Z$')
    add_node(pos_x, r'$X$')

    # 2. Edges
    # Decoder: Z -> X
    arrow_dec = FancyArrowPatch((pos_z[0]+0.1, pos_z[1]-node_radius),
                                (pos_x[0]+0.1, pos_x[1]+node_radius),
                                connectionstyle="arc3,rad=-0.3",
                                arrowstyle="->,head_width=0.4,head_length=0.5",
                                color=decoder_color, lw=2.5, zorder=5)
    ax.add_patch(arrow_dec)

    # Label for Decoder
    ax.text(pos_z[0] + 0.8, (pos_z[1]+pos_x[1])/2 + 0.3, r'$p_\theta(x|Z)$',
            fontsize=font_size, color=decoder_color, ha='left', va='center')
    ax.text(pos_z[0] + 0.8, (pos_z[1]+pos_x[1])/2 - 0.3, r'Decoder',
            fontsize=font_size-2, color=decoder_color, ha='left', va='center')

    # Encoder: X -> Z
    arrow_enc = FancyArrowPatch((pos_x[0]-0.1, pos_x[1]+node_radius),
                                (pos_z[0]-0.1, pos_z[1]-node_radius),
                                connectionstyle="arc3,rad=-0.3",
                                arrowstyle="->,head_width=0.4,head_length=0.5",
                                color=encoder_color, lw=2.0, linestyle='--', zorder=5)
    ax.add_patch(arrow_enc)

    # Label for Encoder
    ax.text(pos_z[0] - 0.8, (pos_z[1]+pos_x[1])/2, r'$q_\phi(Z|x)$',
            fontsize=font_size, color=encoder_color, ha='right', va='center')

    # Optional Title
    # ax.set_title("Variational Autoencoder (VAE)", fontsize=18, pad=20)

    # Optimize layout
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '../notes/chapters/assets')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'ch32_vae_pgm.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Generated VAE PGM plot: {save_path}")
    plt.close()

if __name__ == '__main__':
    draw_vae_pgm()
