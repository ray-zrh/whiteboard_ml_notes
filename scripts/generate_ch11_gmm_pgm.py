import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_pgm():
    """Draws a high-quality, 'Daft'-style Probabilistic Graphical Model (PGM) for GMM."""

    # Modern, publication-quality settings
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Style constants
    node_radius = 0.45
    node_lw = 2.0
    edge_lw = 1.5
    font_size = 18
    param_font_size = 14

    # Color palette (Soft, professional)
    observed_face = '#dddddd'    # Light gray for observed
    latent_face = '#ffffff'      # White for latent
    node_edge = '#333333'        # Dark charcoal
    text_color = '#333333'
    plate_edge = '#555555'
    plate_text = '#555555'

    # Positions
    # Top row: p (parameter) --------> z (latent)
    # Bottom row: mu, Sigma (param) -> x (observed)
    pos_p = (1.5, 3.5)
    pos_z = (4.0, 3.5)
    pos_theta = (1.5, 1.5) # mu, Sigma
    pos_x = (4.0, 1.5)

    # --- Helper Functions ---
    def add_node(pos, label, face_color=latent_face, label_offset=(0,0)):
        circle = patches.Circle(pos, node_radius,
                                edgecolor=node_edge, facecolor=face_color,
                                linewidth=node_lw, zorder=10)
        ax.add_patch(circle)
        ax.text(pos[0]+label_offset[0], pos[1]+label_offset[1], label,
                ha='center', va='center', fontsize=font_size,
                color=text_color, fontweight='bold', zorder=11)
        return circle

    def add_small_node(pos, label):
        # For parameters like p, mu, Sigma - drawn as small text nodes or dots
        # Using standard notation: just the text label, or a tiny dot
        dot = patches.Circle(pos, 0.05, color=node_edge, zorder=10)
        ax.add_patch(dot)
        ax.text(pos[0]-0.2, pos[1], label,
                ha='right', va='center', fontsize=param_font_size,
                color=text_color, zorder=11)
        return dot

    def add_edge(start, end):
        # Calculate arrow shortening to not overlap nodes
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4",
                                    color=node_edge, lw=edge_lw,
                                    shrinkA=12, shrinkB=12))

    # --- Draw Components ---

    # 1. Plate (N samples)
    # Wraps around z and x
    plate_margin = 0.8
    plate_x = pos_x[0] - plate_margin
    plate_y = pos_x[1] - plate_margin
    plate_w = 2 * plate_margin
    plate_h = (pos_z[1] - pos_x[1]) + 2 * plate_margin

    rect = patches.Rectangle((plate_x, plate_y), plate_w, plate_h,
                             linewidth=1.5, edgecolor=plate_edge,
                             facecolor='none', linestyle='-', zorder=1,
                             joinstyle='round', capstyle='round') # Rounded feel
    ax.add_patch(rect)

    # Plate label "N"
    ax.text(plate_x + plate_w - 0.2, plate_y + 0.2, r'$N$',
            fontsize=14, ha='right', va='bottom', color=plate_text, fontweight='bold')

    # 2. Nodes
    add_node(pos_z, r'$z$')
    add_node(pos_x, r'$x$', face_color=observed_face) # Shaded

    # 3. Parameters (outside plate)
    add_small_node(pos_p, r'$p$')
    add_small_node(pos_theta, r'$\mu, \Sigma$')

    # 4. Edges (Dependencies)
    add_edge(pos_p, pos_z)      # p -> z
    add_edge(pos_z, pos_x)      # z -> x
    add_edge(pos_theta, pos_x)  # mu, Sigma -> x

    # --- Annotations for "Easy to Understand" ---
    # Optional: concise labels to explain the arrows
    # ax.text(2.7, 3.65, "Selects\nCluster", fontsize=10, ha='center', color='#777777', style='italic')
    # ax.text(2.7, 1.65, "Generates\nData", fontsize=10, ha='center', color='#777777', style='italic')

    # Optimize layout
    plt.tight_layout()

    # Save
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '../notes/chapters/assets')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'ch11_gmm_pgm.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Generated PGM plot: {save_path}")
    plt.close()

if __name__ == "__main__":
    draw_pgm()
