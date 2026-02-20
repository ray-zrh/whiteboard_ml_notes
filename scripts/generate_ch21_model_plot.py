import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_rbm_diagram():
    output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 0
    })

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    color_h_fill = '#e8f5e9'  # Light Green
    color_h_edge = '#2e7d32'  # Dark Green

    color_v_fill = '#e3f2fd'  # Light Blue
    color_v_edge = '#1565c0'  # Dark Blue

    color_factor = '#ffecb3'  # Light Amber
    color_factor_edge = '#ff8f00' # Dark Amber

    color_line = '#9e9e9e'

    # Coordinates
    x_h = [3, 5, 7, 9]
    y_h = 6.5

    x_v = [2, 4, 6, 8, 10]
    y_v = 2.5

    # 1. Background Boxes for Layers to emphasize Bipartite Nature
    layer_h_box = patches.FancyBboxPatch((0.5, y_h - 1.2), 10, 2.4, boxstyle="round,pad=0.2",
                                         fc='#f5f5f5', ec='none', zorder=0, alpha=0.5)
    layer_v_box = patches.FancyBboxPatch((0.0, y_v - 1.2), 11, 2.4, boxstyle="round,pad=0.2",
                                         fc='#f5f5f5', ec='none', zorder=0, alpha=0.5)
    ax.add_patch(layer_h_box)
    ax.add_patch(layer_v_box)

    ax.text(12.0, y_h, r"Hidden Layer $\mathbf{h}$" + "\n(Latent Variables)", fontsize=16, color=color_h_edge, va='center', fontweight='bold', ha='center')
    ax.text(12.0, y_v, r"Visible Layer $\mathbf{v}$" + "\n(Observed Data)", fontsize=16, color=color_v_edge, va='center', fontweight='bold', ha='center')

    # 2. Draw Edges (Intersection lines)
    for i, h in enumerate(x_h):
        for j, v in enumerate(x_v):
            ax.plot([h, v], [y_h, y_v], color=color_line, lw=1.0, zorder=1, alpha=0.6)

    # Draw W weight label on middle edge - move to the left a bit to avoid edge clutter
    ax.text(1.2, 4.5, r"$\mathbf{W} \in \mathbb{R}^{m \times n}$", fontsize=18, color='#424242', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='#e0e0e0', boxstyle='round,pad=0.5', lw=1), zorder=10)

    # 3. Draw Hidden Nodes & Factors
    for i, h in enumerate(x_h):
        # Node
        circle = patches.Circle((h, y_h), 0.5, facecolor=color_h_fill, edgecolor=color_h_edge, lw=2.0, zorder=5)
        ax.add_patch(circle)
        ax.text(h, y_h, f"$h_{i+1}$", fontsize=16, ha='center', va='center', zorder=6, color='#212121')

        # Factor (Top)
        factor_y = y_h + 1.5
        square = patches.Rectangle((h - 0.2, factor_y - 0.2), 0.4, 0.4, facecolor=color_factor, edgecolor=color_factor_edge, lw=2.0, zorder=5)
        ax.add_patch(square)
        ax.plot([h, h], [y_h + 0.5, factor_y - 0.2], color=color_factor_edge, lw=2.0, zorder=1, linestyle='--')

        if i == len(x_h) - 1:
            ax.text(h + 0.6, factor_y, r"$\exp(\beta_i h_i)$", fontsize=12, color=color_factor_edge, va='center', fontweight='bold')

    # 4. Draw Visible Nodes & Factors
    for j, v in enumerate(x_v):
        # Node
        circle = patches.Circle((v, y_v), 0.5, facecolor=color_v_fill, edgecolor=color_v_edge, lw=2.0, zorder=5)
        ax.add_patch(circle)
        ax.text(v, y_v, f"$v_{j+1}$", fontsize=16, ha='center', va='center', zorder=6, color='#212121')

        # Factor (Bottom)
        factor_y = y_v - 1.5
        square = patches.Rectangle((v - 0.2, factor_y - 0.2), 0.4, 0.4, facecolor=color_factor, edgecolor=color_factor_edge, lw=2.0, zorder=5)
        ax.add_patch(square)
        ax.plot([v, v], [y_v - 0.5, factor_y + 0.2], color=color_factor_edge, lw=2.0, zorder=1, linestyle='--')

        if j == len(x_v) - 1:
            ax.text(v + 0.6, factor_y, r"$\exp(\alpha_j v_j)$", fontsize=12, color=color_factor_edge, va='center', fontweight='bold')

    # 5. Title
    ax.text(7.0, 9.2, "Restricted Boltzmann Machine (RBM)",
            fontsize=22, ha='center', color='#212121', fontweight='bold')
    ax.text(7.0, 8.7, "Factor Graph View: Bipartite Structure with Node & Edge Factors",
            fontsize=16, ha='center', color='#616161')

    output_path = os.path.join(output_dir, 'ch21_rbm_structure.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_rbm_diagram()
