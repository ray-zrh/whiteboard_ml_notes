import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_layerwise_pretraining_diagram():
    output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 0
    })

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 32)
    ax.set_ylim(-1, 10)
    ax.axis('off')

    # Colors defined based on ch21 and project styles
    color_h_fill = '#e8f5e9'  # Light Green
    color_h_edge = '#2e7d32'  # Dark Green

    color_v_fill = '#e3f2fd'  # Light Blue
    color_v_edge = '#1565c0'  # Dark Blue

    color_frozen_fill = '#f5f5f5' # Light Gray
    color_frozen_edge = '#9e9e9e' # Dark Gray

    color_line = '#bdbdbd'
    color_active_line = '#4caf50' # Green for active training

    layer_y = [2, 4.5, 7] # v, h1, h2
    node_xs = [-1.5, 0, 1.5]

    def draw_layer_nodes(center_x, y, labels, style='active', layer_type='v'):
        """Draws a row of three nodes."""
        fill = color_h_fill if layer_type == 'h' else color_v_fill
        edge = color_h_edge if layer_type == 'h' else color_v_edge

        if style == 'frozen':
            fill = color_frozen_fill
            edge = color_frozen_edge
            text_color = '#757575'
        else:
            text_color = '#212121'

        for i, dx in enumerate(node_xs):
            x = center_x + dx
            circle = patches.Circle((x, y), 0.5, facecolor=fill, edgecolor=edge, lw=2.0, zorder=5)
            ax.add_patch(circle)
            if i == 0:
                ax.text(x - 1.2, y, labels, fontsize=16, ha='right', va='center', color=text_color, fontweight='bold')

    def draw_connections(center_x, y_bottom, y_top, style='active'):
        """Draws a fully connected bipartite graph between two layers."""
        color = color_active_line if style == 'active' else color_line
        alpha = 0.8 if style == 'active' else 0.4
        lw = 2.0 if style == 'active' else 1.0

        for bottom_x in node_xs:
            for top_x in node_xs:
                ax.plot([center_x + bottom_x, center_x + top_x], [y_bottom + 0.5, y_top - 0.5],
                        color=color, lw=lw, zorder=1, alpha=alpha)

    # --- STEP 1: Train 1st RBM (Left) ---
    c1 = 5
    # Bounding Box
    bbox1 = patches.FancyBboxPatch((c1 - 4, layer_y[0] - 1), 7, layer_y[1] - layer_y[0] + 2,
                                   boxstyle="round,pad=0.2,rounding_size=0.5",
                                   fc='#e8f5e9', ec='#4caf50', lw=2, alpha=0.3, zorder=0)
    ax.add_patch(bbox1)

    draw_layer_nodes(c1, layer_y[0], r"$\mathbf{v}$", style='active', layer_type='v')
    draw_layer_nodes(c1, layer_y[1], r"$\mathbf{h}^{(1)}$", style='active', layer_type='h')
    draw_connections(c1, layer_y[0], layer_y[1], style='active')

    ax.text(c1 + 2.5, (layer_y[0] + layer_y[1])/2, r"$W^{(1)}$", fontsize=16, color=color_active_line, fontweight='bold', va='center')
    ax.text(c1, 9, "Step 1: Train 1st RBM", fontsize=18, fontweight='bold', ha='center', color='#212121')
    ax.text(c1, 8.2, r"Maximize $P(\mathbf{v})$", fontsize=14, ha='center', color='#616161')

    # -> Arrow
    ax.arrow(c1 + 4, 4.5, 3, 0, head_width=0.4, head_length=0.4, fc='#424242', ec='#424242', lw=2)

    # --- STEP 2: Train 2nd RBM (Middle) ---
    c2 = 16
    # Bounding box for active top
    bbox2 = patches.FancyBboxPatch((c2 - 4.2, layer_y[1] - 1), 7.5, layer_y[2] - layer_y[1] + 2,
                                   boxstyle="round,pad=0.2,rounding_size=0.5",
                                   fc='#e8f5e9', ec='#4caf50', lw=2, alpha=0.3, zorder=0)
    ax.add_patch(bbox2)

    draw_layer_nodes(c2, layer_y[0], r"$\mathbf{v}$", style='frozen', layer_type='v')
    draw_layer_nodes(c2, layer_y[1], r"$\mathbf{h}^{(1)}$", style='active', layer_type='h')
    draw_layer_nodes(c2, layer_y[2], r"$\mathbf{h}^{(2)}$", style='active', layer_type='h')
    draw_connections(c2, layer_y[0], layer_y[1], style='frozen')
    draw_connections(c2, layer_y[1], layer_y[2], style='active')

    # Frozen labels
    ax.text(c2 - 4, (layer_y[0] + layer_y[1])/2, r"Frozen $W^{(1)}$", fontsize=12, color='#757575', va='center', ha='right')

    ax.text(c2 + 2.5, (layer_y[1] + layer_y[2])/2, r"$W^{(2)}$", fontsize=16, color=color_active_line, fontweight='bold', va='center')
    ax.text(c2, 9, "Step 2: Train 2nd RBM", fontsize=18, fontweight='bold', ha='center', color='#212121')
    ax.text(c2, 8.2, r"Given $W^{(1)}$, Maximize bounds on $P(\mathbf{v})$", fontsize=14, ha='center', color='#616161')

    # -> Arrow
    ax.arrow(c2 + 4.5, 4.5, 3, 0, head_width=0.4, head_length=0.4, fc='#424242', ec='#424242', lw=2)

    # --- STEP 3: Complete DBN Generation (Right) ---
    c3 = 27

    # Show active top down arrows for full Generative Model
    layer_y_dbn = [2, 4.5, 7]
    for i, dx in enumerate(node_xs):
        x = c3 + dx

        # h2
        circle = patches.Circle((x, layer_y_dbn[2]), 0.5, facecolor=color_h_fill, edgecolor=color_h_edge, lw=2.0, zorder=5)
        ax.add_patch(circle)
        if i == 0: ax.text(x - 1.2, layer_y_dbn[2], r"$\mathbf{h}^{(2)}$", fontsize=16, ha='right', va='center', fontweight='bold')

        # h1
        circle = patches.Circle((x, layer_y_dbn[1]), 0.5, facecolor=color_h_fill, edgecolor=color_h_edge, lw=2.0, zorder=5)
        ax.add_patch(circle)
        if i == 0: ax.text(x - 1.2, layer_y_dbn[1], r"$\mathbf{h}^{(1)}$", fontsize=16, ha='right', va='center', fontweight='bold')

        # v
        circle = patches.Circle((x, layer_y_dbn[0]), 0.5, facecolor=color_v_fill, edgecolor=color_v_edge, lw=2.0, zorder=5)
        ax.add_patch(circle)
        if i == 0: ax.text(x - 1.2, layer_y_dbn[0], r"$\mathbf{v}$", fontsize=16, ha='right', va='center', fontweight='bold')

    # Draw RBM Top connections
    for bottom_x in node_xs:
        for top_x in node_xs:
            ax.plot([c3 + bottom_x, c3 + top_x], [layer_y_dbn[1] + 0.5, layer_y_dbn[2] - 0.5],
                    color=color_line, lw=1.5, zorder=1)

    # Draw Directed SBN Connections for bottom generative part
    for bottom_x in node_xs:
        for top_x in node_xs:
            ax.annotate("",
                        xy=(c3 + bottom_x, layer_y_dbn[0] + 0.5), xycoords='data',
                        xytext=(c3 + top_x, layer_y_dbn[1] - 0.5), textcoords='data',
                        arrowprops=dict(arrowstyle="-|>", color=color_line, lw=1.5, shrinkA=0, shrinkB=0))


    # Add large generation arrow indicator
    ax.annotate('', xy=(c3 + 3.5, 2.5), xytext=(c3 + 3.5, 6.5),
                arrowprops=dict(arrowstyle='-|>', color='#d32f2f', lw=3, shrinkA=0, shrinkB=0))
    ax.text(c3 + 4, 4.5, "Top-Down\nGeneration", fontsize=14, color='#d32f2f', va='center', fontweight='bold')

    ax.text(c3, 9, "Step 3: Assembled DBN", fontsize=18, fontweight='bold', ha='center', color='#212121')
    ax.text(c3, 8.2, r"RBM on top of directed SBN", fontsize=14, ha='center', color='#616161')

    # 5. Title
    fig.suptitle("Greedy Layer-wise Pre-training Concept", y=1.05, fontsize=24, fontweight='bold', color='#212121')

    output_path = os.path.join(output_dir, 'ch27_layerwise_pretraining.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    create_layerwise_pretraining_diagram()
