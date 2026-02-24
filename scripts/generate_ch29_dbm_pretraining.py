import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_dbm_pretraining_diagram():
    output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 0
    })

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-1, 15)
    ax.axis('off')

    # Styles
    color_h_fill = '#e8f5e9'  # Light Green
    color_h_edge = '#2e7d32'  # Dark Green
    color_v_fill = '#e3f2fd'  # Light Blue
    color_v_edge = '#1565c0'  # Dark Blue
    color_box_edge = '#1565c0'
    color_line = '#bdbdbd'
    color_active_line = '#4caf50'

    node_xs = [-1.5, 0, 1.5]

    def draw_layer_nodes(center_x, y, labels, layer_type='v', is_input=False):
        fill = color_h_fill if layer_type == 'h' else color_v_fill
        edge = color_h_edge if layer_type == 'h' else color_v_edge

        for i, dx in enumerate(node_xs):
            x = center_x + dx
            circle = patches.Circle((x, y), 0.5, facecolor=fill, edgecolor=edge, lw=2.0, zorder=5)
            if layer_type == 'v' and is_input:
                circle.set_hatch('//////')
            ax.add_patch(circle)
            if i == 0:
                ax.text(x - 1.2, y, labels, fontsize=16, ha='right', va='center', color='#212121', fontweight='bold')

    def draw_rbm_block(y_bottom, label_bottom, label_top, is_bottom_v=False, label_w_up="", label_w_dn=""):
        # Box
        bbox = patches.FancyBboxPatch((-3, y_bottom - 1), 6, 4,
                                       boxstyle="round,pad=0.2,rounding_size=0.5",
                                       fc='none', ec=color_box_edge, lw=2.0, ls='--', zorder=0)
        ax.add_patch(bbox)

        y_top = y_bottom + 2

        draw_layer_nodes(0, y_bottom, label_bottom, layer_type='v' if is_bottom_v else 'h', is_input=is_bottom_v)
        draw_layer_nodes(0, y_top, label_top, layer_type='h')

        # Connections
        for bx in node_xs:
            for tx in node_xs:
                ax.plot([bx, tx], [y_bottom + 0.5, y_top - 0.5], color=color_line, lw=1.5, zorder=1)

        # Arrows for weights
        if label_w_up:
            ax.annotate('', xy=(1.0, y_top - 0.6), xytext=(1.0, y_bottom + 0.6),
                        arrowprops=dict(arrowstyle="->", color=color_active_line, lw=2))
            ax.text(1.2, (y_bottom + y_top)/2, label_w_up, color=color_active_line, fontsize=16, va='center')

        if label_w_dn:
            ax.annotate('', xy=(-1.0, y_bottom + 0.6), xytext=(-1.0, y_top - 0.6),
                        arrowprops=dict(arrowstyle="->", color=color_active_line, lw=2))
            ax.text(-1.2, (y_bottom + y_top)/2, label_w_dn, color=color_active_line, fontsize=16, va='center', ha='right')

        ax.text(3.5, (y_bottom + y_top)/2, 'RBM', fontsize=18, color=color_box_edge, va='center')

    # Draw the 3 RBM blocks from bottom to top
    draw_rbm_block(1, r"$\mathbf{v}$", r"$\mathbf{h}^{(1)}$", is_bottom_v=True, label_w_up=r"$2W^{(1)}$", label_w_dn=r"$W^{(1)}$")
    draw_rbm_block(6, r"$\mathbf{h}^{(1)}$", r"$\mathbf{h}^{(2)}$", is_bottom_v=False, label_w_up=r"$W^{(2)}$", label_w_dn=r"$W^{(2)}$")
    draw_rbm_block(11, r"$\mathbf{h}^{(2)}$", r"$\mathbf{h}^{(3)}$", is_bottom_v=False, label_w_up=r"$W^{(3)}$", label_w_dn=r"$2W^{(3)}$")

    output_path = os.path.join(output_dir, 'ch29_dbm_pretraining.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    create_dbm_pretraining_diagram()
