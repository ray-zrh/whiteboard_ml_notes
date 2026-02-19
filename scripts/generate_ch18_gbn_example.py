
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_gbn_concept_diagram():
    # Style settings similar to ch14_hmm_concept
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 0
    })

    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Styles
    color_node = '#ffffff'
    color_edge = '#333333'
    color_text = '#333333'
    color_param = '#d62728' # Red for weights

    # Node Geometry
    radius = 0.6
    node_lw = 2.0

    # Define Node Positions (Diamond Structure)
    # x1 (Top)
    # x2 (Left), x3 (Right)
    # x4 (Bottom)
    nodes = {
        'x1': {'pos': (5.0, 8.5), 'label': r'$x_1$'},
        'x2': {'pos': (3.0, 5.5), 'label': r'$x_2$'},
        'x3': {'pos': (7.0, 5.5), 'label': r'$x_3$'},
        'x4': {'pos': (5.0, 2.5), 'label': r'$x_4$'}
    }

    # Draw Edges first (so they are behind nodes)
    # x1 -> x2, x1 -> x3
    # x2 -> x4, x3 -> x4
    edges = [
        ('x1', 'x2', r'$w_{12}$'),
        ('x1', 'x3', r'$w_{13}$'),
        ('x2', 'x4', r'$w_{24}$'),
        ('x3', 'x4', r'$w_{34}$')
    ]

    for start, end, weight_label in edges:
        p1 = nodes[start]['pos']
        p2 = nodes[end]['pos']

        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Shorten start/end by radius
        dist = np.sqrt(dx**2 + dy**2)
        scale_start = (radius + 0.1) / dist
        scale_end = (radius + 0.1) / dist

        x_start = p1[0] + dx * scale_start
        y_start = p1[1] + dy * scale_start
        x_end = p2[0] - dx * scale_end
        y_end = p2[1] - dy * scale_end

        # Draw Arrow
        ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge, shrinkA=0, shrinkB=0))

        # Label Weight (roughly middle, slight offset)
        mid_x = (x_start + x_end) / 2
        mid_y = (y_start + y_end) / 2

        # Adjust offset based on slope to avoid overlapping line
        offset_x = 0.3 if dx > 0 else -0.3
        offset_y = 0.1

        ax.text(mid_x + offset_x, mid_y + offset_y, weight_label,
                ha='center', va='center', fontsize=12, color=color_param, fontweight='bold')

    # Draw Nodes
    for name, data in nodes.items():
        x, y = data['pos']
        label = data['label']

        circle = patches.Circle((x, y), radius, facecolor=color_node, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=16, fontweight='bold', zorder=11, color=color_text)

        # Add small text for local model?
        # e.g. next to nodes
        if name == 'x4':
            note = r"$x_4 \sim \mathcal{N}(\mu_4 + w_{24}(x_2-\mu_2) + w_{34}(x_3-\mu_3), \sigma_4^2)$"
            ax.text(x + 1.2, y, note, ha='left', va='center', fontsize=12, color='#555555',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Title
    ax.text(5.0, 9.5, "Gaussian Bayesian Network (Structure)", ha='center', va='center', fontsize=18, fontweight='bold', color='#333333')

    # Save
    output_path = os.path.join(output_dir, 'ch18_gbn_example.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_gbn_concept_diagram()
