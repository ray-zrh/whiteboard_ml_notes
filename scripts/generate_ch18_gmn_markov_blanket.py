
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_gmn_markov_blanket():
    # Style settings
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 0
    })

    # Figure setup - Taller to fit title and annotations
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10) # increased from 9 to 10
    ax.axis('off')

    # Styles
    color_node_default = '#ffffff'
    color_node_target = '#fff59d' # Yellow
    color_node_mb = '#e1bed7' # Purple

    color_edge = '#555555'
    color_text = '#333333'

    # Node Geometry
    radius = 0.5
    node_lw = 2.0

    # Layout Strategy:
    # Use distinct vertical layers to separate components

    nodes = {
        'target': {'pos': (6.0, 4.0), 'label': r'$x_i$', 'color': color_node_target},

        # Markov Blanket (Neighbors)
        'n1': {'pos': (4.0, 6.0), 'label': r'$x_j$', 'color': color_node_mb}, # Top Left
        'n2': {'pos': (8.0, 6.0), 'label': r'$x_k$', 'color': color_node_mb}, # Top Right
        'n3': {'pos': (6.0, 1.5), 'label': r'$x_m$', 'color': color_node_mb}, # Bottom

        # Others (Independent)
        'o1': {'pos': (1.5, 7.0), 'label': r'$x_a$', 'color': color_node_default}, # Lowered slightly
        'o2': {'pos': (10.5, 7.0), 'label': r'$x_b$', 'color': color_node_default}, # Lowered slightly
    }

    # Undirected Edges
    edges = [
        ('target', 'n1'),
        ('target', 'n2'),
        ('target', 'n3'),
        ('n1', 'n2'),
        ('n1', 'o1'),
        ('n2', 'o2'),
    ]

    # Draw Edges
    for start, end in edges:
        p1 = nodes[start]['pos']
        p2 = nodes[end]['pos']
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color_edge, lw=2, zorder=1)

    # Draw Nodes
    for name, data in nodes.items():
        x, y = data['pos']
        label = data['label']
        color = data['color']

        circle = patches.Circle((x, y), radius, facecolor=color, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=16, fontweight='bold', zorder=11, color=color_text)

    # --- Annotations ---

    # 1. Target Node Label (Bottom Left of Target)
    ax.annotate("Target Node\n" + r"$x_i$",
                xy=(6.0 - radius/1.4, 4.0 - radius/1.4),
                xytext=(3.5, 3.0),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='#555555'),
                fontsize=12, color='#555555', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))

    # 2. Markov Blanket Label (Top Center)
    # Move text higher to y=8.0, but node n1 is at y=6.0.
    # Title will be at y=9.5. Plenty of space.
    ax.annotate("Markov Blanket\n(Neighbors)",
                xy=(4.0, 6.0 + radius),
                xytext=(6.0, 8.0),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", color='#555555'),
                fontsize=12, color='#555555', ha='center', va='center')

    ax.annotate("",
                xy=(8.0, 6.0 + radius),
                xytext=(6.0, 7.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='#555555'))

    ax.annotate("",
                xy=(6.0 + radius, 1.5),
                xytext=(6.8, 7.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="bar,fraction=-0.4", path_effects=None, color='#555555', alpha=0.3))


    # 3. Other Nodes Label (Top Corners)
    # o1 is at y=7.0.
    # Text at y=8.0.
    ax.annotate("Indep. Node",
                xy=(1.5, 7.0 + radius),
                xytext=(1.5, 8.2),
                arrowprops=dict(arrowstyle="->", color='#555555'),
                fontsize=11, color='#555555', ha='center', va='center')

    ax.annotate("Indep. Node",
                xy=(10.5, 7.0 + radius),
                xytext=(10.5, 8.2),
                arrowprops=dict(arrowstyle="->", color='#555555'),
                fontsize=11, color='#555555', ha='center', va='center')


    # Property Box
    ax.text(9.0, 1.0, r"Property:" + "\n" + r"$x_i \perp x_{others} \mid x_{ne(i)}$",
            ha='center', va='center', fontsize=14, fontweight='bold', color='#333333',
            bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.5'))

    # Title - Use plt.title for safety, or place high up at y=9.5
    # Since axis limit is 10, 9.5 is safe.
    plt.text(6.0, 9.5, "Gaussian Markov Network (Undirected)", ha='center', va='center', fontsize=18, fontweight='bold', color='#333333')

    # Save
    output_path = os.path.join(output_dir, 'ch18_gmn_markov_blanket.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_gmn_markov_blanket()
