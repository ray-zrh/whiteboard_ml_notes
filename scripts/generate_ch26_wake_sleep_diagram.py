import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib.patches as mpatches

# Configuration to match project style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.linewidth": 0
})

# Project Colors
COLOR_HIDDEN_FILL = '#e3f2fd' # Light Blue / White
COLOR_HIDDEN_EDGE = '#1565c0' # Dark Blue
COLOR_VISIBLE_FILL = '#e0e0e0' # Shaded Gray
COLOR_VISIBLE_EDGE = '#616161' # Dark Gray
COLOR_GENERATIVE_EDGE = '#424242' # Black/Dark Gray for Generative (top-down)
COLOR_RECOGNITION_EDGE = '#1976d2' # Blue for Recognition (bottom-up)

def create_wake_sleep_diagram():
    fig, ax = plt.subplots(figsize=(12, 9))

    # Create directed graph
    G = nx.MultiDiGraph() # Using MultiDiGraph to support explicit bidirectional distinct edges if needed

    # Define Nodes and their positions
    # Layer h^(2)
    G.add_node('S_j', pos=(-1.5, 4), type='h', label=r'$S_j$')
    G.add_node('S_j+1', pos=(1.5, 4), type='h', label=r'$S_{j+1}$')

    # Layer h^(1)
    G.add_node('S_i', pos=(-2.5, 2), type='h', label=r'$S_i$')
    G.add_node('h1_2', pos=(0, 2), type='h', label='')
    G.add_node('h1_3', pos=(2.5, 2), type='h', label='')

    # Layer v (Visible)
    G.add_node('v1', pos=(-1.5, 0), type='v', label='')
    G.add_node('v2', pos=(1.5, 0), type='v', label='')

    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    node_types = nx.get_node_attributes(G, 'type')

    # Define Edges: Generative (Top-Down, W) and Recognition (Bottom-Up, R)
    # We will draw them using matplotlib annotations to control curves properly
    gen_edges = [
        ('S_j', 'S_i', r'$w_{ji}$'),
        ('S_j', 'h1_2', ''),
        ('S_j+1', 'S_i', r'$w_{j+1,i}$'),
        ('S_j+1', 'h1_2', ''),
        ('S_i', 'v1', ''),
        ('h1_2', 'v1', ''),
        ('h1_2', 'v2', ''),
        ('h1_3', 'v2', '')
    ]

    rec_edges = [
        ('S_i', 'S_j', r'$R_{ji}$'),
        ('h1_2', 'S_j', ''),
        ('S_i', 'S_j+1', ''),
        ('h1_2', 'S_j+1', ''),
        ('v1', 'S_i', ''),
        ('v1', 'h1_2', ''),
        ('v2', 'h1_2', ''),
        ('v2', 'h1_3', '')
    ]

    # Draw Nodes
    for node in G.nodes():
        node_type = node_types[node]
        fill_color = COLOR_HIDDEN_FILL if node_type == 'h' else COLOR_VISIBLE_FILL
        edge_color = COLOR_HIDDEN_EDGE if node_type == 'h' else COLOR_VISIBLE_EDGE

        nodes_collection = nx.draw_networkx_nodes(G, {node: pos[node]}, nodelist=[node], ax=ax,
                               node_color=fill_color, edgecolors=edge_color,
                               node_size=2500, linewidths=2)

        if node_type == 'v':
            nodes_collection.set_hatch('//')

    # Draw Node Labels
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_family='serif')

    # Function to draw curved edges
    def draw_curved_edge(u, v, color, text, rad, text_offset=(0,0)):
        start = pos[u]
        end = pos[v]
        # Use an annotation with an arrow
        ax.annotate("",
                    xy=end, xycoords='data',
                    xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    shrinkA=22, shrinkB=22,
                                    patchA=None, patchB=None,
                                    connectionstyle=f"arc3,rad={rad}",
                                    linewidth=1.7),
                    )
        if text:
            # Simple midpoint estimate for text
            mid_x = (start[0] + end[0])/2 + text_offset[0]
            mid_y = (start[1] + end[1])/2 + text_offset[1]
            ax.text(mid_x, mid_y, text, color=color, fontsize=16,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # Draw Generative Edges (curve right/down)
    for u, v, text in gen_edges:
        draw_curved_edge(u, v, COLOR_GENERATIVE_EDGE, text, rad=0.15, text_offset=(0.2, 0))

    # Draw Recognition Edges (curve left/up)
    for u, v, text in rec_edges:
        draw_curved_edge(u, v, COLOR_RECOGNITION_EDGE, text, rad=0.15, text_offset=(-0.35, 0))


    # Annotations for layers and directions
    ax.text(-3.5, 4, r'$h^{(2)}$', fontsize=20, va='center')
    ax.text(-3.5, 2, r'$h^{(1)}$', fontsize=20, va='center')
    ax.text(-3.5, 0, r'$v$', fontsize=20, va='center')

    # Draw macroscopic arrows for Generation (W) and Recognition (R)
    # R: Bottom to Top
    ax.annotate("", xy=(-4.5, 4.5), xytext=(-4.5, -0.5),
                arrowprops=dict(arrowstyle="->", color=COLOR_RECOGNITION_EDGE, lw=2))
    ax.text(-4.8, 2, r'$R$', color=COLOR_RECOGNITION_EDGE, fontsize=20, va='center', ha='right')
    ax.text(-5.5, 4.5, "Recognition Connection", color=COLOR_RECOGNITION_EDGE, fontsize=14, va='bottom', ha='left')

    # W: Top to Bottom
    ax.annotate("", xy=(3.5, -0.5), xytext=(3.5, 4.5),
                arrowprops=dict(arrowstyle="->", color=COLOR_GENERATIVE_EDGE, lw=2))
    ax.text(3.8, 2, r'$W$', color=COLOR_GENERATIVE_EDGE, fontsize=20, va='center', ha='left')
    ax.text(2.5, 1.5, "Generative Connection", color=COLOR_GENERATIVE_EDGE, fontsize=14, va='top', ha='right')

    # Ensure the plot bounds comfortably encapsulate all elements and text
    ax.set_ylim(-1.5, 5.5)
    ax.set_xlim(-6.5, 5.5)

    ax.set_title("Wake-Sleep Algorithm (Sigmoid Belief Network)", y=0.98, fontsize=24, fontweight='bold', pad=20)
    ax.axis('off')

    # Ensure directories exist
    os.makedirs('notes/chapters/assets', exist_ok=True)
    output_path = 'notes/chapters/assets/ch26_wake_sleep_diagram.png'

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    create_wake_sleep_diagram()
