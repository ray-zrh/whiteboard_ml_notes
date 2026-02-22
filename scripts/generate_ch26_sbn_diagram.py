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
COLOR_HIDDEN_FILL = '#e3f2fd' # Light Blue / White (for hidden nodes in whiteboard)
COLOR_HIDDEN_EDGE = '#1565c0' # Dark Blue
COLOR_VISIBLE_FILL = '#e0e0e0' # Shaded Gray (for visible nodes in whiteboard)
COLOR_VISIBLE_EDGE = '#616161' # Dark Gray
COLOR_EDGE = '#424242'
COLOR_HIGHLIGHT_EDGE = '#d32f2f' # Red for specific edges

def create_sbn_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create directed graph
    G = nx.DiGraph()

    # Define Nodes and their positions (Layered structure from whiteboard)
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

    # Define Edges based on whiteboard
    edges = [
        ('S_j', 'S_i'),
        ('S_j', 'h1_2'),
        ('S_j+1', 'S_i'), # Highlighted edge in whiteboard
        ('S_j+1', 'h1_2'),
        ('S_i', 'v1'),
        ('h1_2', 'v1'),
        ('h1_2', 'v2'),
        ('h1_3', 'v2')
    ]
    G.add_edges_from(edges)

    # Draw Nodes
    for node in G.nodes():
        node_type = node_types[node]
        fill_color = COLOR_HIDDEN_FILL if node_type == 'h' else COLOR_VISIBLE_FILL
        edge_color = COLOR_HIDDEN_EDGE if node_type == 'h' else COLOR_VISIBLE_EDGE

        # Determine filling style (hatch for visible like whiteboard)
        if node_type == 'v':
            # NetworkX does not support hatch directly in draw_networkx_nodes well,
            # so we draw them using circles manually or rely on colors.
            # We will use shaded gray to represent the hatch.
            pass

        nodes_collection = nx.draw_networkx_nodes(G, {node: pos[node]}, nodelist=[node], ax=ax,
                               node_color=fill_color, edgecolors=edge_color,
                               node_size=2500, linewidths=2)

        if node_type == 'v':
            # Set the hatch pattern directly on the PathCollection returned by nx.draw_networkx_nodes
            nodes_collection.set_hatch('//')

    # Draw Edges
    for u, v in G.edges():
        edge_c = COLOR_HIGHLIGHT_EDGE if (u == 'S_j+1' and v == 'S_i') else COLOR_EDGE
        width = 2.5 if (u == 'S_j+1' and v == 'S_i') else 1.5
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, arrows=True,
                               arrowstyle='-|>', arrowsize=20, width=width,
                               edge_color=edge_c, node_size=2500, connectionstyle='arc3,rad=0.0')

    # Draw specially curved edge w_ji
    # Networkx allows rad to curve edges. The whiteboard has a slight curve for S_j to S_i
    # We will just draw labels on edges

    # Edge Labels
    # S_j -> S_i is W_ji
    # S_j+1 -> S_i is W_j+1,i (approx)
    edge_labels = {
        ('S_j', 'S_i'): r'$W_{ji}$',
        ('S_j+1', 'S_i'): r'$W_{j+1,i}$'
    }

    # Use ax.text for better placement than draw_networkx_edge_labels
    ax.text(-2.3, 3.2, r'$w_{ji}$', fontsize=16, color=COLOR_EDGE)
    ax.text(-0.5, 3.2, r'$w_{j+1,i}$', fontsize=16, color=COLOR_HIGHLIGHT_EDGE)

    # Draw Node Labels
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_family='serif')

    # Layer Annotations (h^(2), h^(1), v)
    ax.text(-3.5, 4, r'$h^{(2)}$', fontsize=20, va='center')
    ax.text(-3.5, 2, r'$h^{(1)}$', fontsize=20, va='center')
    ax.text(-3.5, 0, r'$v$', fontsize=20, va='center')

    ax.set_title("Sigmoid Belief Network", y=0.95, fontsize=24, fontweight='bold', pad=20)
    ax.axis('off')

    # Ensure directories exist
    os.makedirs('notes/chapters/assets', exist_ok=True)
    output_path = 'notes/chapters/assets/ch26_sbn_diagram.png'

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    create_sbn_diagram()
