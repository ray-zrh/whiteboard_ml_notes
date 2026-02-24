import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

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
COLOR_EDGE = '#424242'
COLOR_TEXT_BLUE = '#1565c0'

def create_bm_diagram():
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create an undirected graph for the Boltzmann Machine
    G = nx.Graph()

    # Number of nodes
    n_visible = 3
    n_hidden = 4
    total_nodes = n_visible + n_hidden

    # Calculate positions for a circular layout
    angles = np.linspace(0, 2 * np.pi, total_nodes, endpoint=False)
    # Rotate by 90 degrees (pi/2) to have symmetry nicely aligned
    # Actually, we want visible nodes at the bottom.
    # Let's adjust angles so that the 3 visible nodes are at the bottom.
    # Angles near 3*pi/2 (-pi/2) are at the bottom.
    # Let's shift angles to start from bottom-ish
    shift = np.pi / 2  + (np.pi / total_nodes)
    angles += shift

    nodes = []

    # 4 hidden nodes on top
    for i in range(n_hidden):
        node_id = f'h_{i+1}'
        # Use first 4 angles (which would be on top if we shifted properly)
        idx = i # 0, 1, 2, 3
        # We want top 4 to be above y=0, bottom 3 to be below y=0.
        # Let's just manually set positions to resemble the whiteboard
        pass

    # Actually, let's set manual pos on a circle
    # The whiteboard has them arranged roughly in a circle.
    # 7 nodes on a circle. Let's use indices 0..6
    # 0, 1, 2 at the bottom (visible)
    # 3, 4, 5, 6 at the top (hidden)

    start_angle_v = -3 * np.pi / 4
    end_angle_v = -np.pi / 4
    angles_v = np.linspace(start_angle_v, end_angle_v, n_visible)

    start_angle_h = np.pi # left
    end_angle_h = 0 # right
    # To space 4 nodes evenly on the top arc
    angles_h = np.linspace(start_angle_h - np.pi/8, end_angle_h + np.pi/8, n_hidden)

    # However, regular circle is usually prettier
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    # [0, 51, 102, 154, 205, 257, 308] degrees
    # If we shift by 90, we get top node.
    # Let's map indices:
    # Top 4: angles between 0 and pi.
    # Bottom 3: angles between pi and 2pi.

    pos = {}

    # Visible nodes (v1, v2, v3)
    pos['v1'] = (-0.8, -0.6)
    pos['v2'] = (0.0, -0.9)
    pos['v3'] = (0.8, -0.6)

    # Hidden nodes (h1, h2, h3, h4)
    pos['h1'] = (-0.9, 0.2)
    pos['h2'] = (-0.4, 0.9)
    pos['h3'] = (0.4, 0.9)
    pos['h4'] = (0.9, 0.2)

    G.add_nodes_from(['v1', 'v2', 'v3'], type='v')
    G.add_nodes_from(['h1', 'h2', 'h3', 'h4'], type='h')

    # Add edges between all pairs (fully connected)
    import itertools
    for u, v in itertools.combinations(G.nodes(), 2):
        G.add_edge(u, v)

    node_types = nx.get_node_attributes(G, 'type')

    # Draw Edges first so they are behind nodes
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, edge_color=COLOR_EDGE, alpha=0.6)

    # Draw Hidden Nodes
    h_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'h']
    nx.draw_networkx_nodes(G, pos, nodelist=h_nodes, ax=ax,
                           node_color=COLOR_HIDDEN_FILL, edgecolors=COLOR_HIDDEN_EDGE,
                           node_size=1500, linewidths=1.5)

    # Draw Visible Nodes
    v_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'v']
    nodes_collection = nx.draw_networkx_nodes(G, pos, nodelist=v_nodes, ax=ax,
                           node_color=COLOR_VISIBLE_FILL, edgecolors=COLOR_VISIBLE_EDGE,
                           node_size=1500, linewidths=1.5)

    # Hatch pattern for visible nodes
    nodes_collection.set_hatch('////')

    # Add labels for specific nodes matching whiteboard
    # The whiteboard has v_i for one visible node, h_j for one hidden node, w_ij for edge
    ax.text(-1.1, -0.6, r'$v_i$', fontsize=18, va='center', ha='right')
    ax.text(-1.2, 0.2, r'$h_j$', fontsize=18, va='center', ha='right')

    # Add label for an edge
    # Midpoint between v1 and h1
    mid_x = (pos['v1'][0] + pos['h1'][0]) / 2
    mid_y = (pos['v1'][1] + pos['h1'][1]) / 2
    ax.text(mid_x - 0.1, mid_y, r'$w_{ij}$', fontsize=18, va='center', ha='right', color=COLOR_EDGE)

    # Add some additional text definitions
    # V = {0,1}^D, h = {0,1}^P
    # Since we shouldn't clutter the main graph, we can leave the heavy equations for Markdown.

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    # Ensure directories exist
    os.makedirs('notes/chapters/assets', exist_ok=True)
    output_path = 'notes/chapters/assets/ch29_boltzmann_machine.png'

    # remove margins
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    create_bm_diagram()
