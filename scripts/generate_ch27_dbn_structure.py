import matplotlib.pyplot as plt
import networkx as nx
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
COLOR_HIGHLIGHT_EDGE = '#d32f2f' # Red
COLOR_TEXT_BLUE = '#1565c0'

def create_dbn_diagram():
    # Make the figure dimensions smaller to naturally compact elements
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create directed graph for the visual nodes
    G = nx.DiGraph()

    # Define Nodes and their positions
    # Compressing the y-axis, original spacing was 2, changing to 1.5
    # Compressing the x-axis, original spacing was 1.5, changing to 1.2
    y_step = 1.3
    x_step = 1.1

    # Layer h^(3) (Top RBM layer)
    G.add_node('h3_1', pos=(-x_step, y_step*3), type='h', label='')
    G.add_node('h3_2', pos=(0, y_step*3), type='h', label='')
    G.add_node('h3_3', pos=(x_step, y_step*3), type='h', label='')

    # Layer h^(2) (Bottom RBM layer / Top SBN layer)
    G.add_node('h2_1', pos=(-x_step, y_step*2), type='h', label='')
    G.add_node('h2_2', pos=(0, y_step*2), type='h', label='')
    G.add_node('h2_3', pos=(x_step, y_step*2), type='h', label='')

    # Layer h^(1)
    G.add_node('h1_1', pos=(-x_step, y_step), type='h', label='')
    G.add_node('h1_2', pos=(0, y_step), type='h', label='')
    G.add_node('h1_3', pos=(x_step, y_step), type='h', label='')

    # Layer v (Visible)
    G.add_node('v1', pos=(-x_step/2, 0), type='v', label='')
    G.add_node('v2', pos=(x_step/2, 0), type='v', label='')

    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    node_types = nx.get_node_attributes(G, 'type')

    # Draw Nodes
    for node in G.nodes():
        node_type = node_types[node]
        fill_color = COLOR_HIDDEN_FILL if node_type == 'h' else COLOR_VISIBLE_FILL
        edge_color = COLOR_HIDDEN_EDGE if node_type == 'h' else COLOR_VISIBLE_EDGE

        # Slightly smaller nodes to fit the compact layout
        nodes_collection = nx.draw_networkx_nodes(G, {node: pos[node]}, nodelist=[node], ax=ax,
                               node_color=fill_color, edgecolors=edge_color,
                               node_size=600, linewidths=1.5)

        if node_type == 'v':
            # Hatch pattern for visible nodes like whiteboard
            nodes_collection.set_hatch('////')

    # Define Edges based on whiteboard
    # Top RBM layer h3 to h2 (Undirected in concept, drawing as lines without arrows)
    rbm_edges = []
    for h3 in ['h3_1', 'h3_2', 'h3_3']:
        for h2 in ['h2_1', 'h2_2', 'h2_3']:
            rbm_edges.append((h2, h3))

    # SBN layers h2 -> h1 -> v (Directed)
    sbn_edges_h2_h1 = []
    for h2 in ['h2_1', 'h2_2', 'h2_3']:
        for h1 in ['h1_1', 'h1_2', 'h1_3']:
            sbn_edges_h2_h1.append((h2, h1))

    sbn_edges_h1_v = []
    for h1 in ['h1_1', 'h1_2', 'h1_3']:
        for v in ['v1', 'v2']:
            sbn_edges_h1_v.append((h1, v))

    # Draw edges
    # Undirected RBM
    nx.draw_networkx_edges(G, pos, edgelist=rbm_edges, ax=ax, arrows=False, width=1.0, edge_color=COLOR_EDGE)

    # Directed SBN
    nx.draw_networkx_edges(G, pos, edgelist=sbn_edges_h2_h1 + sbn_edges_h1_v, ax=ax, arrows=True,
                           arrowstyle='-|>', arrowsize=15, width=1.0, edge_color=COLOR_EDGE, node_size=600)

    # Label positioning relative to steps
    w_x = -x_step * 1.5
    h_x = -x_step * 1.9
    b_x = x_step * 1.4
    brace_x = x_step * 1.8
    text_x = x_step * 2.1

    # Add text annotations for Weight matrices W^(1), W^(2), W^(3)
    ax.text(w_x, y_step*2.5, r'$W^{(3)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(w_x, y_step*1.5, r'$W^{(2)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(w_x, y_step*0.5, r'$W^{(1)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')

    # Add text annotations for Layers h^(1), h^(2), h^(3), v
    ax.text(h_x, y_step*3, r'$h^{(3)}$', fontsize=14, va='center')
    ax.text(h_x, y_step*2, r'$h^{(2)}$', fontsize=14, va='center')
    ax.text(h_x, y_step, r'$h^{(1)}$', fontsize=14, va='center')
    ax.text(h_x, 0, r'$v$', fontsize=14, va='center')

    # Add Biases b^(0), b^(1), b^(2), b^(3)
    ax.text(b_x, y_step*3, r'$b^{(3)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(b_x, y_step*2, r'$b^{(2)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(b_x, y_step, r'$b^{(1)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(b_x, 0, r'$b^{(0)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')

    # Braces on the right to indicate RBM and Sigmoid Belief Network parts
    # Top brace for RBM (layers h3 to h2)
    # The whiteboard has the brace covering from b^(3) down to b^(2)
    # The tip of the brace points right
    ax.annotate('', xy=(brace_x, y_step*2), xytext=(brace_x, y_step*3),
                arrowprops=dict(arrowstyle='-', color='black', lw=1, shrinkA=0, shrinkB=0, connectionstyle="bar,fraction=-0.1"))
    ax.text(text_x, y_step*2.5, 'RBM', fontsize=14, va='center')

    # Bottom brace for Sigmoid Belief Network (layers h2 to v)
    # The whiteboard has the brace covering from just below b^(2) down to b^(0)
    # The tip of the brace points right
    ax.annotate('', xy=(brace_x, 0), xytext=(brace_x, y_step*1.8),
                arrowprops=dict(arrowstyle='-', color='black', lw=1, shrinkA=0, shrinkB=0, connectionstyle="bar,fraction=-0.1"))
    ax.text(text_x, y_step*0.9, 'Sigmoid Belief Network', fontsize=14, va='center')

    ax.set_title("Deep Belief Network - Hybrid Model", y=1.05, fontsize=16, fontweight='bold')
    ax.text(0, y_step*3.8, r'Hybrid Model Structure', fontsize=14, ha='center', va='center', style='italic')

    # Parameters theta
    theta_text = r'$\theta = \{W^{(1)}, W^{(2)}, W^{(3)}, b^{(0)}, b^{(1)}, b^{(2)}, b^{(3)}\}$'
    ax.text(0, y_step*3.5, theta_text, fontsize=12, ha='center', color=COLOR_TEXT_BLUE)

    # Set axis limits to further constrain the plotted area
    ax.set_xlim(-x_step*2.6, x_step*3.8)
    ax.set_ylim(-y_step*0.5, y_step*4)
    ax.axis('off')

    # Ensure directories exist
    os.makedirs('notes/chapters/assets', exist_ok=True)
    output_path = 'notes/chapters/assets/ch27_dbn_structure.png'

    # remove margins
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    create_dbn_diagram()
