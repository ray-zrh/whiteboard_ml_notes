import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib.patches as patches

# Configuration to match project style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.linewidth": 0
})

# Project Colors
COLOR_HIDDEN_FILL = '#e3f2fd'
COLOR_HIDDEN_EDGE = '#1565c0'
COLOR_VISIBLE_FILL = '#e0e0e0'
COLOR_VISIBLE_EDGE = '#616161'
COLOR_EDGE = '#424242'
COLOR_HIGHLIGHT_EDGE = '#d32f2f' # Red for generation arrow
COLOR_TEXT_BLUE = '#1565c0'

def create_generation_diagram():
    fig, ax = plt.subplots(figsize=(8, 7))

    G = nx.DiGraph()
    y_step = 1.3
    x_step = 1.2

    # Draw the main DBN model (standard 4 layers)
    # Layer h^(3) (Top RBM layer)
    G.add_node('h3_1', pos=(-x_step, y_step*3), type='h', label='')
    G.add_node('h3_2', pos=(0, y_step*3), type='h', label='')
    G.add_node('h3_3', pos=(x_step, y_step*3), type='h', label='')

    # Layer h^(2)
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
    node_types = nx.get_node_attributes(G, 'type')

    center_x, center_y = 0, 0

    for node in G.nodes():
        node_type = node_types[node]
        fill_color = COLOR_HIDDEN_FILL if node_type == 'h' else COLOR_VISIBLE_FILL
        edge_color = COLOR_HIDDEN_EDGE if node_type == 'h' else COLOR_VISIBLE_EDGE

        nodes_collection = nx.draw_networkx_nodes(G, {node: pos[node]}, nodelist=[node], ax=ax,
                               node_color=fill_color, edgecolors=edge_color,
                               node_size=600, linewidths=1.5)
        if node_type == 'v':
            nodes_collection.set_hatch('////')

    # Define Edges based on whiteboard
    rbm_edges = []
    for h3 in ['h3_1', 'h3_2', 'h3_3']:
        for h2 in ['h2_1', 'h2_2', 'h2_3']:
            rbm_edges.append((h2, h3))

    sbn_edges_h2_h1 = []
    for h2 in ['h2_1', 'h2_2', 'h2_3']:
        for h1 in ['h1_1', 'h1_2', 'h1_3']:
            sbn_edges_h2_h1.append((h2, h1))

    sbn_edges_h1_v = []
    for h1 in ['h1_1', 'h1_2', 'h1_3']:
        for v in ['v1', 'v2']:
            sbn_edges_h1_v.append((h1, v))

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=rbm_edges, ax=ax, arrows=False, width=1.0, edge_color=COLOR_EDGE)
    nx.draw_networkx_edges(G, pos, edgelist=sbn_edges_h2_h1 + sbn_edges_h1_v, ax=ax, arrows=True,
                           arrowstyle='-|>', arrowsize=15, width=1.0, edge_color=COLOR_EDGE, node_size=600)

    # Label positions
    w_x = -x_step * 1.6
    h_x = -x_step * 2.0
    b_x = x_step * 1.5

    # Weight matrices
    ax.text(w_x, y_step*2.5, r'$W^{(3)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(w_x, y_step*1.5, r'$W^{(2)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(w_x, y_step*0.5, r'$W^{(1)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')

    # Layers
    ax.text(h_x, y_step*3, r'$h^{(3)}$', fontsize=14, va='center')
    ax.text(h_x, y_step*2, r'$h^{(2)}$', fontsize=14, va='center')
    ax.text(h_x, y_step, r'$h^{(1)}$', fontsize=14, va='center')
    ax.text(h_x, 0, r'$v$', fontsize=14, va='center')

    # Biases
    ax.text(b_x, y_step*3, r'$b^{(3)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(b_x, y_step*2, r'$b^{(2)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(b_x, y_step, r'$b^{(1)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')
    ax.text(b_x, 0, r'$b^{(0)}$', fontsize=12, color=COLOR_TEXT_BLUE, va='center')

    # Add Generative Process long downward arrow on the right side
    gen_arrow_x = x_step * 2.5
    ax.annotate('', xy=(gen_arrow_x, 0), xytext=(gen_arrow_x, y_step*3),
                arrowprops=dict(arrowstyle='simple', color=COLOR_HIGHLIGHT_EDGE, lw=1))
    ax.text(gen_arrow_x + 0.2, y_step*2.5, "Generative\nProcess", fontsize=14, color=COLOR_HIGHLIGHT_EDGE, va='center')

    # Add the "explain away" concept at the top right
    ax.text(x_step*2.0, y_step*3.5, "explain away", fontsize=16, color=COLOR_HIGHLIGHT_EDGE, style='italic')

    # Add formulas mapping to the "q(h|v)" approximation and DBN bounds loose discussion below the graph
    bottom_y = -1.2

    # Text for q(h|v)
    q_formula = r'$q(h^{(1)}|v) = \prod_{i} q(h_i^{(1)}|v) = \prod_{i} \text{sigmoid}(W_{:,i}^{(1)} \cdot v + b_i^{(1)})$'
    ax.text(0, bottom_y, q_formula, fontsize=14, color=COLOR_HIGHLIGHT_EDGE, ha='center')

    # Text for loose ELBO
    loose_text = r'$DBN \rightarrow ELBO \text{ is relatively loose.}$'
    ax.text(0, bottom_y - 0.7, loose_text, fontsize=14, color=COLOR_HIGHLIGHT_EDGE, ha='center')

    ax.set_title("Generation vs. Inference in DBNs", y=1.05, fontsize=16, fontweight='bold')

    ax.set_xlim(-x_step*2.5, x_step*4.5)
    ax.set_ylim(bottom_y - 1.2, y_step*3.8)
    ax.axis('off')

    os.makedirs('notes/chapters/assets', exist_ok=True)
    output_path = 'notes/chapters/assets/ch27_generation_inference.png'

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    create_generation_diagram()
