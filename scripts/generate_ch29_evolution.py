import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import matplotlib.patches as patches

# Configuration to match project style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.linewidth": 0
})

# Project Colors
COLOR_HIDDEN_FILL = '#ffffff'
COLOR_HIDDEN_EDGE = '#000000'
COLOR_VISIBLE_FILL = '#ffffff'
COLOR_VISIBLE_EDGE = '#000000'
COLOR_EDGE = '#424242'
COLOR_TEXT_BLUE = '#1565c0'

def set_node_attributes(G):
    pass

def draw_model(ax, title, year, model_type):
    ax.set_title(f"{title}\n{year}", color=COLOR_TEXT_BLUE, fontsize=14, fontweight='bold')
    ax.axis('off')

    G = nx.DiGraph()
    pos = {}

    h_nodes = []
    v_nodes = []
    undirected_edges = []
    directed_edges = []

    if model_type == 'GBM':
        # General Boltzmann Machine: 3 visible, 4 hidden. Arranged in a circle.
        v_nodes = ['v1', 'v2', 'v3']
        h_nodes = ['h1', 'h2', 'h3', 'h4']

        pos['v1'] = (-0.8, -0.6)
        pos['v2'] = (0.0, -0.9)
        pos['v3'] = (0.8, -0.6)

        pos['h1'] = (-0.9, 0.2)
        pos['h2'] = (-0.4, 0.9)
        pos['h3'] = (0.4, 0.9)
        pos['h4'] = (0.9, 0.2)

        G.add_nodes_from(v_nodes, type='v')
        G.add_nodes_from(h_nodes, type='h')

        # All connected
        import itertools
        for u, v in itertools.combinations(G.nodes(), 2):
            undirected_edges.append((u, v))

    elif model_type == 'RBM':
        # 3 visible, 3 hidden
        v_nodes = [f'v{i}' for i in range(3)]
        h_nodes = [f'h{i}' for i in range(3)]

        for i, n in enumerate(v_nodes):
            pos[n] = (i - 1, -0.5)
        for i, n in enumerate(h_nodes):
            pos[n] = (i - 1, 0.5)

        G.add_nodes_from(v_nodes, type='v')
        G.add_nodes_from(h_nodes, type='h')

        for v in v_nodes:
            for h in h_nodes:
                undirected_edges.append((v, h))

    elif model_type in ['DBN', 'DBM']:
        # 4 layers: 1 visible (bottom), 3 hidden
        # 3 nodes per layer
        layers = [[f'v{i}' for i in range(3)]]
        layers.append([f'h1_{i}' for i in range(3)])
        layers.append([f'h2_{i}' for i in range(3)])
        layers.append([f'h3_{i}' for i in range(3)])

        v_nodes = layers[0]
        h_nodes = layers[1] + layers[2] + layers[3]

        for l_idx, layer in enumerate(layers):
            y = (l_idx * 1.0) - 1.5
            for i, n in enumerate(layer):
                pos[n] = (i - 1, y)
                if l_idx == 0:
                    G.add_node(n, type='v')
                else:
                    G.add_node(n, type='h')

        if model_type == 'DBN':
            # Top 2 layers undirected
            for u in layers[2]:
                for v in layers[3]:
                    undirected_edges.append((u, v))
            # Lower layers directed downwards
            for u in layers[2]:
                for v in layers[1]:
                    directed_edges.append((u, v))
            for u in layers[1]:
                for v in layers[0]:
                    directed_edges.append((u, v))
        else: # DBM
            # All adjacent layers undirected
            for l in range(3):
                for u in layers[l]:
                    for v in layers[l+1]:
                        undirected_edges.append((u, v))

    # Draw Undirected Edges
    if undirected_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=undirected_edges, width=1.0, edge_color=COLOR_EDGE, alpha=0.6, arrows=False, node_size=800)

    # Draw Directed Edges
    if directed_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=directed_edges, width=1.5, edge_color=COLOR_EDGE, alpha=0.8, arrows=True, arrowstyle='-|>', arrowsize=18, node_size=800)

    # Draw Visible Nodes
    v_collection = nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=v_nodes, node_size=800, node_color=COLOR_VISIBLE_FILL, edgecolors=COLOR_VISIBLE_EDGE)
    v_collection.set_hatch('/////')

    # Draw Hidden Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=h_nodes, node_size=800, node_color=COLOR_HIDDEN_FILL, edgecolors=COLOR_HIDDEN_EDGE)

    # Adjust limits
    if model_type == 'GBM':
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-1.5, 1.5)
    elif model_type == 'RBM':
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-1.5, 1.5)
    else:
        ax.set_ylim(-2.0, 2.0)
        ax.set_xlim(-1.5, 1.5)

def main():
    fig, axs = plt.subplots(2, 2, figsize=(10, 11))

    draw_model(axs[0, 0], "General Boltzmann Machine", "1983", "GBM")
    draw_model(axs[0, 1], "RBM", "1986, 2002", "RBM")
    draw_model(axs[1, 0], "DBN", "2006", "DBN")
    draw_model(axs[1, 1], "DBM", "2008", "DBM")

    plt.tight_layout(pad=3.0)

    os.makedirs('notes/chapters/assets', exist_ok=True)
    out_path = 'notes/chapters/assets/ch29_evolution.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved diagram to {out_path}")

if __name__ == "__main__":
    main()
