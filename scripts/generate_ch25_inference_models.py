import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patches as mpatches
import os

# Aligning with ch21 styling
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.linewidth": 0
})

# Project Colors
COLOR_HIDDEN_FILL = '#e8f5e9'  # Light Green
COLOR_HIDDEN_EDGE = '#2e7d32'  # Dark Green
COLOR_VISIBLE_FILL = '#e3f2fd' # Light Blue
COLOR_VISIBLE_EDGE = '#1565c0' # Dark Blue
COLOR_EDGE = '#9e9e9e'

def draw_graph(ax, G, pos, title, directed=False, node_types=None):
    if node_types is None:
        node_types = ['h' for _ in G.nodes()]

    node_colors = [COLOR_HIDDEN_FILL if t == 'h' else COLOR_VISIBLE_FILL for t in node_types]
    edge_colors = [COLOR_HIDDEN_EDGE if t == 'h' else COLOR_VISIBLE_EDGE for t in node_types]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           edgecolors=edge_colors, node_size=1200, linewidths=2.5)

    # Draw edges
    if directed:
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>',
                               arrowsize=20, width=1.5, edge_color=COLOR_EDGE,
                               node_size=1200)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, edge_color=COLOR_EDGE)

    ax.set_title(title, y=-0.15, fontsize=16, fontweight='bold', color='#212121', pad=20)
    ax.axis('off')

    # Layer Annotations
    y_values = sorted(list(set([p[1] for p in pos.values()])))
    min_x = min([p[0] for p in pos.values()])

    # Add text labels 'h' and 'v'
    if title.startswith("Boltzmann Machine") and len(y_values) > 2:
        ax.text(min_x - 0.5, 2.0, r'$\mathbf{h}$', fontsize=20, va='center', color=COLOR_HIDDEN_EDGE)
        ax.text(min_x - 0.5, 0.0, r'$\mathbf{v}$', fontsize=20, va='center', color=COLOR_VISIBLE_EDGE)
    elif len(y_values) > 1:
        for y in y_values:
            if title.startswith("Deep Boltzmann Machine"):
                if y == max(y_values):
                    label = r"$\mathbf{h}^{(2)}$"
                    c = COLOR_HIDDEN_EDGE
                elif y == min(y_values):
                    label = r"$\mathbf{v}$"
                    c = COLOR_VISIBLE_EDGE
                else:
                    label = r"$\mathbf{h}^{(1)}$"
                    c = COLOR_HIDDEN_EDGE
            else:
                if y == max(y_values):
                    label = r"$\mathbf{h}$"
                    c = COLOR_HIDDEN_EDGE
                else:
                    label = r"$\mathbf{v}$"
                    c = COLOR_VISIBLE_EDGE
            ax.text(min_x - 0.5, y, label, fontsize=20, va='center', color=c)


fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes_flat = axes.flatten()

# --------------------------
# 1. Boltzmann Machine
# --------------------------
G1 = nx.Graph()
G1.add_nodes_from([1, 2, 3, 4, 5])
pos1 = {
    1: (-1, 1.5), 2: (0, 2.8), 3: (1, 1.5),  # hidden
    4: (-0.6, 0), 5: (0.6, 0)                # visible
}
G1.add_edges_from([(1,2), (2,3), (1,3), (1,4), (2,4), (2,5), (3,5), (4,5), (3,4), (1,5)])
types1 = ['h', 'h', 'h', 'v', 'v']
draw_graph(axes_flat[0], G1, pos1, "Boltzmann Machine\n(General Undirected)", node_types=types1)

# --------------------------
# 2. RBM
# --------------------------
G2 = nx.Graph()
h_nodes2 = [1, 2, 3]
v_nodes2 = [4, 5, 6]
G2.add_nodes_from(h_nodes2 + v_nodes2)
pos2 = {1: (-1, 2), 2: (0, 2), 3: (1, 2), 4: (-1, 0), 5: (0, 0), 6: (1, 0)}
for h in h_nodes2:
    for v in v_nodes2:
        G2.add_edge(h, v)
types2 = ['h', 'h', 'h', 'v', 'v', 'v']
draw_graph(axes_flat[1], G2, pos2, "Restricted Boltzmann Machine\n(No Intra-layer Edges)", node_types=types2)

# --------------------------
# 3. Deep Boltzmann Machine
# --------------------------
G3 = nx.Graph()
h2_nodes3 = [1, 2, 3]
h1_nodes3 = [4, 5, 6]
v_nodes3 = [7, 8, 9]
G3.add_nodes_from(h2_nodes3 + h1_nodes3 + v_nodes3)
pos3 = {
    1: (-1, 4), 2: (0, 4), 3: (1, 4),
    4: (-1, 2), 5: (0, 2), 6: (1, 2),
    7: (-1, 0), 8: (0, 0), 9: (1, 0)
}
for top in h2_nodes3:
    for mid in h1_nodes3:
        G3.add_edge(top, mid)
for mid in h1_nodes3:
    for bot in v_nodes3:
        G3.add_edge(mid, bot)
types3 = ['h']*6 + ['v']*3
draw_graph(axes_flat[2], G3, pos3, "Deep Boltzmann Machine\n(Multi-layer Undirected)", node_types=types3)

# --------------------------
# 4. Sigmoid Belief Network
# --------------------------
G4 = nx.DiGraph()
h_nodes4 = [1, 2, 3]
v_nodes4 = [4, 5, 6]
G4.add_nodes_from(h_nodes4 + v_nodes4)
pos4 = {1: (-1, 2), 2: (0, 2), 3: (1, 2), 4: (-1, 0), 5: (0, 0), 6: (1, 0)}
for h in h_nodes4:
    for v in v_nodes4:
        G4.add_edge(h, v)
types4 = ['h', 'h', 'h', 'v', 'v', 'v']
draw_graph(axes_flat[3], G4, pos4, "Sigmoid Belief Network\n(Directed, Explain Away)", directed=True, node_types=types4)

# Legend
hidden_patch = mpatches.Patch(facecolor=COLOR_HIDDEN_FILL, edgecolor=COLOR_HIDDEN_EDGE, linewidth=2, label=r'Hidden Variables ($\mathbf{h}$)')
visible_patch = mpatches.Patch(facecolor=COLOR_VISIBLE_FILL, edgecolor=COLOR_VISIBLE_EDGE, linewidth=2, label=r'Visible Variables ($\mathbf{v}$)')
fig.legend(handles=[hidden_patch, visible_patch], loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.05), fontsize=16, frameon=True, facecolor='white', edgecolor='#e0e0e0', borderpad=1)

plt.tight_layout(pad=3.0)
plt.subplots_adjust(bottom=0.15)

# Save
os.makedirs('notes/chapters/assets', exist_ok=True)
plt.savefig('notes/chapters/assets/ch25_inference_models.png', bbox_inches='tight', dpi=300, facecolor='white')
print("Beautiful image saved as notes/chapters/assets/ch25_inference_models.png")
