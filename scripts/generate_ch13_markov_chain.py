
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_markov_chain_diagram():
    # Style settings
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12
    })

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Node positions
    # X1, X2, X3, ..., Xt(i), Xt+1(j), ...
    nodes = [
        {'pos': (1.5, 2.5), 'label': r'$X_1$', 'dist': r'$\pi_1$'},
        {'pos': (3.5, 2.5), 'label': r'$X_2$', 'dist': r'$\pi_2$'},
        {'pos': (5.5, 2.5), 'label': r'$X_3$', 'dist': r'$\pi_3$'},
        {'pos': (7.2, 2.5), 'label': r'$\dots$', 'dist': None, 'is_dots': True},
        {'pos': (9.0, 2.5), 'label': r'$X_t=i$', 'dist': r'$\pi_t$'},
        {'pos': (11.5, 2.5), 'label': r'$X_{t+1}=j$', 'dist': r'$\pi_{t+1}$'},
        {'pos': (13.5, 2.5), 'label': r'$\dots$', 'dist': None, 'is_dots': True}
    ]

    radius = 0.7

    for k, node in enumerate(nodes):
        x, y = node['pos']

        if node.get('is_dots'):
            ax.text(x, y, node['label'], ha='center', va='center', fontsize=20)
            continue

        # Style
        # Default: Light Blue (concept map root style)
        facecolor = '#e6f3ff'
        edgecolor = '#0066cc'
        linewidth = 2

        # Highlight: Light Yellow/Orange for i, j (concept map branch style)
        if 'i' in node['label'] or 'j' in node['label']:
            facecolor = '#fff9e6'
            edgecolor = '#ffcc00'
            linewidth = 2.5

        circle = patches.Circle((x, y), radius, facecolor=facecolor, edgecolor=edgecolor, lw=linewidth, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, node['label'], ha='center', va='center', fontsize=14, zorder=11)

        # Draw downward arrow for distribution
        if node['dist']:
            ax.annotate("", xy=(x, y - 1.5), xytext=(x, y - radius),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color='#555555', shrinkA=5, shrinkB=5))
            ax.text(x, y - 1.8, node['dist'], ha='center', va='top', fontsize=16, color='#333333')

    # Draw Arrows between nodes
    for i in range(len(nodes) - 1):
        curr_node = nodes[i]
        next_node = nodes[i+1]

        x1, y1 = curr_node['pos']
        x2, y2 = next_node['pos']

        # Adjust start/end for radius
        start_x = x1 + (radius if not curr_node.get('is_dots') else 0.2)
        end_x = x2 - (radius if not next_node.get('is_dots') else 0.2)

        if start_x >= end_x: continue

        arrow_color = '#333333'

        # Label Pij
        label = ""
        if 'i' in curr_node['label'] and 'j' in next_node['label']:
            label = r'$P_{ij}$'
            # Highlight this arrow?
            arrow_color = '#d62728' # Red

        ax.annotate("", xy=(end_x, y2), xytext=(start_x, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color=arrow_color))

        if label:
            ax.text((start_x + end_x)/2, y1 + 0.3, label, ha='center', va='bottom', fontsize=14, color=arrow_color)

    output_path = os.path.join(output_dir, 'ch13_markov_chain.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_markov_chain_diagram()
