import networkx as nx
import matplotlib.pyplot as plt

def create_naive_bayes_pgm():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    # y is the parent class variable
    G.add_node('y', pos=(0, 1))

    # x1 to xp are the feature variables
    # We'll show x1, x2, ..., xp
    G.add_node('x1', pos=(-2, 0))
    G.add_node('x2', pos=(-1, 0))
    G.add_node('...', pos=(0, 0))
    G.add_node('xp', pos=(2, 0))

    # Add edges from y to all xi
    G.add_edge('y', 'x1')
    G.add_edge('y', 'x2')
    G.add_edge('y', '...')
    G.add_edge('y', 'xp')

    # Get positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw graph
    plt.figure(figsize=(8, 5))

    # Draw nodes
    # y node (shaded/highlighted usually, but here we just make it distinct)
    nx.draw_networkx_nodes(G, pos, nodelist=['y'], node_color='#FFD700', node_size=2000, edgecolors='black') # Gold for y
    nx.draw_networkx_nodes(G, pos, nodelist=['x1', 'x2', '...', 'xp'], node_color='lightgrey', node_size=1500, edgecolors='black')

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20, width=1.5)

    # Draw labels
    # Use standard notation labels
    labels = {
        'y': r'$y$',
        'x1': r'$x_1$',
        'x2': r'$x_2$',
        '...': r'$\dots$',
        'xp': r'$x_p$'
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=16)

    # Annotations for distribution types (optional but helpful context from screenshot)
    plt.text(0.7, 1, r'$y \in \{0, 1\}$', fontsize=12, ha='left')
    plt.text(0.7, 0.85, r'$x \in \mathbb{R}^p$', fontsize=12, ha='left')

    plt.title("Naive Bayes: Probabilistic Graphical Model", fontsize=15)
    plt.axis('off')

    # Save the plot
    output_path = 'notes/chapters/assets/ch04_naive_bayes_pgm.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    create_naive_bayes_pgm()
