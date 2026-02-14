
import graphviz
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_moral_graph_diagram():
    # We will create one graph with 3 clusters for the 3 cases
    dot = graphviz.Digraph(comment='Moral Graph Examples', format='png')
    dot.attr(rankdir='LR')
    dot.attr(bgcolor='transparent')
    dot.attr(compound='true')

    # Common styles
    dot.attr('node', shape='circle', style='filled', fillcolor='#e6f3ff', color='#0066cc', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica')

    # Case 1: Chain (No extra edges needed)
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='(a) Chain (Head-to-Tail)\nNo Extra Edge', fontname='Helvetica', fontsize='14', color='transparent')
        c.node('a1', 'A')
        c.node('b1', 'B')
        c.node('c1', 'C')
        # Draw directed edges (solid)
        c.edge('a1', 'b1')
        c.edge('b1', 'c1')

        # Draw moralized (undirected) appearance?
        # Actually, let's just show the directed graph and imply the moralization is trivial.
        # But to be clear, maybe we show the resulting Undirected edges locally?
        # Let's just show the standard directed graph, and maybe 'dashed' undirected edges if strictly needed.
        # For Chain, existing edges just become undirected.

    # Case 2: Common Parent (Tail-to-Tail)
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='(b) Common Parent (Tail-to-Tail)\nNo Moral Edge Needed', fontname='Helvetica', fontsize='14', color='transparent')
        c.node('a2', 'A') # The Parent
        c.node('b2', 'B')
        c.node('c2', 'C')
        # Edges from A to B and C
        c.edge('a2', 'b2')
        c.edge('a2', 'c2')

    # Case 3: V-Structure (Head-to-Head) - NEEDS MORALIZATION
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='(c) V-Structure (Head-to-Head)\nAdd Moral Edge!', fontname='Helvetica', fontsize='14', color='transparent')
        # V-structure is A -> C <- B
        c.node('a3', 'A') # Parent 1
        c.node('b3', 'B') # Parent 2
        c.node('c3', 'C') # Common Child (Collider)

        c.edge('a3', 'c3')
        c.edge('b3', 'c3')

        # The Moral Edge (Marrying Parents A and B)
        c.edge('a3', 'b3', style='dashed', color='red', label='Moral Edge', constraint='false', dir='none')

    output_path = os.path.join(output_dir, 'ch09_moral_graph_examples')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

if __name__ == "__main__":
    create_moral_graph_diagram()
