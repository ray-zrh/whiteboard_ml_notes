
import graphviz
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_bp_tree_diagram():
    # Use 'neato' for absolute positioning or 'dot' with ranks
    # 'dot' is usually better for hierarchical (tree) structures
    dot = graphviz.Digraph(comment='Belief Propagation Tree', format='png')

    # Graph attributes
    dot.attr(rankdir='TB') # Top to Bottom
    # Removed splines='curved' because it conflicts with edge labels in dot
    # dot.attr(splines='curved')
    dot.attr(nodesep='1.0', ranksep='1.2') # Increase separation

    # Node styles
    dot.attr('node', shape='circle', style='filled', fillcolor='#f0f0f0',
             fontname='Helvetica', fontsize='16', fixedsize='true', width='0.9', penwidth='1.5', color='#333333')

    # Edge styles
    dot.attr('edge', fontname='Helvetica', fontsize='12', arrowsize='0.8', penwidth='1.5')
    dot.attr(bgcolor='transparent')

    # Nodes
    dot.node('a', 'a', fillcolor='#e6f3ff', color='#0066cc') # Root-like - Light Blue
    dot.node('b', 'b', fillcolor='#fff9e6', color='#ffcc00') # Center - Light Yellow

    # Subgraph for leaves to keep them on same rank
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('c', 'c', fillcolor='#e6ffe6', color='#00cc00') # Leaf - Light Green
        s.node('d', 'd', fillcolor='#e6ffe6', color='#00cc00') # Leaf - Light Green

    # Colors for messages
    # Upward (Collect): Leaves -> Root (Blue-ish)
    # Downward (Distribute): Root -> Leaves (Red/Orange-ish)
    color_collect = "#007acc" # Blue
    color_distribute = "#e67300" # Orange

    # Function to add pair of edges
    def add_message_pair(u, v):
        # u is parent, v is child (e.g. a -> b)

        # Distribute: u -> v (Downward)
        # Using ports to separate edges (e.g., left side vs right side)
        # But simple multiple edges usually work in dot

        dot.edge(u, v,
                 label=f'<<font color="{color_distribute}">m<sub>{u}&rarr;{v}</sub></font>>',
                 color=color_distribute, fontcolor=color_distribute, dir='forward')

        # Collect: v -> u (Upward)
        dot.edge(v, u,
                 label=f'<<font color="{color_collect}">m<sub>{v}&rarr;{u}</sub></font>>',
                 color=color_collect, fontcolor=color_collect, dir='forward')

    # Edges
    # a is top, b is middle, c/d are bottom
    add_message_pair('a', 'b')
    add_message_pair('b', 'c')
    add_message_pair('b', 'd')

    output_path = os.path.join(output_dir, 'ch09_bp_tree_structure')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

if __name__ == "__main__":
    create_bp_tree_diagram()
