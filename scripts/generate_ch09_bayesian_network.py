
import graphviz
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_node_style(dot):
    # Default size, rankdir is set per graph
    dot.attr(size='4,3', ratio='fill', margin='0')
    dot.attr('node', shape='circle', style='filled', fillcolor='white', fontname='Arial', fixedsize='true', width='0.6')
    dot.attr('edge', arrowsize='0.8')

def create_tail_to_tail():
    dot = graphviz.Digraph(comment='Tail-to-Tail', format='png')
    dot.attr(rankdir='TB') # Top to Bottom
    create_node_style(dot)

    # Label is better handled in the markdown caption or text,
    # but I'll keep the graph label for clarity if standalone
    # actually user asked to split according to separate explain.
    # A clean graph is better for inline embedding. Removing internal titles.

    dot.attr(bgcolor='transparent')

    dot.node('a1', 'a', fillcolor='#ffcccc') # a is observed
    dot.node('b1', 'b')
    dot.node('c1', 'c')
    dot.edge('a1', 'b1')
    dot.edge('a1', 'c1')

    output_path = os.path.join(output_dir, 'ch09_bn_tail_to_tail')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

def create_head_to_tail():
    dot = graphviz.Digraph(comment='Head-to-Tail', format='png')
    dot.attr(rankdir='LR') # Left to Right
    create_node_style(dot)
    dot.attr(bgcolor='transparent')

    dot.node('a2', 'a')
    dot.node('b2', 'b', fillcolor='#ffcccc') # b is observed
    dot.node('c2', 'c')
    dot.edge('a2', 'b2')
    dot.edge('b2', 'c2')

    output_path = os.path.join(output_dir, 'ch09_bn_head_to_tail')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

def create_head_to_head():
    dot = graphviz.Digraph(comment='Head-to-Head', format='png')
    dot.attr(rankdir='TB') # Top to Bottom
    create_node_style(dot)
    dot.attr(bgcolor='transparent')

    dot.node('a3', 'a')
    dot.node('c3', 'c') # Parents
    dot.node('b3', 'b') # Child
    dot.node('d3', 'd', fillcolor='#e0e0e0') # Descendant

    dot.edge('a3', 'b3')
    dot.edge('c3', 'b3')
    dot.edge('b3', 'd3')

    output_path = os.path.join(output_dir, 'ch09_bn_head_to_head')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

if __name__ == "__main__":
    create_tail_to_tail()
    create_head_to_tail()
    create_head_to_head()
