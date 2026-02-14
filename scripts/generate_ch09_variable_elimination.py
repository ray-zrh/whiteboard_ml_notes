
import graphviz
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_ve_chain_diagram():
    dot = graphviz.Digraph(comment='Variable Elimination Chain', format='png')
    dot.attr(rankdir='LR', size='5,2', ratio='fill', margin='0')
    dot.attr('node', shape='circle', style='filled', fillcolor='white', fontname='Times-Roman', fixedsize='true', width='0.6')
    dot.attr('edge', arrowsize='0.8')
    dot.attr(bgcolor='transparent')

    # Nodes a, b, c, d
    dot.node('a', '<<i>a</i>>')
    dot.node('b', '<<i>b</i>>')
    dot.node('c', '<<i>c</i>>')
    dot.node('d', '<<i>d</i>>', fillcolor='#ffcccc') # Target node d highlighted? Or query node.

    # Edges
    dot.edge('a', 'b')
    dot.edge('b', 'c')
    dot.edge('c', 'd')

    output_path = os.path.join(output_dir, 'ch09_variable_elimination')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

if __name__ == "__main__":
    create_ve_chain_diagram()
