
import graphviz
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_global_markov_property():
    dot = graphviz.Digraph(comment='Global Markov Property', format='png')
    dot.attr(rankdir='LR', size='6,4', ratio='fill', margin='0')
    dot.attr('node', shape='circle', style='filled', fillcolor='white', fontname='Arial', fixedsize='true', width='0.6')
    dot.attr('edge', arrowsize='0.8')
    dot.attr(bgcolor='transparent')

    # Clusters for X_A, X_B, X_C
    # X_B separates X_A and X_C

    with dot.subgraph(name='cluster_A') as c:
        c.attr(label='X_A', style='rounded', color='black')
        c.node('a1', 'a1')
        c.node('a2', 'a2')

    with dot.subgraph(name='cluster_B') as c:
        c.attr(label='X_B (Observed)', style='filled', color='lightgrey', fillcolor='#ffcccc')
        c.node('b1', 'b1', fillcolor='#ffcccc')
        c.node('b2', 'b2', fillcolor='#ffcccc')

    with dot.subgraph(name='cluster_C') as c:
        c.attr(label='X_C', style='rounded', color='black')
        c.node('c1', 'c1')
        c.node('c2', 'c2')

    # Edges flowing from A -> B -> C (blocking path)
    dot.edge('a1', 'b1')
    dot.edge('a2', 'b2')
    dot.edge('b1', 'c1')
    dot.edge('b2', 'c2')
    # Use constraint=false for some edges to allow flexible layout if needed, but LR usually handles this well.

    output_path = os.path.join(output_dir, 'ch09_global_markov_property')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

def create_markov_blanket():
    dot = graphviz.Digraph(comment='Markov Blanket', format='png')
    dot.attr(layout='neato') # Use neato for more organic layout
    dot.attr(overlap='false')
    dot.attr('node', shape='circle', style='filled', fillcolor='white', fontname='Arial', fixedsize='true', width='0.7')
    dot.attr(bgcolor='transparent')

    # Central Node
    dot.node('xi', 'xi', fillcolor='lightblue', pos='0,0!')

    # Parents
    dot.node('p1', 'Pa1', fillcolor='#ffe6cc', pos='-1.5,1.5!')
    dot.node('p2', 'Pa2', fillcolor='#ffe6cc', pos='1.5,1.5!')

    # Children
    dot.node('c1', 'Ch1', fillcolor='#e6ffcc', pos='-1.5,-1.5!')
    dot.node('c2', 'Ch2', fillcolor='#e6ffcc', pos='1.5,-1.5!')

    # Co-parents (Parents of Children)
    dot.node('cp1', 'CoP1', fillcolor='#e6ccff', pos='-2.5,-0.5!') # Parent of Ch1
    dot.node('cp2', 'CoP2', fillcolor='#e6ccff', pos='2.5,-0.5!') # Parent of Ch2

    # Edges
    dot.edge('p1', 'xi')
    dot.edge('p2', 'xi')

    dot.edge('xi', 'c1')
    dot.edge('xi', 'c2')

    dot.edge('cp1', 'c1')
    dot.edge('cp2', 'c2')

    # Circle concept (conceptually) - hard to draw actual circle around them with graphviz without clusters
    # But coloring helps identify the blanket group.

    output_path = os.path.join(output_dir, 'ch09_markov_blanket')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")


if __name__ == "__main__":
    create_global_markov_property()
    create_markov_blanket()
