
import graphviz
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_mrf_global_markov():
    dot = graphviz.Graph(comment='Global Markov Property MRF', format='png')
    dot.attr(rankdir='LR', size='5,3', ratio='fill', margin='0')
    dot.attr('node', shape='circle', style='filled', fillcolor='white', fontname='Times-Roman', fixedsize='true', width='0.6')
    dot.attr('edge', len='1.5')
    dot.attr(bgcolor='transparent')

    # X_A nodes
    with dot.subgraph(name='cluster_A') as c:
        c.attr(label='X_A', style='dashed', color='blue')
        c.node('a1', '<<i>x</i><sub>a1</sub>>')
        c.node('a2', '<<i>x</i><sub>a2</sub>>')

    # X_B nodes (observed/separator)
    with dot.subgraph(name='cluster_B') as c:
        c.attr(label='X_B', style='filled', color='lightgrey')
        c.node('b1', '<<i>x</i><sub>b1</sub>>', fillcolor='lightgrey')
        c.node('b2', '<<i>x</i><sub>b2</sub>>', fillcolor='lightgrey')

    # X_C nodes
    with dot.subgraph(name='cluster_C') as c:
        c.attr(label='X_C', style='dashed', color='blue')
        c.node('c1', '<<i>x</i><sub>c1</sub>>')
        c.node('c2', '<<i>x</i><sub>c2</sub>>')

    # Edges: A - B - C
    dot.edge('a1', 'b1')
    dot.edge('a2', 'b1')
    dot.edge('a1', 'b2')

    dot.edge('b1', 'c1')
    dot.edge('b2', 'c2')
    dot.edge('b1', 'c2')

    output_path = os.path.join(output_dir, 'ch09_mrf_global_markov')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

def create_mrf_local_markov():
    # Node independent of rest given neighbors
    dot = graphviz.Graph(comment='Local Markov Property MRF', format='png')
    dot.attr(layout='neato', size='4,4', ratio='fill', margin='0')
    dot.attr('node', shape='circle', style='filled', fillcolor='white', fontname='Times-Roman', fixedsize='true', width='0.6')
    dot.attr(bgcolor='transparent')

    # Central node
    dot.node('i', '<<i>x</i><sub>i</sub>>', pos='0,0!')

    # Neighbors (Markov Blanket)
    dot.node('n1', 'n1', fillcolor='lightgrey', pos='1,1!')
    dot.node('n2', 'n2', fillcolor='lightgrey', pos='1,-1!')
    dot.node('n3', 'n3', fillcolor='lightgrey', pos='-1.2,0!')

    # Rest of graph
    dot.node('r1', 'r1', pos='2,2!')
    dot.node('r2', 'r2', pos='-2,-1!')

    # Edges
    dot.edge('i', 'n1')
    dot.edge('i', 'n2')
    dot.edge('i', 'n3')

    dot.edge('n1', 'r1')
    dot.edge('n3', 'r2')
    dot.edge('n1', 'n2') # Neighbors can be connected

    output_path = os.path.join(output_dir, 'ch09_mrf_local_markov')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

def create_mrf_pairwise_markov():
    # Non-adjacent nodes independent given all others
    dot = graphviz.Graph(comment='Pairwise Markov Property MRF', format='png')
    dot.attr(layout='neato', size='5,3', ratio='fill', margin='0')
    dot.attr('node', shape='circle', style='filled', fillcolor='white', fontname='Times-Roman', fixedsize='true', width='0.6')
    dot.attr(bgcolor='transparent')

    dot.node('i', '<<i>x</i><sub>i</sub>>', pos='-2,0!')
    dot.node('j', '<<i>x</i><sub>j</sub>>', pos='2,0!')

    # Others
    dot.node('o1', 'o1', fillcolor='lightgrey', pos='0,1!')
    dot.node('o2', 'o2', fillcolor='lightgrey', pos='0,-1!')

    # Edges (i and j not connected directly)
    dot.edge('i', 'o1')
    dot.edge('i', 'o2')
    dot.edge('j', 'o1')
    dot.edge('j', 'o2')
    dot.edge('o1', 'o2')

    # Dashed line indicating i not connected to j
    dot.edge('i', 'j', style='invis')

    output_path = os.path.join(output_dir, 'ch09_mrf_pairwise_markov')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

def create_mrf_clique():
    # Factorization: Maximal Clique vs Clique
    dot = graphviz.Graph(comment='Factorization MRF', format='png')
    dot.attr(layout='neato', size='5,4', ratio='fill', margin='0')
    dot.attr('node', shape='circle', style='filled', fillcolor='white', fontname='Times-Roman', fixedsize='true', width='0.6')
    dot.attr(bgcolor='transparent')

    # Triangle a-b-c (Maximal Clique)
    dot.node('a', 'a', pos='0,2!')
    dot.node('b', 'b', pos='-1.5,0!')
    dot.node('c', 'c', pos='1.5,0!')

    dot.edge('a', 'b')
    dot.edge('b', 'c')
    dot.edge('c', 'a')

    # d connected to a (Pair is a maximal clique if not fully connected to others)
    dot.node('d', 'd', pos='0,3.5!')
    dot.edge('a', 'd')

    output_path = os.path.join(output_dir, 'ch09_mrf_factorization')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

if __name__ == "__main__":
    create_mrf_global_markov()
    create_mrf_local_markov()
    create_mrf_pairwise_markov()
    create_mrf_clique()
