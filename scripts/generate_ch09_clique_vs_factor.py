
import graphviz
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_clique_vs_factor_diagram():
    dot = graphviz.Digraph(comment='Clique vs Factor', format='png')
    dot.attr(rankdir='LR')
    dot.attr(bgcolor='transparent')

    # Subgraph 1: Undirected Clique (Triangle)
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='(a) MAX Clique (Undirected)', fontname='Helvetica', fontsize='14', color='transparent')
        c.attr('node', shape='circle', style='filled', fillcolor='#e6f3ff', color='#0066cc', width='0.5')
        c.node('a1', 'a')
        c.node('b1', 'b')
        c.node('c1', 'c')

        c.attr('edge', dir='none', penwidth='1.5')
        c.edge('a1', 'b1')
        c.edge('b1', 'c1')
        c.edge('c1', 'a1')

    # Subgraph 2: Factor Graph (Single Factor)
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='(b) Factor Graph (Coarse)', fontname='Helvetica', fontsize='14', color='transparent')
        c.attr('node', shape='circle', style='filled', fillcolor='#e6f3ff', color='#0066cc', width='0.5')
        c.node('a2', 'a')
        c.node('b2', 'b')
        c.node('c2', 'c')

        c.attr('node', shape='square', style='filled', fillcolor='#333333', fontcolor='white', width='0.4')
        c.node('f_abc', 'f')

        c.attr('edge', dir='none', penwidth='1.5')
        c.edge('a2', 'f_abc')
        c.edge('b2', 'f_abc')
        c.edge('c2', 'f_abc')

    # Subgraph 3: Factor Graph (Pairwise)
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='(c) Factor Graph (Fine)', fontname='Helvetica', fontsize='14', color='transparent')
        c.attr('node', shape='circle', style='filled', fillcolor='#e6f3ff', color='#0066cc', width='0.5')
        c.node('a3', 'a')
        c.node('b3', 'b')
        c.node('c3', 'c')

        c.attr('node', shape='square', style='filled', fillcolor='#333333', fontcolor='white', width='0.3')
        c.node('f_ab', 'f1')
        c.node('f_bc', 'f2')
        c.node('f_ac', 'f3')

        c.attr('edge', dir='none', penwidth='1.5')
        c.edge('a3', 'f_ab')
        c.edge('b3', 'f_ab')

        c.edge('b3', 'f_bc')
        c.edge('c3', 'f_bc')

        c.edge('a3', 'f_ac')
        c.edge('c3', 'f_ac')

    output_path = os.path.join(output_dir, 'ch09_clique_vs_factor')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")

if __name__ == "__main__":
    create_clique_vs_factor_diagram()
