import os
from graphviz import Digraph

def generate_tree():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'notes', 'chapters', 'assets')
    os.makedirs(output_dir, exist_ok=True)

    dot = Digraph(comment='Deep Generative Models Taxonomy', format='png')
    dot.attr(rankdir='LR', size='10,6')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Arial')

    # Root
    dot.node('ML', 'Maximum Likelihood')

    # Level 1
    dot.node('LB', 'Likelihood-based\n(explicit density)')
    dot.node('LF', 'Likelihood-free\n(implicit density)')
    dot.edge('ML', 'LB')
    dot.edge('ML', 'LF')

    # Level 2 (LB)
    dot.node('Tract', 'tractable')
    dot.node('App', 'approximate loop/inference')
    dot.edge('LB', 'Tract')
    dot.edge('LB', 'App')

    # Level 3 (Tractable)
    dot.node('FO', 'Fully observed\n(Autoregressive Model)')
    dot.node('CV', 'Change of Variable\nNon-linear ICA\n(Flow-based Model)')
    dot.edge('Tract', 'FO')
    dot.edge('Tract', 'CV')

    # Level 3 (Approximate)
    dot.node('Var', 'Variational\n(VAE)')
    dot.node('MC_app', 'MC\n(Energy-based)')
    dot.edge('App', 'Var')
    dot.edge('App', 'MC_app')

    # Level 2 (LF)
    dot.node('Dir', 'Direct\n(GAN)')
    dot.node('MC_lf', 'MC\n(GSN)')
    dot.edge('LF', 'Dir')
    dot.edge('LF', 'MC_lf')

    # Save
    output_path = os.path.join(output_dir, 'ch30_taxonomy_tree')
    dot.render(output_path, cleanup=True)
    print(f"Graph generated at {output_path}.png")

if __name__ == '__main__':
    generate_tree()
