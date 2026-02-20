import os
from graphviz import Digraph

def generate_ml_taxonomy():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, '../notes/chapters/assets')
    os.makedirs(assets_dir, exist_ok=True)

    dot = Digraph('ML_Taxonomy', format='png')
    dot.attr(rankdir='LR', dpi='300')
    dot.attr('node', shape='box', style='rounded', color='#4A90E2', fontname='Arial', fontsize='12')
    dot.attr('edge', color='#999999', arrowsize='0.7')

    dot.node('ML', 'Machine Learning')

    # Two Schools
    dot.node('Freq', 'Frequency\n(Frequentist)\n|\nStatistical\nLearning')
    dot.node('Bayes', 'Bayesian\n|\nPGM')

    dot.edge('ML', 'Freq')
    dot.edge('ML', 'Bayes')

    # Frequentist Branches
    dot.node('Reg', 'Regularization')
    dot.node('Kernel', 'Kernelization:\nKernel SVM')
    dot.node('Ensemble', 'Ensemble:\nAdaBoost, RF')
    dot.node('NN', 'Layering: NN\n(MLP, Autoencoder,\nCNN, RNN)')

    dot.edge('Freq', 'Reg')
    dot.edge('Freq', 'Kernel')
    dot.edge('Freq', 'Ensemble')
    dot.edge('Freq', 'NN')

    dot.node('DNN', 'Deep Neural\nNetwork')
    dot.edge('NN', 'DNN')

    # Bayesian Branches
    dot.node('Dir', 'Directed:\nBayesian Net')
    dot.node('Undir', 'Undirected:\nMarkov Net')
    dot.node('Mixed', 'Mixed:\nMixed Net')

    dot.edge('Bayes', 'Dir')
    dot.edge('Bayes', 'Undir')
    dot.edge('Bayes', 'Mixed')

    dot.node('DDN', 'Deep Directed\nNetwork (VAE, GAN)')
    dot.node('DBM', 'Deep Boltzmann\nMachine')
    dot.node('DBN', 'Deep Belief\nNetwork')

    dot.edge('Dir', 'DDN')
    dot.edge('Undir', 'DBM')
    dot.edge('Mixed', 'DBN')

    dot.node('DGM', 'Deep Generative\nModel')
    dot.edge('DDN', 'DGM')
    dot.edge('DBM', 'DGM')
    dot.edge('DBN', 'DGM')

    dot.node('DL', 'Deep Learning', shape='ellipse', style='filled', fillcolor='#D4EDDA')
    dot.edge('DNN', 'DL')
    dot.edge('DGM', 'DL')

    # MLP highlight
    dot.node('MLP', 'Multi-layer\nPerceptron', shape='plaintext', fontcolor='#D0021B')
    dot.edge('NN', 'MLP', style='dashed', dir='none')

    output_path = os.path.join(assets_dir, 'ch23_ml_taxonomy')
    dot.render(output_path, cleanup=True)
    print(f"Graph generated at {output_path}.png")

if __name__ == "__main__":
    generate_ml_taxonomy()
