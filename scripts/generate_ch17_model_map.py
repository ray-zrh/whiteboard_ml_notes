import os
from graphviz import Digraph

def generate_model_map():
    dot = Digraph(comment='Model Relationship')
    dot.attr(rankdir='LR', dpi='300')
    dot.attr('node', shape='box', style='rounded', fontname='Sans')

    # Hard Classification
    with dot.subgraph(name='cluster_hard') as c:
        c.attr(label='Hard Classification')
        c.node('svm', 'SVM\n(Max Margin)')
        c.node('pla', 'PLA\n(Error Driven)')
        c.node('lda', 'LDA\n(Fisher)')

    # Soft Classification
    with dot.subgraph(name='cluster_soft') as c:
        c.attr(label='Soft Classification')

        # Generative
        with dot.subgraph(name='cluster_gen') as g:
            g.attr(label='Probability Generative\nP(x,y)', style='dashed')
            g.node('nb', 'Naive Bayes')
            g.node('hmm', 'HMM\n(Hidden Markov Model)')
            g.edge('nb', 'hmm', label='+ Sequence')

        # Discriminative
        with dot.subgraph(name='cluster_disc') as d:
            d.attr(label='Probability Discriminative\nP(y|x)', style='dashed')
            d.node('lr', 'Logistic Regression')
            d.node('memm', 'MEMM\n(MaxEnt Markov Model)')
            d.node('crf', 'CRF\n(Conditional Random Field)')

            d.edge('lr', 'memm', label='+ Sequence')
            d.edge('memm', 'crf', label='+ Global Norm\nSolve Label Bias')

    # Connecting main clusters visually (invisible or labelled edges)
    # This part is tricky in dot, sometimes better to just let them flow

    output_dir = os.path.join(os.path.dirname(__file__), '../notes/chapters/assets')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ch17_model_map')

    dot.render(output_path, format='png', cleanup=True)
    print(f"Generated image at {output_path}.png")

if __name__ == '__main__':
    generate_model_map()
