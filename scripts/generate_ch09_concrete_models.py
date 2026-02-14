
import graphviz
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_naive_bayes_diagram():
    dot = graphviz.Digraph(comment='Naive Bayes', format='png')
    dot.attr(rankdir='TB', size='6,4', ratio='fill', margin='0')
    dot.attr('node', shape='circle', style='filled', fillcolor='white', fontname='Times-Roman', fixedsize='true', width='0.7')
    dot.attr('edge', arrowsize='0.8')
    dot.attr(bgcolor='transparent')

    # Latent Variable y (Class)
    # Use HTML label for italic y
    dot.node('y', '<<i>y</i>>', fillcolor='#ffcccc')

    # Features x1...xp
    with dot.subgraph(name='cluster_features') as c:
        c.attr(style='invis')
        # Use HTML labels for subscripts
        c.node('x1', '<<i>x</i><sub>1</sub>>', fillcolor='lightgrey')
        c.node('x2', '<<i>x</i><sub>2</sub>>', fillcolor='lightgrey')
        c.node('xp', '<<i>x</i><sub>p</sub>>', fillcolor='lightgrey')

    dot.edge('y', 'x1')
    dot.edge('y', 'x2')
    dot.edge('y', 'xp')

    # Ellipsis with Math font if possible
    dot.node('dots', '...', shape='none', fillcolor='transparent', width='0.5', fontname='Times-Roman')
    dot.edge('y', 'dots', style='invis')

    output_path = os.path.join(output_dir, 'ch09_naive_bayes_struc')
    dot.render(output_path, cleanup=True)
    print(f"Saved {output_path}.png")


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Ensure covariance is numpy array
    covariance = np.array(covariance)

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw multiple confidence intervals
    for nsig in range(1, 4):
        alpha = kwargs.pop('alpha', 0.2) / nsig # Fade out outer ellipses
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle=angle, alpha=alpha, **kwargs))

def create_gmm_concept_diagram():
    """
    Creates a combined diagram for GMM:
    1. A scatter plot showing clusters + Ellipses for Gaussians (Beautiful style)
    2. A PGM structure z -> x.
    """
    # Use a style for better aesthetics if available, otherwise default
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass # Fallback to default

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Generate Data
    np.random.seed(42)

    # Cluster 1
    mean1 = [2, 3]
    cov1 = [[0.6, 0.4], [0.4, 0.6]]
    data1 = np.random.multivariate_normal(mean1, cov1, 50)

    # Cluster 2
    mean2 = [6, 6]
    cov2 = [[0.8, -0.5], [-0.5, 0.8]]
    data2 = np.random.multivariate_normal(mean2, cov2, 50)

    # Cluster 3
    mean3 = [7, 2]
    cov3 = [[0.5, 0], [0, 0.5]]
    data3 = np.random.multivariate_normal(mean3, cov3, 50)

    # 2. Draw Scatter Plot & Ellipses
    # Cluster 1
    ax.scatter(data1[:, 0], data1[:, 1], s=20, color='tab:blue', alpha=0.6, label='Cluster 1')
    draw_ellipse(mean1, cov1, ax=ax, color='tab:blue')

    # Cluster 2
    ax.scatter(data2[:, 0], data2[:, 1], s=20, color='tab:orange', alpha=0.6, label='Cluster 2')
    draw_ellipse(mean2, cov2, ax=ax, color='tab:orange')

    # Cluster 3
    ax.scatter(data3[:, 0], data3[:, 1], s=20, color='tab:green', alpha=0.6, label='Cluster 3')
    draw_ellipse(mean3, cov3, ax=ax, color='tab:green')

    # Axes Arrows (Manually drawn to look like whiteboard)
    ax.arrow(0, 0, 9, 0, head_width=0.3, head_length=0.3, fc='k', ec='k', zorder=10)
    ax.arrow(0, 0, 0, 8, head_width=0.3, head_length=0.3, fc='k', ec='k', zorder=10)
    ax.text(9.2, 0, 'Dim 1', ha='left', va='center')
    ax.text(0, 8.2, 'Dim 2', ha='center', va='bottom')

    # Label "GMM"
    ax.text(8, 8, 'GMM', fontsize=24, fontfamily='sans-serif', fontweight='bold', color='#333333')

    # 3. PGM Structure (Bottom Right area, overlapping or distinct)
    # Define positions
    z_pos = (8.5, 4.0)
    x_pos = (8.5, 2.0)

    # Function to draw a node
    def draw_node(pos, text, observed=False, radius=0.4):
        circle = plt.Circle(pos, radius, facecolor='lightgrey' if observed else 'white', edgecolor='k', zorder=20, lw=1.5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=18, fontstyle='italic', fontfamily='serif', zorder=21)

    # Draw PGM nodes
    draw_node(z_pos, 'z', observed=False)
    draw_node(x_pos, 'x', observed=True)

    # Draw Arrow z -> x
    ax.annotate("", xy=(x_pos[0], x_pos[1]+0.4), xytext=(z_pos[0], z_pos[1]-0.4),
                arrowprops=dict(arrowstyle="->", lw=2, color='black', shrinkA=0, shrinkB=0))

    # Add Text: "z is discrete z=1,2,...,K"
    ax.text(z_pos[0] + 0.6, z_pos[1], r'$z$ is discrete', fontsize=14, color='darkblue', ha='left')
    ax.text(z_pos[0] + 0.6, z_pos[1] - 0.5, r'$z \in \{1, \dots, K\}$', fontsize=14, color='black', ha='left')

    # Adjust Plot Limits & Style
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.axis('off')

    output_path = os.path.join(output_dir, 'ch09_gmm_struc.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_naive_bayes_diagram()
    create_gmm_concept_diagram()
