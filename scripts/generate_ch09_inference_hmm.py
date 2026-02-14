
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_hmm_inference_diagram():
    # Use a style for better aesthetics if available, otherwise default
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass # Fallback to default

    # Revert to default font for reliability
    # Usually DejaVu Sans or similar works well for mixed text/math
    # If explicit 'cm' caused issues, remove it.
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        # "font.serif": ["Computer Modern Roman"], # Removed explicit CMR requirement
        "mathtext.fontset": "cm", # CM math font is usually built-in and safe
    })

    fig, ax = plt.subplots(figsize=(12, 7)) # Slightly larger
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Coordinates
    positions_i = [(2, 4), (5, 4), (8, 4)] # Spread out more
    positions_o = [(2, 2), (5, 2), (8, 2)]

    radius = 0.5

    # Draw Nodes
    for i, (x, y) in enumerate(positions_i):
        # Hidden nodes
        circle = patches.Circle((x, y), radius, facecolor='white', edgecolor='black', lw=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, f'$I_{i+1}$', ha='center', va='center', fontsize=20, zorder=11)

    for i, (x, y) in enumerate(positions_o):
        # Observed nodes
        circle = patches.Circle((x, y), radius, facecolor='#dddddd', edgecolor='black', lw=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, f'$O_{i+1}$', ha='center', va='center', fontsize=20, zorder=11)

    # Draw Arrows
    # I -> I
    for i in range(len(positions_i) - 1):
        x1, y1 = positions_i[i]
        x2, y2 = positions_i[i+1]
        ax.annotate("", xy=(x2-radius, y2), xytext=(x1+radius, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color='black'))

    # I -> O
    for i in range(len(positions_i)):
        xi, yi = positions_i[i]
        xo, yo = positions_o[i]
        ax.annotate("", xy=(xo, yo+radius), xytext=(xi, yi-radius),
                    arrowprops=dict(arrowstyle="->", lw=2, color='black'))

    # Annotations - Use simpler Latex or Text
    # Evaluation (Problem 1)

    # 1. Evaluation
    ax.plot([1.5, 1.5, 8.5, 8.5], [1.2, 0.8, 0.8, 1.2], color='#d62728', lw=2) # Red Bracket below O
    ax.plot([5, 5], [0.8, 0.5], color='#d62728', lw=2)

    # Use \mathbf instead of \bf which is safer in Matplotlib
    ax.text(5, 0.2, r'$\mathbf{1.\ Evaluation:}$' + '\n' + r'$P(O) = \sum_I P(I,O)$ (Forward Alg.)',
            ha='center', va='top', fontsize=14, color='#d62728')

    # 3. Decoding
    ax.plot([1.5, 1.5, 8.5, 8.5], [4.8, 5.2, 5.2, 4.8], color='#1f77b4', lw=2) # Blue Bracket above I
    ax.plot([5, 5], [5.2, 5.5], color='#1f77b4', lw=2)

    ax.text(5, 5.6, r'$\mathbf{3.\ Decoding:}$' + '\n' + r'$\hat{I} = \arg\max_I P(I|O)$ (Viterbi Alg.)',
            ha='center', va='bottom', fontsize=14, color='#1f77b4')

    # 2. Learning
    # Point to the transitions naturally
    # Arrow pointing to the edge between I1 and I2 maybe?
    # Or just a box on the side.

    # Let's put a box on the right
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="#2ca02c", lw=2, alpha=0.9)
    ax.text(8.8, 3, r'$\mathbf{2.\ Learning:}$' + '\n' + r'$\hat{\lambda}$ (Baum-Welch / EM)',
            ha='left', va='center', fontsize=14, color='#2ca02c', bbox=bbox_props)

    output_path = os.path.join(output_dir, 'ch09_hmm_inference.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_hmm_inference_diagram()
