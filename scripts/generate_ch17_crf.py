import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_crf_diagram():
    # Style settings
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 0
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors
    color_hidden = '#ffffff'
    color_observed = '#e0e0e0'
    color_edge = '#333333'
    color_param = '#d62728' # Red for consistent highlights
    color_text = '#333333'

    # Config
    y_hidden = 4.0
    y_obs = 2.0
    radius = 0.5
    node_lw = 2.0

    # Nodes
    T = 4
    x_positions = [2.0 + i*2.5 for i in range(T)]

    # Title & Formula
    ax.text(6, 5.5, "Conditional Random Field (CRF)", ha='center', va='center', fontsize=20, fontweight='bold', color=color_text)
    ax.text(6, 0.5, r"Discriminative (Global Norm): $P(Y|X) = \frac{1}{Z(X)} \exp(\sum \lambda_k f_k)$", ha='center', va='center', fontsize=16, color=color_param)

    # Legend
    ax.text(0.5, y_hidden, "Latent\n$y$", ha='center', va='center', fontsize=12, color='#555555')
    ax.text(0.5, y_obs, "Observed\n$x$", ha='center', va='center', fontsize=12, color='#555555')

    # Draw Nodes
    for t, x in enumerate(x_positions):
        # Hidden Node
        circle_h = patches.Circle((x, y_hidden), radius, facecolor=color_hidden, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle_h)
        ax.text(x, y_hidden, f"$y_{t+1}$", ha='center', va='center', fontsize=14, zorder=11)

        # Observed Node
        circle_o = patches.Circle((x, y_obs), radius, facecolor=color_observed, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle_o)
        ax.text(x, y_obs, f"$x_{t+1}$", ha='center', va='center', fontsize=14, zorder=11)

        # Vertical Edge (Undirected)
        ax.plot([x, x], [y_obs + radius, y_hidden - radius], color=color_edge, lw=2.0, zorder=5)

        # Horizontal Edge (Undirected)
        if t > 0:
            prev_x = x_positions[t-1]
            ax.plot([prev_x + radius, x - radius], [y_hidden, y_hidden], color=color_edge, lw=2.0, zorder=5)

    # Global Norm visual cue
    # Draw a big box around the whole sequence
    rect = patches.FancyBboxPatch((1.0, 1.0), 10.0, 4.0,
                                  boxstyle="round,pad=0.2",
                                  fc='none', ec=color_param, lw=2, linestyle='-')
    ax.add_patch(rect)
    ax.text(11.2, 5.0, "Global\nPartition\n$Z(X)$", color=color_param, fontsize=12, ha='left')

    output_path = os.path.join(output_dir, 'ch17_crf_structure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_crf_diagram()
