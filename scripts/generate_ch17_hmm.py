import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_hmm_diagram():
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

    # Colors (Professional Whiteboard Style)
    color_hidden = '#ffffff'
    color_observed = '#e0e0e0'
    color_edge = '#333333'
    color_param = '#d62728' # Red for parameters/highlights
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
    ax.text(6, 5.5, "Hidden Markov Model (HMM)", ha='center', va='center', fontsize=20, fontweight='bold', color=color_text)
    ax.text(6, 0.5, r"Generative: $P(x, y) = \prod P(y_t|y_{t-1}) P(x_t|y_t)$", ha='center', va='center', fontsize=16, color=color_param)

    # Legend Labels
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

        # Vertical Arrow (y -> x) Emission
        ax.annotate("", xy=(x, y_obs + radius), xytext=(x, y_hidden - radius),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))

        # Horizontal Arrow (y_t-1 -> y_t) Transition
        if t > 0:
            prev_x = x_positions[t-1]
            ax.annotate("", xy=(x - radius, y_hidden), xytext=(prev_x + radius, y_hidden),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))

    # Annotate Assumptions
    ax.text(x_positions[-1] + 1.2, y_hidden, "Markov\nAssumption", fontsize=10, color='#777777', ha='left', va='center')
    ax.text(x_positions[-1] + 1.2, y_obs, "Output\nIndependence", fontsize=10, color='#777777', ha='left', va='center')


    output_path = os.path.join(output_dir, 'ch17_hmm_structure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_hmm_diagram()
