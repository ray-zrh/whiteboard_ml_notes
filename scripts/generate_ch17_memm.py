import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_memm_diagram():
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

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Colors
    color_hidden = '#ffffff'
    color_observed = '#e0e0e0'
    color_edge = '#333333'
    color_param = '#d62728'
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
    ax.text(6, 6.5, "Maximum Entropy Markov Model (MEMM)", ha='center', va='center', fontsize=20, fontweight='bold', color=color_text)
    ax.text(6, 0.5, r"Discriminative: $P(Y|X) = \prod P(y_t|y_{t-1}, X)$", ha='center', va='center', fontsize=16, color=color_param)

    # Label Bias Highlight
    ax.text(6, 1.0, "(Local Normalization causes Label Bias)", ha='center', va='center', fontsize=12, color='#888888', style='italic')

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

        # Vertical Arrow (x -> y)
        ax.annotate("", xy=(x, y_hidden - radius), xytext=(x, y_obs + radius),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))

        # Horizontal Arrow (y_t-1 -> y_t)
        if t > 0:
            prev_x = x_positions[t-1]
            ax.annotate("", xy=(x - radius, y_hidden), xytext=(prev_x + radius, y_hidden),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))

    # Global X node (Xg) - Move higher to avoid clutter
    xg_x, xg_y = 6.0, 5.8
    circle_xg = patches.Circle((xg_x, xg_y), radius, facecolor=color_observed, edgecolor=color_edge, lw=node_lw, zorder=10)
    ax.add_patch(circle_xg)
    ax.text(xg_x, xg_y, r"$X_g$", ha='center', va='center', fontsize=14, zorder=11)
    ax.text(xg_x + 0.8, xg_y, "Global\nInput", ha='left', va='center', fontsize=10, color='#555555')

    # Arrows from Xg to all Y - Curved to be "sensible"
    for x in x_positions:
        # Calculate angle for connection
        style = "Simple,tail_width=0.5,head_width=4,head_length=8"
        kw = dict(arrowstyle="->", color='#999999', lw=1.0, linestyle='--')

        # Use curved connector to avoid straight line clutter
        # connectionstyle="arc3,rad=-0.2" etc.
        if x < xg_x:
            rad = 0.1
        elif x > xg_x:
            rad = -0.1
        else:
            rad = 0

        ax.annotate("", xy=(x, y_hidden + radius), xytext=(xg_x, xg_y - radius),
                    arrowprops=dict(arrowstyle="->", color='#999999', lw=1.0, connectionstyle=f"arc3,rad={rad}"))

    # Local Norm visual cue
    # Box typically encompasses y_t-1 and x_t leading to y_t normalization?
    # Or just the fact that at y_t we sum to 1?
    # Let's draw a box around (y_2, x_2) and maybe part of the incoming arrow from y_1?
    # To be "sensible", let's box the elements that contribute to P(y_t | ...) for one step
    # P(y_t | y_{t-1}, x_t, X)

    # Let's box x_2, y_2 and the incoming transition?
    # Whiteboard usually boxes y_t and x_t and incoming arrow.

    target_idx = 1
    tx = x_positions[target_idx]

    # Dashed box around y_t and x_t
    rect_w = 1.6
    rect_h = 3.5
    rect = patches.FancyBboxPatch((tx - 0.8, y_obs - 0.6), rect_w, rect_h,
                                  boxstyle="round,pad=0.1",
                                  fc='none', ec=color_param, lw=1.5, linestyle='--')
    ax.add_patch(rect)

    # Annotation "Local Norm" with arrow pointing to the box
    ax.annotate("Local Norm\n$\sum_{y_t} P(y_t|\dots)=1$",
                xy=(tx + 0.9, 3.0), xytext=(tx + 2.0, 3.0),
                arrowprops=dict(arrowstyle="->", color=color_param),
                color=color_param, fontsize=10, ha='left', va='center')

    output_path = os.path.join(output_dir, 'ch17_memm_structure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_memm_diagram()
