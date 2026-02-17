
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets/ch14")
os.makedirs(output_dir, exist_ok=True)

def create_forward_alg_diagram():
    # Style settings
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 0
    })

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Colors (Consistent with HMM Concept)
    color_hidden = '#ffffff'
    color_observed = '#e0e0e0'
    color_edge = '#333333'
    color_param = '#d62728' # Red
    color_text = '#333333'
    color_highlight = '#fff9c4' # Light yellow for target

    node_radius = 0.5
    node_lw = 2.0

    # Layout Config
    # Columns: t-1, t, t+1 (Focus on t -> t+1)
    # Rows: 1, ..., i, ..., N

    x_t = 4.0
    x_tp1 = 8.0

    # Y positions for states 1, i, N
    y_states = [6.0, 4.0, 2.0]
    labels_states = [r'$1$', r'$i$', r'$N$']

    # --- Draw Time t Column ---
    nodes_t = []
    for idx, (y, label) in enumerate(zip(y_states, labels_states)):
        # Node
        circle = patches.Circle((x_t, y), node_radius, facecolor=color_hidden, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle)
        ax.text(x_t, y, label, ha='center', va='center', fontsize=16, zorder=11, color=color_text)

        # Alpha Label
        ax.text(x_t - 0.8, y, r'$\alpha_t({})$'.format(label.replace('$', '')),
                ha='right', va='center', fontsize=16, color='#1565c0', fontweight='bold')

        nodes_t.append((x_t, y))

    # --- Draw Time t+1 Column ---
    # We focus on a specific state j (middle one)
    nodes_tp1 = []
    for idx, (y, label) in enumerate(zip(y_states, [r'$1$', r'$j$', r'$N$'])):

        is_target = (idx == 1) # Middle node is j, our target

        fc = color_highlight if is_target else color_hidden
        ec = color_param if is_target else color_edge
        lw = 2.5 if is_target else node_lw

        circle = patches.Circle((x_tp1, y), node_radius, facecolor=fc, edgecolor=ec, lw=lw, zorder=10)
        ax.add_patch(circle)
        ax.text(x_tp1, y, label, ha='center', va='center', fontsize=16, zorder=11, color=color_text)

        # Alpha Label for target
        if is_target:
             ax.text(x_tp1 + 0.8, y, r'$\alpha_{t+1}(j)$',
                ha='left', va='center', fontsize=16, color='#1565c0', fontweight='bold')

        nodes_tp1.append((x_tp1, y))

    # --- Transitions (Trellis) ---
    # Draw arrows from ALL t nodes to target t+1 node (j)
    target_pos = nodes_tp1[1]

    for i, start_pos in enumerate(nodes_t):
        # Arrow style
        # If it's the middle one (i -> j), make it distinct or just all same
        arrow_color = color_edge

        # Draw arrow
        # Adjust start/end
        # Calculate angle to adjust properly? Simple adjustment:
        ax.annotate("", xy=(target_pos[0]-node_radius-0.05, target_pos[1]),
                    xytext=(start_pos[0]+node_radius+0.05, start_pos[1]),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=arrow_color))

        # Label a_ij on the arrow
        # Position slightly offset based on source
        mid_x = (start_pos[0] + target_pos[0]) / 2
        mid_y = (start_pos[1] + target_pos[1]) / 2

        if i == 0: # Top
            ax.text(mid_x, mid_y + 0.2, r'$a_{1j}$', ha='center', va='bottom', fontsize=12, color=color_text)
        elif i == 1: # Middle
            ax.text(mid_x, mid_y + 0.2, r'$a_{ij}$', ha='center', va='bottom', fontsize=14, color=color_param, fontweight='bold')
        elif i == 2: # Bottom
            ax.text(mid_x, mid_y + 0.2, r'$a_{Nj}$', ha='center', va='bottom', fontsize=12, color=color_text)

    # --- Ghost Transitions for context ---
    # Show that other nodes at t+1 also receive inputs (faded)
    for idx, end_pos in enumerate(nodes_tp1):
        if idx == 1: continue # Skip target
        # Just one representative arrow per other node to show it exists
        start_pos = nodes_t[1] # From i
        ax.annotate("", xy=(end_pos[0]-node_radius, end_pos[1]), xytext=(start_pos[0]+node_radius, start_pos[1]),
                    arrowprops=dict(arrowstyle="->", lw=1, color='#dddddd', ls='--'))

    # --- Observation ---
    # o_{t+1} below target node j
    obs_x = target_pos[0]
    obs_y = target_pos[1] - 1.8

    circle_obs = patches.Circle((obs_x, obs_y), node_radius, facecolor=color_observed, edgecolor=color_edge, lw=node_lw, zorder=10)
    ax.add_patch(circle_obs)
    ax.text(obs_x, obs_y, r'$o_{t+1}$', ha='center', va='center', fontsize=16, zorder=11, color=color_text)

    # Emission Arrow j -> o
    ax.annotate("", xy=(obs_x, obs_y + node_radius), xytext=(obs_x, target_pos[1] - node_radius),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))
    ax.text(obs_x + 0.2, (obs_y + target_pos[1])/2, r'$b_j(o_{t+1})$', ha='left', va='center', fontsize=14, color=color_param, fontweight='bold')


    # --- Titles / Time Axis ---
    ax.text(x_t, 7.5, r'Time $t$', ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(x_tp1, 7.5, r'Time $t+1$', ha='center', va='center', fontsize=18, fontweight='bold')

    # --- Summation Brace/Text ---
    # Visualizing that alpha_t+1 is sum of (alpha_t * a * b)
    # A large curly brace on the left of target j gathering all incoming?
    # Or just text annotation.

    # Let's add the formula at the bottom or top
    formula = r'$\alpha_{t+1}(j) = \left[ \sum_{i=1}^N \alpha_t(i) a_{ij} \right] b_j(o_{t+1})$'
    ax.text(6.0, 0.5, formula, ha='center', va='center', fontsize=20, color='#333333',
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#dddddd", alpha=0.9))

    # Dots for hidden states 1..N
    ax.text(x_t, 3.0, r'$\vdots$', ha='center', va='center', fontsize=20)
    ax.text(x_tp1, 3.0, r'$\vdots$', ha='center', va='center', fontsize=20)


    output_path = os.path.join(output_dir, 'ch14_forward_alg.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_forward_alg_diagram()
