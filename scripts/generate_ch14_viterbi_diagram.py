
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_viterbi_diagram():
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

    # Colors
    color_hidden = '#ffffff'
    color_observed = '#e0e0e0'
    color_edge = '#333333'
    color_param = '#d62728' # Red for max path
    color_text = '#333333'
    color_highlight = '#fff9c4' # Light yellow for target
    color_faded = '#bdbdbd' # Faded for non-max paths

    node_radius = 0.5
    node_lw = 2.0

    # Layout Config
    # Columns: t, t+1
    x_t = 3.0
    x_tp1 = 8.0

    # Y positions for states 1, i, N
    y_states = [6.5, 4.0, 1.5]
    labels_states = [r'$1$', r'$i$', r'$N$']

    # --- Draw Time t Column ---
    nodes_t = []
    for idx, (y, label) in enumerate(zip(y_states, labels_states)):
        # Node
        # Highlight the source of the max path (middle node i)
        is_source = (idx == 1)
        ec = color_param if is_source else color_edge
        lw = 2.5 if is_source else node_lw

        circle = patches.Circle((x_t, y), node_radius, facecolor=color_hidden, edgecolor=ec, lw=lw, zorder=10)
        ax.add_patch(circle)
        ax.text(x_t, y, label, ha='center', va='center', fontsize=16, zorder=11, color=color_text)

        # Delta Label on the left
        color_label = color_param if is_source else '#1565c0'
        fw = 'bold' if is_source else 'normal'
        ax.text(x_t - 0.8, y, r'$\delta_t({})$'.format(label.replace('$', '')),
                ha='right', va='center', fontsize=16, color=color_label, fontweight=fw)

        nodes_t.append((x_t, y))

    # --- Draw Time t+1 Column ---
    # We focus on a specific state j (middle one)
    nodes_tp1 = []

    for idx, (y, label) in enumerate(zip(y_states, [r'$1$', r'$j$', r'$N$'])):

        is_target = (idx == 1) # Middle node is j, our target

        fc = color_highlight if is_target else color_hidden
        ec = color_param if is_target else '#cccccc' # Darker grey edge
        lw = 2.5 if is_target else 1.5
        alpha = 1.0 # No transparency

        circle = patches.Circle((x_tp1, y), node_radius, facecolor=fc, edgecolor=ec, lw=lw, zorder=10, alpha=alpha)
        ax.add_patch(circle)

        text_col = color_text if is_target else '#aaaaaa'
        ax.text(x_tp1, y, label, ha='center', va='center', fontsize=16, zorder=11, color=text_col)

        # Delta Label for target
        if is_target:
             ax.text(x_tp1 + 0.8, y, r'$\delta_{t+1}(j)$',
                ha='left', va='center', fontsize=18, color=color_param, fontweight='bold')

        nodes_tp1.append((x_tp1, y))

    # --- Transitions (Trellis) ---
    target_pos = nodes_tp1[1]

    for i, start_pos in enumerate(nodes_t):
        # Determine if this is the max path
        is_max_path = (i == 1) # Middle node i -> Middle node j

        if is_max_path:
            arrow_color = color_param
            arrow_lw = 3.0
            zorder = 5
            ls = '-'
        else:
            arrow_color = color_faded
            arrow_lw = 1.5
            zorder = 1
            ls = '--'

        # Draw arrow
        # Use simple straight line for clarity
        ax.annotate("", xy=(target_pos[0]-node_radius-0.05, target_pos[1]),
                    xytext=(start_pos[0]+node_radius+0.05, start_pos[1]),
                    arrowprops=dict(arrowstyle="->", lw=arrow_lw, color=arrow_color, ls=ls),
                    zorder=zorder)

        # Label a_ij on the arrow
        mid_x = (start_pos[0] + target_pos[0]) / 2
        mid_y = (start_pos[1] + target_pos[1]) / 2

        if is_max_path:
            ax.text(mid_x, mid_y + 0.2, r'$a_{ij}$', ha='center', va='bottom', fontsize=16, color=color_param, fontweight='bold', zorder=6,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0))


    # --- Annotation for Max Path ---
    # Label likelihood on the max path
    ax.text(5.5, 5.0, r"$\psi_{t+1}(j) = \arg\max_i [\delta_t(i) a_{ij}]$", ha='center', va='bottom', fontsize=14, color=color_param, rotation=0,
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=color_param, lw=1.5))


    # --- Observation ---
    # o_{t+1} to the right of target node j
    obs_x = target_pos[0] + 3.0 # Increased spacing
    obs_y = target_pos[1]

    circle_obs = patches.Circle((obs_x, obs_y), node_radius, facecolor=color_observed, edgecolor=color_edge, lw=node_lw, zorder=10)
    ax.add_patch(circle_obs)
    ax.text(obs_x, obs_y, r'$o_{t+1}$', ha='center', va='center', fontsize=16, zorder=11, color=color_text)

    # Emission Arrow j -> o
    ax.annotate("", xy=(obs_x - node_radius, obs_y), xytext=(target_pos[0] + node_radius, target_pos[1]),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))
    ax.text((obs_x + target_pos[0])/2, obs_y + 0.3, r'$b_j(o_{t+1})$', ha='center', va='bottom', fontsize=14, color=color_text)


    # --- Titles / Time Axis ---
    ax.text(x_t, 7.5, r'Time $t$', ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(x_tp1, 7.5, r'Time $t+1$', ha='center', va='center', fontsize=18, fontweight='bold')

    # --- Formula at bottom left ---
    formula = r'$\delta_{t+1}(j) = \max_{1 \leq i \leq N} [\delta_t(i) a_{ij}] b_j(o_{t+1})$'
    ax.text(1.0, 0.5, formula, ha='left', va='center', fontsize=20, color='#333333',
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#dddddd", alpha=0.9))

    # Dots for hidden states 1..N
    ax.text(x_t, 2.8, r'$\vdots$', ha='center', va='center', fontsize=20)
    ax.text(x_tp1, 2.8, r'$\vdots$', ha='center', va='center', fontsize=20, color='#aaaaaa')


    output_path = os.path.join(output_dir, 'ch14_viterbi_diagram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_viterbi_diagram()
