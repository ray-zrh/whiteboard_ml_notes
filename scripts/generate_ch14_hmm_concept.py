
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_hmm_concept_diagram():
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

    # Wider figure
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Colors (Professional Whiteboard Style)
    color_hidden = '#ffffff'
    color_observed = '#e0e0e0'
    color_edge = '#333333'
    color_param = '#d62728' # Red for parameters
    color_text = '#333333'

    # Node Geometry
    radius = 0.6
    node_lw = 2.0

    # Layout Config
    y_hidden = 6.0
    y_obs = 3.5

    # 1. Define Nodes Structure (Logic from ch13 script)
    # i1 -> i2 -> dots -> it -> it+1 -> dots -> iT
    nodes = [
        {'pos_x': 2.0,  'label_i': r'$i_1$',     'label_o': r'$o_1$',     'type': 'node', 'prior': r'$\pi$'},
        {'pos_x': 4.5,  'label_i': r'$i_2$',     'label_o': r'$o_2$',     'type': 'node'},
        {'pos_x': 6.5,  'label_i': r'$\cdots$',  'label_o': r'$\cdots$',  'type': 'dots'},
        {'pos_x': 8.5,  'label_i': r'$i_t$',     'label_o': r'$o_t$',     'type': 'node'},
        {'pos_x': 11.0, 'label_i': r'$i_{t+1}$', 'label_o': r'$o_{t+1}$', 'type': 'node'},
        {'pos_x': 13.0, 'label_i': r'$\cdots$',  'label_o': r'$\cdots$',  'type': 'dots'},
        {'pos_x': 15.0, 'label_i': r'$i_T$',     'label_o': r'$o_T$',     'type': 'node'}
    ]

    # --- Draw Nodes & Vertical Arrows ---
    for node in nodes:
        x = node['pos_x']

        # Draw Dots
        if node['type'] == 'dots':
            ax.text(x, y_hidden, node['label_i'], ha='center', va='center', fontsize=24, color=color_edge)
            ax.text(x, y_obs, node['label_o'], ha='center', va='center', fontsize=24, color=color_edge)
            continue

        # Draw Hidden Node
        circle_i = patches.Circle((x, y_hidden), radius, facecolor=color_hidden, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle_i)
        ax.text(x, y_hidden, node['label_i'], ha='center', va='center', fontsize=16, zorder=11, color=color_text)

        # Draw Observed Node
        circle_o = patches.Circle((x, y_obs), radius, facecolor=color_observed, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle_o)
        ax.text(x, y_obs, node['label_o'], ha='center', va='center', fontsize=16, zorder=11, color=color_text)

        # Draw Emission Arrow (Vertical: i -> o)
        # B label only on the first one to avoid clutter
        label_B = r'$B$' if node['label_i'] == r'$i_1$' else None

        ax.annotate("", xy=(x, y_obs + radius), xytext=(x, y_hidden - radius),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge, shrinkA=0, shrinkB=0))

        if label_B:
            ax.text(x - 0.2, (y_hidden + y_obs)/2, label_B, ha='right', va='center', fontsize=14, color=color_param, fontweight='bold')

        # Draw Prior (Pi) if applicable
        if node.get('prior'):
            ax.annotate("", xy=(x - radius - 0.1, y_hidden), xytext=(x - radius - 1.2, y_hidden),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=color_param))
            ax.text(x - radius - 1.0, y_hidden + 0.3, node['prior'], ha='center', va='bottom', fontsize=16, color=color_param, fontweight='bold')

    # --- Draw Horizontal Arrows (Transitions) ---
    for i in range(len(nodes) - 1):
        curr_node = nodes[i]
        next_node = nodes[i+1]

        x1 = curr_node['pos_x']
        x2 = next_node['pos_x']

        # Determine start and end points based on node type
        # If node, radius is used. If dots, reduced gap.

        start_x = x1 + (radius if curr_node['type'] == 'node' else 0.4)
        end_x = x2 - (radius if next_node['type'] == 'node' else 0.4)

        # Additional buffer for arrow head
        if curr_node['type'] == 'node': start_x += 0.1
        if next_node['type'] == 'node': end_x -= 0.1

        # Draw Arrow
        ax.annotate("", xy=(end_x, y_hidden), xytext=(start_x, y_hidden),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))

        # Label A
        # Logic: Label A on typical transitions, avoid on 'dots' transitions if clutter
        # We want A between 1->2, t->t+1.
        # Maybe not on arrows involving dots?

        if curr_node['type'] == 'node' and next_node['type'] == 'node':
            mid_x = (start_x + end_x) / 2
            ax.text(mid_x, y_hidden + 0.3, r'$A$', ha='center', va='bottom', fontsize=14, color=color_param, fontweight='bold')


    # --- Left Labels ---
    ax.text(0.5, y_hidden, "Latent\n(Hidden)", ha='center', va='center', fontsize=14, color='#555555', fontweight='bold')
    ax.text(0.5, y_obs, "Observed", ha='center', va='center', fontsize=14, color='#555555', fontweight='bold')

    # --- Title ---
    ax.text(8.25, 8.0, "Hidden Markov Model (HMM)", ha='center', va='center', fontsize=22, fontweight='bold', color='#333333')

    # --- The Three Basic Problems (Bottom Panel) ---

    panel_y = 0.5
    panel_height = 1.6
    panel_width = 3.5
    gap = 0.8

    total_w = 3 * panel_width + 2 * gap
    start_x_panel = (16.5 - total_w) / 2

    # Colors for problems
    c1 = '#e3f2fd' # Light blue
    c1_edge = '#1e88e5'
    c2 = '#e8f5e9' # Light green
    c2_edge = '#43a047'
    c3 = '#fff3e0' # Light orange
    c3_edge = '#fb8c00'

    configs = [
        (0, "1. Evaluation", r"$P(O|\lambda)$", "Forward-Backward", c1, c1_edge),
        (1, "2. Learning", r"$\hat{\lambda} = \arg\max_\lambda P(O|\lambda)$", "Baum-Welch (EM)", c2, c2_edge),
        (2, "3. Decoding", r"$\hat{I} = \arg\max_I P(I|O)$", "Viterbi", c3, c3_edge),
    ]

    for idx, title, math, algo, face, edge in configs:
        px = start_x_panel + idx * (panel_width + gap)
        rect = patches.FancyBboxPatch((px, panel_y), panel_width, panel_height,
                                      boxstyle="round,pad=0.1",
                                      fc=face, ec=edge, lw=2)
        ax.add_patch(rect)

        # Text
        cx = px + panel_width / 2
        cy = panel_y + panel_height / 2

        ax.text(cx, cy + 0.5, title, ha='center', va='center', fontsize=14, fontweight='bold', color=edge)
        ax.text(cx, cy, math, ha='center', va='center', fontsize=14, color='#333333')
        ax.text(cx, cy - 0.5, algo, ha='center', va='center', fontsize=13, style='italic', color='#555555')

    output_path = os.path.join(output_dir, 'ch14_hmm_concept.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_hmm_concept_diagram()
