
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_backward_definition_diagram():
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

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors
    color_hidden = '#ffffff'
    color_observed = '#e0e0e0'
    color_edge = '#333333'
    color_param = '#d62728'
    color_text = '#333333'
    color_highlight = '#e8f5e9' # Light green for backward
    color_box_edge = '#43a047' # Green dashed line

    node_radius = 0.5
    node_lw = 2.0

    # Layout Config
    y_hidden = 4.0
    y_obs = 1.5

    # Nodes: t, t+1, ..., T
    nodes = [
        {'id': 3, 'x': 2.0,  'li': r'$i_t$',     'lo': r'$o_t$'},
        {'id': 4, 'x': 4.5,  'li': r'$i_{t+1}$', 'lo': r'$o_{t+1}$'},
        {'id': 5, 'x': 7.0,  'li': r'$\cdots$',  'lo': r'$\cdots$', 'type': 'dots'},
        {'id': 6, 'x': 9.5,  'li': r'$i_T$',     'lo': r'$o_T$'}
    ]

    # --- Draw Nodes & Arrows ---
    for i, node in enumerate(nodes):
        x = node['x']

        if node.get('type') == 'dots':
            ax.text(x, y_hidden, node['li'], ha='center', va='center', fontsize=24, color=color_edge)
            ax.text(x, y_obs, node['lo'], ha='center', va='center', fontsize=24, color=color_edge)

            # Arrows to/from dots
            if i > 0:
                prev = nodes[i-1]
                # Arrow prev -> dots
                start = prev['x'] + (node_radius if prev.get('type')!='dots' else 0.4)
                end = x - 0.4
                ax.annotate("", xy=(end, y_hidden), xytext=(start, y_hidden),
                            arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))
            continue

        # Hidden Node
        circle_i = patches.Circle((x, y_hidden), node_radius, facecolor=color_hidden, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle_i)
        ax.text(x, y_hidden, node['li'], ha='center', va='center', fontsize=16, zorder=11, color=color_text)

        # Observed Node
        circle_o = patches.Circle((x, y_obs), node_radius, facecolor=color_observed, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle_o)
        ax.text(x, y_obs, node['lo'], ha='center', va='center', fontsize=16, zorder=11, color=color_text)

        # Emission Arrow
        ax.annotate("", xy=(x, y_obs + node_radius), xytext=(x, y_hidden - node_radius),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))

        # Transition Arrow (from prev)
        if i > 0:
            prev = nodes[i-1]
            start = prev['x'] + (node_radius if prev.get('type')!='dots' else 0.4)
            end = x - node_radius
            ax.annotate("", xy=(end, y_hidden), xytext=(start, y_hidden),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))


    # --- Draw Dashed Box for Beta_t(i) ---
    # Enclosing future observations: o_{t+1} ... o_T
    # t+1 is at x=4.5
    # T is at x=9.5

    # We define box around these observations.
    # Note: Beta conditions on i_t, but calculates probability of o_{t+1}...o_T

    # Let's outline the observations
    # Box x range: from slightly before o_{t+1} to after o_T
    # o_{t+1} x=4.5
    # dots x=7.0
    # o_T x=9.5

    x_min_box = 3.5 # Before t+1
    x_max_box = 10.5 # After T

    # Box y range: around observations
    # But often we visualize the structure.
    # Let's draw a box just around the observations row?
    # Or include the hidden states to show the path?
    # Beta computes P(obs | state), so the state path is marginalized (summed out).
    # Forward alpha computes P(obs, state), so specific state is endpoint.

    # Whiteboard for beta just shows "Beta_t(i)" and arrows.
    # Let's stick to the "future observations" box concept.

    rect_beta = patches.FancyBboxPatch((x_min_box, y_obs - 0.8), x_max_box - x_min_box, 1.6,
                                    boxstyle="round,pad=0.2",
                                    fc="none", ec=color_box_edge, lw=2, linestyle='--')
    ax.add_patch(rect_beta)

    # Label Beta_t(i)
    ax.text((x_min_box + x_max_box)/2, y_obs - 1.2, r'$\beta_t(i)$', ha='center', va='top', fontsize=18, color=color_box_edge, fontweight='bold')

    # Visual connection from i_t to the box?
    # Arrow from i_t (x=2.0) to the box?
    # "Given i_t"

    ax.annotate("", xy=(x_min_box, y_obs), xytext=(2.0, y_hidden-0.5), # From i_t
                arrowprops=dict(arrowstyle="->", lw=1.5, color=color_box_edge, ls='dashed', connectionstyle="arc3,rad=0.2"))

    ax.text(2.5, 2.5, r"Given $i_t=q_i$", fontsize=12, color=color_box_edge, rotation=-30)


    # Add text explaining the definition
    ax.text(6.0, 5.2, r"$\beta_t(i) = P(o_{t+1}, \dots, o_T | i_t=q_i, \lambda)$",
            ha='center', va='center', fontsize=18, color='#333333',
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc"))

    output_path = os.path.join(output_dir, 'ch14_backward_definition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_backward_definition_diagram()
