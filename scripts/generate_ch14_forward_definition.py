
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_forward_definition_diagram():
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
    color_highlight = '#e3f2fd' # Light blue for box background? Or just dashed line
    color_box_edge = '#1e88e5' # Blue dashed line

    node_radius = 0.5
    node_lw = 2.0

    # Layout Config
    y_hidden = 4.0
    y_obs = 1.5

    # Nodes: 1, ..., t, ..., T
    # We need enough nodes to show the box enclosing 1..t

    nodes = [
        {'id': 1, 'x': 2.0,  'li': r'$i_1$',     'lo': r'$o_1$'},
        {'id': 2, 'x': 4.5,  'li': r'$\cdots$',  'lo': r'$\cdots$', 'type': 'dots'},
        {'id': 3, 'x': 7.0,  'li': r'$i_t$',     'lo': r'$o_t$'},
        {'id': 4, 'x': 9.5,  'li': r'$i_{t+1}$', 'lo': r'$o_{t+1}$'},
        {'id': 5, 'x': 11.5, 'li': r'$\cdots$',  'lo': r'$\cdots$', 'type': 'dots'}
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


    # --- Draw Dashed Box for Alpha_t(i) ---
    # Enclosing o1...ot and it
    # i_t is node index 2 (x=7.0)
    # o_1 is node index 0 (x=2.0)

    # We want to enclose:
    # All observations up to t: o1 ... ot
    # The state at t: it
    # In the whiteboard, the box encloses the whole path of observations and the specific state it.

    # Let's draw a fancy box
    # Top bound: around it (y=4.0)
    # Bottom bound: around obs (y=1.5)
    # Left bound: before i1 (x=2.0)
    # Right bound: after it (x=7.0)

    # Actually, strictly definition of alpha_t(i): P(o_1...o_t, i_t = q_i)
    # Box typically encloses these variables.

    # Coordinates for box
    x_min = 1.0
    x_max = 7.8 # Include i_t and o_t
    y_max = 5.0 # Above i_t
    y_min = 0.5 # Below o_obs

    # However, we only care about i_t, not i_1...i_{t-1} in the joint prob definition?
    # Usually we show the path leading to it.
    # On the whiteboard, there are dashed boxes around specific columns (t, t+1)
    # AND a large curly brace grouping the history for derivation.

    # Visual on whiteboard:
    # - Dashed box around (i_t, o_t) ?? No, looks like dashed box around (o_1...o_t, i_t)
    # - Actually there is a box for "Forward Algorithm" showing:
    #   - i_t enclosed
    #   - o_t enclosed
    #   - And a previous box around t-1?

    # Let's draw a polygon/shape that encompasses o1..ot and terminates at it
    # L-shaped?
    # Or just a big box around everything up to t.

    # Let's follow the "Concept" of alpha_t:
    # "Score of reaching state i at time t, having observed o1...ot"

    # I will draw a blue dashed box around o1...ot and i_t.
    # Exclude i1...i{t-1} from the *box*? No, usually they are marginalized out, but visually we just group the relevant termination.

    # Let's draw a box around (i_t) and a larger box around (o1...ot)?
    # Simplest: A box around the column t (it, ot) and an arrow coming from "history".

    # Whiteboard Diagram 2 ("Forward Algorithm"):
    # Shows i_t, o_t enclosed in a dashed box.
    # Shows i_{t+1}, o_{t+1} in a dashed box.
    # Shows arrow from t to t+1.

    # Let's replicate that structure.

    # Box 1: Around (i_t, o_t)
    # Box 2: Around (i_{t+1}, o_{t+1})

    # Draw dashed box around t
    box_t_x = 7.0
    width = 1.6
    height = 4.0
    rect_t = patches.FancyBboxPatch((box_t_x - width/2, y_obs - 0.8), width, height,
                                    boxstyle="round,pad=0.2",
                                    fc="none", ec="#1e88e5", lw=2, linestyle='--')
    ax.add_patch(rect_t)

    ax.text(box_t_x, y_max + 0.2, r'$\alpha_t(i)$', ha='center', va='bottom', fontsize=18, color='#1e88e5', fontweight='bold')

    # Draw dashed box around t+1 (maybe red or differnet color to show next step?)
    # Or just show t and t+1 relation.

    box_tp1_x = 9.5
    rect_tp1 = patches.FancyBboxPatch((box_tp1_x - width/2, y_obs - 0.8), width, height,
                                    boxstyle="round,pad=0.2",
                                    fc="none", ec="#d62728", lw=2, linestyle='--')
    ax.add_patch(rect_tp1)

    ax.text(box_tp1_x, y_max + 0.2, r'$\alpha_{t+1}(j)$', ha='center', va='bottom', fontsize=18, color='#d62728', fontweight='bold')

    # Add text explaining the definition
    # alpha_t(i) = P(o_1...o_t, i_t=q_i)
    # We can place text labels near the bottom or top

    ax.text(6.0, 0.2, r"$\alpha_t(i) = P(o_1, \dots, o_t, i_t=q_i | \lambda)$",
            ha='center', va='center', fontsize=16, color='#333333',
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc"))

    output_path = os.path.join(output_dir, 'ch14_forward_definition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_forward_definition_diagram()
