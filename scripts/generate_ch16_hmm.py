
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_dynamic_model_diagram():
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

    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Colors
    color_hidden = '#ffffff'
    color_observed = '#e0e0e0'
    color_edge = '#333333'
    color_text = '#333333'

    # Node Geometry
    radius = 0.5
    node_lw = 2.0

    # Layout Config
    y_hidden = 5.0
    y_obs = 2.0

    nodes = [
        {'pos_x': 2.0,  'label_z': r'$z_1$',     'label_x': r'$x_1$',     'type': 'node'},
        {'pos_x': 5.0,  'label_z': r'$z_2$',     'label_x': r'$x_2$',     'type': 'node'},
        {'pos_x': 8.0,  'label_z': r'$\cdots$',  'label_x': r'$\cdots$',  'type': 'dots'},
        {'pos_x': 10.0,  'label_z': r'$z_t$',     'label_x': r'$x_t$',     'type': 'node'},
    ]

    # Draw Nodes & Arrows
    for index, node in enumerate(nodes):
        x = node['pos_x']

        if node['type'] == 'dots':
            ax.text(x, y_hidden, node['label_z'], ha='center', va='center', fontsize=20, color=color_edge)
            ax.text(x, y_obs, node['label_x'], ha='center', va='center', fontsize=20, color=color_edge)

            # Transition Arrow to dots
            prev_x = nodes[index-1]['pos_x'] + radius + 0.1
            ax.annotate("", xy=(x - 0.4, y_hidden), xytext=(prev_x, y_hidden),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))

            # Transition Arrow from dots
            if index < len(nodes) - 1:
                next_x = nodes[index+1]['pos_x'] - radius - 0.1
                ax.annotate("", xy=(next_x, y_hidden), xytext=(x + 0.4, y_hidden),
                            arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))
            continue

        # Hidden State Node
        circle_z = patches.Circle((x, y_hidden), radius, facecolor=color_hidden, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle_z)
        ax.text(x, y_hidden, node['label_z'], ha='center', va='center', fontsize=16, zorder=11, color=color_text)

        # Observed Node
        circle_x = patches.Circle((x, y_obs), radius, facecolor=color_observed, edgecolor=color_edge, lw=node_lw, zorder=10)
        ax.add_patch(circle_x)
        ax.text(x, y_obs, node['label_x'], ha='center', va='center', fontsize=16, zorder=11, color=color_text)

        # Emission Arrow (z -> x)
        ax.annotate("", xy=(x, y_obs + radius), xytext=(x, y_hidden - radius),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))

        # Transition Arrow (z_{t-1} -> z_t)
        if index < len(nodes) - 1 and nodes[index+1]['type'] == 'node':
            next_x = nodes[index+1]['pos_x'] - radius - 0.1
            ax.annotate("", xy=(next_x, y_hidden), xytext=(x + radius + 0.1, y_hidden),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=color_edge))

    # Labels
    ax.text(0.5, y_hidden, "State\n(Hidden)", ha='center', va='center', fontsize=14, color='#555555', fontweight='bold')
    ax.text(0.5, y_obs, "Observation\n(Observed)", ha='center', va='center', fontsize=14, color='#555555', fontweight='bold')

    # Title
    ax.text(6.0, 6.5, "Dynamic Model Structure (HMM / State Space)", ha='center', va='center', fontsize=18, fontweight='bold', color='#333333')

    output_path = os.path.join(output_dir, 'ch16_hmm_structure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_dynamic_model_diagram()
