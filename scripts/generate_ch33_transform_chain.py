import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_transform_chain():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    # Nodes
    nodes = ['z_0', 'z_1', '...', 'z_k', 'x']
    dists = ['p_{z_0}(z_0)', 'p_{z_1}(z_1)', '', 'p_{z_k}(z_k)', 'p_x(x)']
    funcs = ['f_1', '...', 'f_k', 'f_{k+1}']

    x_pos = [1, 3, 5, 7, 9]
    y_pos = 2
    radius = 0.6

    for i, (p, node, dist) in enumerate(zip(x_pos, nodes, dists)):
        # Draw circle
        if i != 2: # Skip dots
            circle = patches.Circle((p, y_pos), radius, fill=False, edgecolor='navy', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(p, y_pos, f'${node}$', ha='center', va='center', fontsize=14)

            # Draw distribution arrow and text
            ax.annotate('', xy=(p, y_pos - radius - 0.2), xytext=(p, y_pos - radius),
                        arrowprops=dict(arrowstyle='->', color='black', linestyle='dashed'))
            if dist:
                ax.text(p, y_pos - radius - 0.5, f'${dist}$', ha='center', va='center', fontsize=12)
        else:
            ax.text(p, y_pos, node, ha='center', va='center', fontsize=16)

    # Draw transformation arrows
    for i in range(len(funcs)):
        start_x = x_pos[i] + radius
        end_x = x_pos[i+1] - (radius if i+1 != 2 else 0.2)
        if i == 1:
            start_x -= radius - 0.2

        ax.annotate('', xy=(end_x, y_pos), xytext=(start_x, y_pos),
                    arrowprops=dict(arrowstyle='->', color='navy', linewidth=1.5))

        mid_x = (start_x + end_x) / 2
        ax.text(mid_x, y_pos + 0.2, f'${funcs[i]}$', ha='center', va='bottom', fontsize=12)

    ax.text(1, y_pos - radius - 1.2, r'$\hookrightarrow \mathcal{N}(0, I)$', ha='center', va='center', fontsize=14, color='navy')
    ax.text(5, y_pos + 1.2, 'Normalizing Flow', ha='center', va='center', fontsize=18, color='navy')

    plt.tight_layout()
    plt.savefig('notes/chapters/assets/ch33_transform_chain.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    draw_transform_chain()
