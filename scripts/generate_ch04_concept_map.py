
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Ensure the output directory exists
# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def draw_concept_map():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Hide axes
    ax.axis('off')

    # Set font to support Chinese
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # Define box style
    box_props = dict(boxstyle="round,pad=0.5", fc="white", ec="#333333", lw=1.5)
    root_props = dict(boxstyle="round,pad=0.6", fc="#e6f3ff", ec="#0066cc", lw=2)
    branch_props = dict(boxstyle="round,pad=0.5", fc="#fff9e6", ec="#ffcc00", lw=1.5)
    leaf_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=1)

    # Coordinates
    root_xy = (0.5, 0.9)

    # Level 1
    l1_left_xy = (0.25, 0.65)
    l1_right_xy = (0.75, 0.65)

    # Level 2 (Leaves)
    l2_ll_xy = (0.12, 0.35) # Hard -> LDA
    l2_lr_xy = (0.38, 0.35) # Hard -> Perceptron

    l2_rl_xy = (0.62, 0.35) # Soft -> Discriminative
    l2_rr_xy = (0.88, 0.35) # Soft -> Generative

    # Draw Lines
    def connect(xy1, xy2, color='#555555'):
        ax.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], color=color, lw=1.5, zorder=0)

    connect(root_xy, l1_left_xy)
    connect(root_xy, l1_right_xy)

    connect(l1_left_xy, l2_ll_xy)
    connect(l1_left_xy, l2_lr_xy)

    connect(l1_right_xy, l2_rl_xy)
    connect(l1_right_xy, l2_rr_xy)

    # Draw Boxes (Text)
    ax.text(root_xy[0], root_xy[1], "Linear Classification\n(线性分类)", ha="center", va="center", size=14, fontweight="bold", bbox=root_props)

    ax.text(l1_left_xy[0], l1_left_xy[1], "Hard Output\n(硬输出)\n$y \\in \\{-1, +1\\}$", ha="center", va="center", size=12, bbox=branch_props)
    ax.text(l1_right_xy[0], l1_right_xy[1], "Soft Output\n(软输出)\n$y \in [0, 1]$", ha="center", va="center", size=12, bbox=branch_props)

    ax.text(l2_ll_xy[0], l2_ll_xy[1], "Fisher's LDA\n(线性判别分析)", ha="center", va="center", size=10, bbox=leaf_props)
    ax.text(l2_lr_xy[0], l2_lr_xy[1], "Perceptron\n(感知机)", ha="center", va="center", size=10, bbox=leaf_props)

    ax.text(l2_rl_xy[0], l2_rl_xy[1], "Probabilistic\nDiscriminative\n(Logistic Regression)", ha="center", va="center", size=10, bbox=leaf_props)
    ax.text(l2_rr_xy[0], l2_rr_xy[1], "Probabilistic\nGenerative\n(GDA)", ha="center", va="center", size=10, bbox=leaf_props)

    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "ch04_concept_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    draw_concept_map()
