
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def draw_pgm_framework():
    fig, ax = plt.subplots(figsize=(14, 10))

    # Hide axes
    ax.axis('off')

    # Set font to support Chinese
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # Define box style
    root_props = dict(boxstyle="round,pad=0.6", fc="#e6f3ff", ec="#0066cc", lw=2)
    l1_props = dict(boxstyle="round,pad=0.5", fc="#fff9e6", ec="#ffcc00", lw=1.5)
    l2_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=1)
    l3_props = dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#aaaaaa", lw=1, ls="--")

    # Coordinates
    x_shift = 0.05  # Margin shift

    # Root
    root_xy = (0.05 + x_shift, 0.5)

    # Level 1
    l1_x = 0.25 + x_shift
    l1_y_rep = 0.8
    l1_y_inf = 0.5
    l1_y_lrn = 0.2

    # Level 2 - Representation
    l2_x_rep = 0.55 + x_shift
    l2_y_dir = 0.9
    l2_y_undir = 0.8
    l2_y_gauss = 0.7

    # Level 2 - Inference
    l2_x_inf = 0.55 + x_shift
    l2_y_exact = 0.55
    l2_y_approx = 0.45

    # Level 3 - Approximate Inference
    l3_x_inf = 0.85 + x_shift
    l3_y_det = 0.48
    l3_y_stoch = 0.42

    # Level 2 - Learning
    l2_x_lrn = 0.55 + x_shift
    l2_y_param = 0.25
    l2_y_struct = 0.15

    # Level 3 - Parameter Learning
    l3_x_lrn = 0.85 + x_shift
    l3_y_complete = 0.28
    l3_y_latent = 0.22


    # Draw Lines helper
    def connect_hv(xy1, xy2, color='#555555'):
        # Horizontal then Vertical connection
        mid_x = (xy1[0] + xy2[0]) / 2
        ax.plot([xy1[0], mid_x, mid_x, xy2[0]], [xy1[1], xy1[1], xy2[1], xy2[1]], color=color, lw=1.5, zorder=0)

    def connect_direct(xy1, xy2, color='#555555'):
        ax.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], color=color, lw=1.5, zorder=0)

    # Add an invisible anchor at x=0 to ensure margin is preserved
    ax.text(0, 0.5, ".", alpha=0)

    # Connect Root to L1
    connect_hv(root_xy, (l1_x, l1_y_rep))
    connect_hv(root_xy, (l1_x, l1_y_inf))
    connect_hv(root_xy, (l1_x, l1_y_lrn))

    # Connect L1 Rep to L2
    connect_hv((l1_x, l1_y_rep), (l2_x_rep, l2_y_dir))
    connect_hv((l1_x, l1_y_rep), (l2_x_rep, l2_y_undir))
    connect_hv((l1_x, l1_y_rep), (l2_x_rep, l2_y_gauss))

    # Connect L1 Inf to L2
    connect_hv((l1_x, l1_y_inf), (l2_x_inf, l2_y_exact))
    connect_hv((l1_x, l1_y_inf), (l2_x_inf, l2_y_approx))

    # Connect L2 Approx to L3
    connect_hv((l2_x_inf, l2_y_approx), (l3_x_inf, l3_y_det))
    connect_hv((l2_x_inf, l2_y_approx), (l3_x_inf, l3_y_stoch))

    # Connect L1 Learn to L2
    connect_hv((l1_x, l1_y_lrn), (l2_x_lrn, l2_y_param))
    connect_hv((l1_x, l1_y_lrn), (l2_x_lrn, l2_y_struct))

    # Connect L2 Param to L3
    connect_hv((l2_x_lrn, l2_y_param), (l3_x_lrn, l3_y_complete))
    connect_hv((l2_x_lrn, l2_y_param), (l3_x_lrn, l3_y_latent))


    # Draw Boxes (Text)
    # Root
    ax.text(root_xy[0], root_xy[1], "Probabilistic\nGraphical Model\n(概率图)", ha="center", va="center", size=14, fontweight="bold", bbox=root_props)

    # Level 1
    ax.text(l1_x, l1_y_rep, "Representation\n(表示)", ha="center", va="center", size=12, bbox=l1_props)
    ax.text(l1_x, l1_y_inf, "Inference\n(推断)", ha="center", va="center", size=12, bbox=l1_props)
    ax.text(l1_x, l1_y_lrn, "Learning\n(学习)", ha="center", va="center", size=12, bbox=l1_props)

    # Level 2 - Rep
    ax.text(l2_x_rep, l2_y_dir, "Directed Graph (有向图)\nBayesian Network", ha="center", va="center", size=10, bbox=l2_props)
    ax.text(l2_x_rep, l2_y_undir, "Undirected Graph (无向图)\nMarkov Network", ha="center", va="center", size=10, bbox=l2_props)
    ax.text(l2_x_rep, l2_y_gauss, "Gaussian Graph (高斯图)\n(Continuous Variables)", ha="center", va="center", size=10, bbox=l2_props)

    # Level 2 - Inf
    ax.text(l2_x_inf, l2_y_exact, "Exact Inference\n(精确推断)", ha="center", va="center", size=10, bbox=l2_props)
    ax.text(l2_x_inf, l2_y_approx, "Approximate Inference\n(近似推断)", ha="center", va="center", size=10, bbox=l2_props)

    # Level 3 - Inf Details
    ax.text(l3_x_inf, l3_y_det, "Deterministic (确定性)\nVariational Inference", ha="center", va="center", size=9, bbox=l3_props)
    ax.text(l3_x_inf, l3_y_stoch, "Stochastic (随机)\nMCMC", ha="center", va="center", size=9, bbox=l3_props)

    # Level 2 - Learn
    ax.text(l2_x_lrn, l2_y_param, "Parameter Learning\n(参数学习)", ha="center", va="center", size=10, bbox=l2_props)
    ax.text(l2_x_lrn, l2_y_struct, "Structure Learning\n(结构学习)", ha="center", va="center", size=10, bbox=l2_props)

    # Level 3 - Learn Details
    ax.text(l3_x_lrn, l3_y_complete, "Complete Data\n(完备数据)", ha="center", va="center", size=9, bbox=l3_props)
    ax.text(l3_x_lrn, l3_y_latent, "Incomplete/Latent\n(隐变量) -> EM", ha="center", va="center", size=9, bbox=l3_props)

    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "ch09_pgm_framework.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    draw_pgm_framework()
