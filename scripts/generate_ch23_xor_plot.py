import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import rcParams

def generate_xor_plot():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, '../notes/chapters/assets')
    os.makedirs(assets_dir, exist_ok=True)

    rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
    rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw axes
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1)

    # XOR points
    # class 0: (0,0), (1,1) -> represented as filled black dots
    ax.plot([0, 1], [0, 1], 'ko', markersize=10, label='Class 0 (相同)')
    # class 1: (0,1), (1,0) -> represented as open circles or different symbol
    ax.plot([0, 1], [1, 0], 'wo', markeredgecolor='b', markersize=10, markeredgewidth=2, label='Class 1 (不同)')

    # Add text labels for coordinates
    ax.text(0-0.1, 0-0.1, '(0,0)', ha='right', va='top', fontsize=12)
    ax.text(1+0.1, 1+0.1, '(1,1)', ha='left', va='bottom', fontsize=12)
    ax.text(0-0.1, 1+0.1, '(0,1)', ha='right', va='bottom', fontsize=12)
    ax.text(1+0.1, 0-0.1, '(1,0)', ha='left', va='top', fontsize=12)

    # Highlight that class 1 is enclosed or cannot be separated by a single line
    ellipse = Ellipse(xy=(0.5, 0.5), width=0.4, height=1.6, angle=-45,
                      edgecolor='b', fc='None', lw=2, linestyle='--')
    ax.add_patch(ellipse)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title("XOR Problem (异或问题)", fontsize=16)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    output_path = os.path.join(assets_dir, 'ch23_xor_problem.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"XOR Problem plot generated at {output_path}")

if __name__ == "__main__":
    generate_xor_plot()
