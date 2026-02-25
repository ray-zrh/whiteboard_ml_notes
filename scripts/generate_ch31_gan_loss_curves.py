import matplotlib.pyplot as plt
import numpy as np
import os

def create_loss_curve_diagram():
    output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False
    })
    # Set font to support Chinese
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(9, 6))

    # X axis range representing D(G(z)) probability [0.001 to 0.999]
    x = np.linspace(0.001, 0.999, 400)

    # Loss Functions for Generator G
    # 1. Minimax Loss: minimize log(1 - D(G(z))) -> equivalent to plotting log(1-x)
    y_minimax = np.log(1 - x)

    # 2. Non-Saturating Heuristic Loss: maximize log(D(G(z))) -> minimize -log(D(G(z)))
    # We plot the negative so it represents a "Loss" to be minimized similar to minimax
    y_heuristic = -np.log(x)

    # Plot
    ax.plot(x, y_minimax, '-', color='#1565c0', lw=3.0, label=r'Minimax Objective: $\log(1 - D(G(z)))$', alpha=0.9)
    ax.plot(x, y_heuristic, '--', color='#e65100', lw=3.0, label=r'Non-Saturating (Heuristic): $-\log(D(G(z)))$', alpha=0.9)

    # Styling Axes
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['$0$\n(Poor Fake)', '$1$\n(Perfect Fake)'], size=12)
    # Remove y ticks for cleaner theoretical look
    ax.set_yticks([])

    # Axis Labels
    ax.text(1.05, 0.1, r'$D(G(z))$', size=16, ha='center', va='center')
    ax.text(-0.05, 4.0, r'Loss $J^{(G)}$', size=16, ha='center', va='center', rotation=90)

    # Annotate vanishing gradient
    ax.annotate(
        "梯度消失 (Vanishing Gradient)\n曲线平缓，学习信号微弱",
        xy=(0.1, -0.1), xycoords='data',
        xytext=(0.3, -2.0), textcoords='data',
        arrowprops=dict(facecolor='#1565c0', shrink=0.05, width=2, headwidth=8),
        horizontalalignment='center', verticalalignment='top',
        fontsize=12, color='#1565c0',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1565c0", lw=1)
    )

    # Annotate strong gradient
    ax.annotate(
        "梯度很大 (Strong Gradient)\n提供强大的学习信号",
        xy=(0.1, 2.3), xycoords='data',
        xytext=(0.4, 3.5), textcoords='data',
        arrowprops=dict(facecolor='#e65100', shrink=0.05, width=2, headwidth=8),
        horizontalalignment='center', verticalalignment='top',
        fontsize=12, color='#e65100',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e65100", lw=1)
    )

    # Title and Legend
    ax.legend(loc='lower right', frameon=True, fontsize=12)
    plt.suptitle("生成器损失函数对比 (Generator Loss Functions)", fontsize=18, fontweight='bold', y=0.95)

    # Save
    output_path = os.path.join(output_dir, 'ch31_gan_loss_curves.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_loss_curve_diagram()
