import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def generate_timeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, '../notes/chapters/assets')
    os.makedirs(assets_dir, exist_ok=True)

    # Configure matplotlib for Chinese characters if possible, or fallback to sans-serif
    rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
    rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(15, 8), facecolor='white')

    # Timeline axis
    ax.axhline(0, color='black', linewidth=2)

    # Create an arrow for the t-axis
    ax.annotate('', xy=(2020, 0), xytext=(1950, 0),
                arrowprops=dict(arrowstyle="->", color='black', lw=2))
    ax.text(2018, 0.5, 't', fontsize=20, fontstyle='italic')

    # Data points
    # (Year, Above/Below label position (y), Label, Text color, Connecting line style)
    events_above = [
        (1958, 3.0, "PLA\n(Perceptron)", 'navy'),
        (1981, 5.0, "MLP -> Feedforward NN", 'navy'),
        (1986, 2.5, "RNN", 'navy'),
        (1986, 7.0, "BP + MLP", 'navy'),
        (1989, 4.5, "CNN", 'navy'),
        (1997, 3.0, "LSTM", 'navy'),
        (2006, 7.5, "Deep Belief Network <-> RBM\nDeep Autoencoder\n(效果 > SVM)", 'navy'),
        (2009, 2.0, "GPU", 'navy'),
        (2011, 4.0, "Speech", 'navy'),
        (2012, 6.0, "ImageNet", 'navy'),
        (2013, 2.5, "VAE\n(data, compute, GPU)", 'navy'),
        (2014, 4.5, "GAN", 'navy'),
        (2016, 6.5, "图模型\n(GNN)", 'navy')
    ]

    events_below = [
        (1969, -2.5, "PLA has limitation\n(Minsky)", 'green'),
        (1969, -4.5, "解决不了 Non-linear\n-> XOR Problem", 'green'),
        (1986, -5.5, "Universal Approximation Theorem\n(>=1 hidden layer -> 逼近函数)", 'green'),
        (1986, -8.0, "1 layer is good -> why deep?\nBP -> 梯度消失\n(AI 寒冬)", 'green'),
        (1993, -2.5, "SVM (kernel + theory)\n-> SVM 演进", 'green'),
        (1995, -4.5, "AdaBoost, Random Forest", 'green')
    ]

    # Plot above timeline
    for year, y_pos, label, color in events_above:
        # Plot point on timeline
        ax.plot(year, 0, 'ko', markersize=8)
        # Draw line and text
        ax.plot([year, year], [0, y_pos - 0.2], color=color, linewidth=1.5, linestyle='-')
        ax.text(year, y_pos, label, ha='center', va='bottom', fontsize=11,
                color=color, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax.text(year, -0.5, str(year), ha='center', va='top', fontsize=10, fontweight='bold')

    # Plot below timeline
    for year, y_pos, label, color in events_below:
        # Plot point on timeline (only if not already plotted)
        if not any(year == e[0] for e in events_above):
            ax.plot(year, 0, 'ko', markersize=8)
            ax.text(year, 0.5, str(year), ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Connection line types based on the image's curved/straight lines
        ax.annotate(label, xy=(year, 0), xytext=(year+2 if year == 1969 else year, y_pos),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.5,
                                    connectionstyle="angle3,angleA=90,angleB=0"),
                    ha='left' if year == 1969 else 'center', va='top', fontsize=11, color=color,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # General styling
    ax.set_xlim(1955, 2022)
    ax.set_ylim(-10, 10)
    ax.axis('off')
    ax.set_title("从感知机到深度学习 (From Perceptron to Deep Learning)", fontsize=26, pad=30)

    output_path = os.path.join(assets_dir, 'ch23_timeline.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Timeline generated at {output_path}")

if __name__ == "__main__":
    generate_timeline()
