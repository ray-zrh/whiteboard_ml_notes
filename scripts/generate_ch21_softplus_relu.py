import matplotlib.pyplot as plt
import numpy as np
import os

def create_softplus_relu_diagram():
    output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # X axis range
    x = np.linspace(-5, 5, 400)

    # Functions
    y_relu = np.maximum(0, x)
    y_softplus = np.log(1 + np.exp(x))

    # Plot
    ax.plot(x, y_relu, '-', color='#1565c0', lw=2.5, label='ReLU', alpha=0.9)
    ax.plot(x, y_softplus, '--', color='#e65100', lw=3.0, label=r'Softplus: $\log(1+e^x)$', alpha=0.9)

    # Styling Axes
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Remove tick marks for cleaner theoretical look
    ax.set_xticks([])
    ax.set_yticks([])

    # Labels and Arrows imitating hand-drawn axes
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # Annotate functions directly on the lines
    ax.text(2.5, 2.0, 'ReLU', color='#1565c0', fontsize=16, fontweight='bold', rotation=35,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    ax.text(-2.5, 1.0, 'Softplus', color='#e65100', fontsize=18, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    ax.text(-2.5, 0.5, r'$\log(1 + e^x)$', color='#e65100', fontsize=16)

    # Title
    ax.text(0, 5.5, 'Smooth Approximation: Softplus vs ReLU', fontsize=18, ha='center', color='#212121', fontweight='bold')

    # Save
    output_path = os.path.join(output_dir, 'ch21_softplus_relu.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_softplus_relu_diagram()
