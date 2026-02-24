import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.linewidth": 0
})

def main():
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-6, 6, 500)

    # Red curve: "true" broad distribution
    y_red = 0.4 * np.exp(-0.08 * (x - 1.5)**2) + 0.1
    # Black curve: double counted
    y_black = 0.8 * np.exp(-0.4 * (x - 0.5)**2)
    # Blue curve: triple counted / very sharp
    y_blue = 1.3 * np.exp(-1.5 * (x + 0.2)**2)

    ax.plot(x, y_red, color='#e53935', linewidth=2.5, label='Broad Distribution')
    ax.plot(x, y_black, color='#424242', linewidth=2.5, label='Sharper (Double Counted)')
    ax.plot(x, y_blue, color='#1e88e5', linewidth=2.5, label='Very Sharp (Triple Counted)')

    ax.axis('off')
    ax.set_title("Double Counting $\\Rightarrow$ Distribution becomes too sharp", color='#d32f2f', fontsize=16)

    # Add text to simulate the \propto labels
    ax.text(-2, -0.2, r'$\propto$', color='#1e88e5', fontsize=18, ha='center')
    ax.text(0, -0.2, r'$\propto \times \propto$', color='#424242', fontsize=18, ha='center')
    ax.text(2, -0.2, r'$\propto$', color='#e53935', fontsize=18, ha='center')

    plt.tight_layout()

    os.makedirs('notes/chapters/assets', exist_ok=True)
    out_path = 'notes/chapters/assets/ch29_double_counting.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved diagram to {out_path}")

if __name__ == "__main__":
    main()
