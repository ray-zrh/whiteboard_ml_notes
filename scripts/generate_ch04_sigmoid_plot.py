import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    # Ensure output directory exists
    # Script is in scripts/, so we go up one level then to notes/chapters/assets
    output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
    os.makedirs(output_dir, exist_ok=True)

    z = np.linspace(-10, 10, 200)
    sigma = sigmoid(z)

    plt.figure(figsize=(8, 5))
    plt.plot(z, sigma, 'b-', linewidth=2, label=r'$\sigma(z) = \frac{1}{1+e^{-z}}$')

    # Asymptotes
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0.5, color='k', linestyle=':', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Annotations
    plt.text(5, 0.9, r'$\lim_{z \to \infty} \sigma(z) = 1$', fontsize=12)
    plt.text(-8, 0.1, r'$\lim_{z \to -\infty} \sigma(z) = 0$', fontsize=12)
    plt.text(0.5, 0.45, r'$\sigma(0) = 0.5$', fontsize=12)

    plt.title('Sigmoid Function')
    plt.xlabel('z')
    plt.ylabel(r'$\sigma(z)$')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    plt.tight_layout()

    output_path = os.path.join(output_dir, "ch04_sigmoid.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
