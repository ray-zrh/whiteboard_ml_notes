import numpy as np
import matplotlib.pyplot as plt

def draw_inverse_derivative():
    fig, ax = plt.subplots(figsize=(8, 8))

    # Data
    x = np.linspace(0, 5, 100)
    y_x = x
    # Let f(x) = e^(x-1) and f^-1(x) = ln(x) + 1 for demonstration
    x_curve = np.linspace(1e-2, 5, 100)
    y_f = np.exp(x_curve - 1.5)
    y_finv = np.log(x_curve) + 1.5

    # Plot lines
    ax.plot(x, y_x, 'k-', label='$y=x$')
    ax.plot(x_curve, y_f, 'r-', label='$y=f(x)$')
    ax.plot(x_curve, y_finv, 'b-', label='$y=f^{-1}(x)$')

    # Points A and B
    a = 2.0
    b = np.exp(a - 1.5)

    # Actually wait, f(a) = b, so a=x, b=y for f. For f^-1, input is b, output is a.
    # Let's pick a nice point
    b_val = 2.5
    a_val = np.log(b_val) + 1.5 # So f(a) = b, f^-1(b) = a

    ax.plot(a_val, b_val, 'ro') # Point on f(x)
    ax.text(a_val - 0.1, b_val + 0.2, '$B(b, a)$', color='red', fontsize=12, ha='right')

    ax.plot(b_val, a_val, 'bo') # Point on f^-1(x)
    ax.text(b_val + 0.2, a_val, '$A(a, b)$', color='navy', fontsize=12)

    # Point C on y=x
    ax.plot(a_val, a_val, 'ko')
    ax.text(a_val - 0.2, a_val - 0.2, '$C$', color='black', fontsize=12)

    # Draw triangles
    # Triangle for B
    ax.plot([a_val, a_val], [a_val, b_val], 'k--')
    ax.plot([a_val, b_val], [a_val, a_val], 'k--')
    ax.plot([a_val, b_val], [b_val, a_val], 'k--') # Connect A and B

    # Labels
    ax.text((a_val+b_val)/2, a_val - 0.2, '$a-c$', ha='center', fontsize=10)
    ax.text(a_val - 0.4, (a_val+b_val)/2, '$b-c$', va='center', fontsize=10)

    # Axis
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    # Tangent lines approximation
    slope_f = np.exp(a_val - 1.5)
    tangent_f = slope_f * (x - a_val) + b_val
    ax.plot(x, tangent_f, 'r--', alpha=0.5)

    slope_finv = 1/b_val
    tangent_finv = slope_finv * (x - b_val) + a_val
    ax.plot(x, tangent_finv, 'b--', alpha=0.5)

    ax.legend(loc='lower right')
    plt.title('Change of Variables Theorem: Inverse Function Derivative', color='navy', fontsize=14)

    plt.tight_layout()
    plt.savefig('notes/chapters/assets/ch33_inverse_derivative.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    draw_inverse_derivative()
