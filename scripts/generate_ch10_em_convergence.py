
import numpy as np
import matplotlib.pyplot as plt

def log_likelihood(theta):
    """
    A non-convex log-likelihood function (e.g., mixture of two Gaussians).
    For simplicity, use a polynomial-like function with local optima.
    L(x) = -(x-2)^2 + 0.5*sin(3x) + 2
    """
    return -(theta - 2)**2 + 0.5 * np.sin(3 * theta) + 2

def lower_bound(theta, theta_t):
    """
    A quadratic lower bound B(theta; theta_t) that touches L(theta) at theta_t.
    B(theta) = L(theta_t) - k * (theta - theta_t)^2
    We need to choose k such that it forces B(theta) <= L(theta).
    This is illustrative.
    """
    # Calculate gradient at theta_t to match slope
    grad = -2 * (theta_t - 2) + 1.5 * np.cos(3 * theta_t)

    # Quadratic term (curvature) needs to be high enough to stay below L(x)
    curvature = 3 # Hand-tuned for visual

    return log_likelihood(theta_t) + grad * (theta - theta_t) - curvature * (theta - theta_t)**2

def main():
    # Setup
    theta = np.linspace(0, 4, 400)
    L_theta = log_likelihood(theta)

    # Iteration 1
    theta_0 = 0.5
    B_0 = lower_bound(theta, theta_0)
    theta_1 = theta[np.argmax(B_0)] # Argmax of lower bound

    # Iteration 2
    B_1 = lower_bound(theta, theta_1)
    theta_2 = theta[np.argmax(B_1)]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Plot Log-Likelihood
    ax.plot(theta, L_theta, 'k-', linewidth=2, label='Log-Likelihood $\log P(X|\\theta)$')

    # 2. Plot Lower Bounds
    ax.plot(theta, B_0, 'b--', linewidth=1.5, label='Lower Bound $B(\\theta; \\theta^{(t)})$')
    ax.plot(theta, B_1, 'g--', linewidth=1.5, label='Lower Bound $B(\\theta; \\theta^{(t+1)})$')

    # 3. Mark Points
    # Step 0: Start
    ax.scatter(theta_0, log_likelihood(theta_0), color='blue', s=50, zorder=5)
    ax.annotate(r'$\theta^{(t)}$', xy=(theta_0, log_likelihood(theta_0)), xytext=(theta_0 - 0.3, log_likelihood(theta_0) - 0.8),
                arrowprops=dict(arrowstyle='->', color='blue'), fontsize=12, color='blue', ha='center')

    # Step 1: E-step (Touch point) -> already implicitly shown by B_0 touching L at theta_0

    # Step 1: M-step (Maximize B_0)
    L_at_theta_1 = log_likelihood(theta_1)
    B_at_theta_1 = np.max(B_0)

    ax.scatter(theta_1, B_at_theta_1, color='blue', marker='x', s=50, zorder=5)
    ax.vlines(theta_1, B_at_theta_1, L_at_theta_1, colors='gray', linestyles=':')
    ax.annotate(r'M-step $\theta^{(t+1)}$', xy=(theta_1, B_at_theta_1), xytext=(theta_1, B_at_theta_1 + 0.5),
                arrowprops=dict(arrowstyle='->', color='blue'), fontsize=10, color='blue', ha='center')

    # Step 2: E-step (Update Bound to B_1)
    ax.scatter(theta_1, L_at_theta_1, color='green', s=50, zorder=5)
    # Arrow from M-step result up to likelihood
    ax.annotate('', xy=(theta_1, L_at_theta_1), xytext=(theta_1, B_at_theta_1),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(theta_1 + 0.1, (L_at_theta_1 + B_at_theta_1)/2, 'E-step', color='green', va='center', fontsize=9)

    # Step 2: M-step (Maximize B_1)
    B_at_theta_2 = np.max(B_1)
    ax.scatter(theta_2, B_at_theta_2, color='green', marker='x', s=50, zorder=5)
    ax.annotate(r'$\theta^{(t+2)}$', xy=(theta_2, B_at_theta_2), xytext=(theta_2 + 0.4, B_at_theta_2 + 0.1),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=12, color='green', ha='center')

    # Shade the gap (KL Divergence concept) - Optional but good context
    # Fill between L and B_0
    # ax.fill_between(theta, B_0, L_theta, where=(theta > 0) & (theta < 1.5), color='blue', alpha=0.1, label='Gap (KL Divergence)')

    ax.set_title("EM Algorithm Convergence: Lower Bound Maximization")
    ax.set_xlabel(r"Parameter $\theta$")
    ax.set_ylabel("Objective Function")
    ax.set_ylim(-3, 3)
    ax.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Save
    output_path = "notes/chapters/assets/ch10_em_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
