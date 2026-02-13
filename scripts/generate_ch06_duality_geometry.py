import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
output_dir = "notes/chapters/assets"
os.makedirs(output_dir, exist_ok=True)

def plot_set_G(ax, u, t_boundary, p_star_idx, lambda_val, title, convex=True):
    # Fill area
    t_upper = t_boundary + 5
    ax.fill_between(u, t_boundary, t_upper, color='#dddddd' if convex else '#ffdddd', alpha=0.5, label='Set $G$')
    ax.plot(u, t_boundary, 'k-', lw=2)

    # Axes
    ax.axvline(0, color='k', lw=1)
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('$u$ (constraint $m(x)$)')
    ax.set_ylabel('$t$ (objective $f(x)$)')
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(0, 10)

    # Primal Optimal p*: Lowest point in G where u <= 0
    # Find min t for u <= 0
    u_le_0 = u[u <= 0]
    t_le_0 = t_boundary[u <= 0]
    if len(t_le_0) > 0:
        p_idx = np.argmin(t_le_0)
        p_star_u = u_le_0[p_idx]
        p_star_val = t_le_0[p_idx]
    else:
        # Fallback if no points <= 0
        p_star_u = 0
        p_star_val = t_boundary[np.argmin(np.abs(u))]

    ax.scatter(p_star_u, p_star_val, color='red', s=80, zorder=5, label='Primal Optimal $p^*$')
    ax.text(p_star_u - 0.4, p_star_val + 0.3, '$p^*$', color='red', fontweight='bold')

    # Draw horizontal line at p*
    # ax.axhline(p_star_val, color='red', linestyle='--', alpha=0.3)

    # Dual: Supporting hyperplane with slope -lambda
    # We want max intercept d* such that line is below G
    # For a fixed lambda, intercept g(lambda) = min (t + lambda*u) over G
    # If we choose lambda optimally (dual optimal), we get d*

    # Calculate optimal intercept d* for the given lambda
    intercepts = t_boundary + lambda_val * u
    d_star = np.min(intercepts)

    # Draw line: t = -lambda * u + d_star
    u_line = np.linspace(-3, 4, 100)
    t_line = -lambda_val * u_line + d_star

    ax.plot(u_line, t_line, 'g-', lw=2, label=f'Supporting Hyperplane (slope $\lambda={lambda_val}$)')

    # Mark intercept d*
    ax.scatter(0, d_star, color='green', s=60, zorder=5)
    ax.text(0.1, d_star, '$g(\lambda) = d^*$', color='green', fontweight='bold', va='bottom')

    # Show Gap if exists
    if p_star_val - d_star > 0.1:
        ax.annotate(
            '', xy=(0.05, p_star_val), xytext=(0.05, d_star),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5)
        )
        ax.text(0.2, (p_star_val + d_star)/2, 'Duality Gap', color='purple', va='center')

    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# ---------------------------------------------------------
# Plot 1: Non-Convex Case (Duality Gap)
# Use a "bean" shape or a higher order polynomial with a dent
u1 = np.linspace(-2, 3, 500)
# Create a non-convex boundary: sum of two parabolas or adding a sine wave
# t = (u-1)^2 + 3 + 1.5 * sin(2u)
# Let's try something that dips up at u=0 but p* is higher
# t = 0.5(u+1)^2 + 4 if u < 0
# t = ... complex shape
# Let's use: t = (u^2) + 2 + 2*exp(-5*(u-0.5)^2) ? No, that's bump down.
# We want a bump UP at the bottom to make it non-convex (a dent).
# Like a "W" shape but tilted.
t1 = 0.5 * (u1 + 0.5)**2 + 4 - 2 * np.exp(-2 * (u1 + 0.5)**2)
# Adjustment to make p* distinct from d*
# Let's use piecewise to be sure.
# Or simpler: t = u^2 + 5 + 2*sin(2u)
t1 = 0.5 * u1**2 + 5 + 1.5 * np.sin(2 * u1)
# p* is min for u<=0.
# At u=0, t=5. At u=-1, t=0.5+5+1.5*sin(-2)=5.5-1.3=4.2.
# Slope at potential tangent points.
# Supporting hyperplane will touch at some u > 0 and u < 0 maybe.
# Let's use a simpler shape: a set union of two circles? No.
# Just use the whiteboard sketch idea: a "kidney" shape.
# We define the boundary explicitly.
# t = (u-1.5)^2 + 3. But we add a "dent" near u=0.
# t = (u)^2 + 4 + 3 * exp(-10 * (u+0.2)**2) -> bump up near u=-0.2
t1 = 0.2 * (u1 - 1)**2 + 4 + 2 * np.exp(-5 * (u1 + 0.2)**2)

# p* is at u ~ -1 where curve is low.
# Min t for u<=0 is around u=-1.5?
# Let's check slope.
# We pick a lambda that supports the global bottom.
lambda1 = 0.5 # Slope -0.5
plot_set_G(ax1, u1, t1, 0, lambda1, 'Non-Convex Set (Weak Duality)\n$d^* < p^*$ (Duality Gap)', convex=False)


# ---------------------------------------------------------
# Plot 2: Convex Case (Strong Duality)
# t = (u-1)^2 + 2 (Convex parabola)
# u<=0 min is at u=0, t=3. p*=3.
# Slope at u=0 is 2(0-1) = -2. So lambda=2.
u2 = np.linspace(-2, 3.5, 500)
t2 = 0.8 * (u2 - 1.5)**2 + 2
# p* is at u=0 (boundary). t = 0.8*2.25 + 2 = 1.8 + 2 = 3.8.
# Slope at u=0 is 1.6(-1.5) = -2.4. Lambda = 2.4.
lambda2 = 2.4
plot_set_G(ax2, u2, t2, 0, lambda2, 'Convex Set + Slater (Strong Duality)\n$d^* = p^*$ (Zero Gap)', convex=True)

plt.tight_layout()

# Save the figure
output_path = os.path.join(output_dir, "ch06_duality_geometry.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
