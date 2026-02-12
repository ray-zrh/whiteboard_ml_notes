
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def plot_svd_geometry():
    # Setup figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3)

    # 1. Original Unit Circle
    ax = axes[0]
    ax.set_title("1. Original Space (x)", fontsize=14)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Draw unit circle
    circle = patches.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)

    # Draw basis vectors
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    ax.arrow(0, 0, v1[0], v1[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='$v_1$')
    ax.arrow(0, 0, v2[0], v2[1], head_width=0.1, head_length=0.1, fc='red', ec='red', label='$v_2$')
    ax.legend(loc='upper right')

    # Define SVD components
    # V^T (Rotation) - Rotate by 45 degrees
    theta = np.radians(45)
    V_T = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

    # Sigma (Scaling) - Scale x by 1.5, y by 0.5
    Sigma = np.array([[1.5, 0],
                      [0, 0.5]])

    # U (Rotation) - Rotate by 30 degrees
    phi = np.radians(30)
    U = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi), np.cos(phi)]])

    # 2. After Rotation V^T
    ax = axes[1]
    ax.set_title("2. Rotation ($V^T$)", fontsize=14)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Transformed vectors
    v1_rot = V_T @ v1
    v2_rot = V_T @ v2

    # Draw rotated circle (still a circle)
    circle = patches.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)

    ax.arrow(0, 0, v1_rot[0], v1_rot[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    ax.arrow(0, 0, v2_rot[0], v2_rot[1], head_width=0.1, head_length=0.1, fc='red', ec='red')

    # 3. After Scaling Sigma
    ax = axes[2]
    ax.set_title("3. Scaling ($\Sigma$)", fontsize=14)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)

    v1_scale = Sigma @ v1_rot
    v2_scale = Sigma @ v2_rot

    # Draw ellipse
    # Parametric equation for ellipse after transform
    t = np.linspace(0, 2*np.pi, 100)
    circle_points = np.stack([np.cos(t), np.sin(t)])
    ellipse_points = Sigma @ V_T @ circle_points
    ax.plot(ellipse_points[0, :], ellipse_points[1, :], color='gray', linestyle='--')

    ax.arrow(0, 0, v1_scale[0], v1_scale[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='$\sigma_1 v_1$')
    ax.arrow(0, 0, v2_scale[0], v2_scale[1], head_width=0.1, head_length=0.1, fc='red', ec='red', label='$\sigma_2 v_2$')

    # 4. Final Rotation U
    ax = axes[3]
    ax.set_title("4. Final Rotation ($U$)", fontsize=14)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)

    v1_final = U @ v1_scale
    v2_final = U @ v2_scale

    # Final ellipse points
    final_points = U @ ellipse_points
    ax.plot(final_points[0, :], final_points[1, :], color='gray', linestyle='--')

    ax.arrow(0, 0, v1_final[0], v1_final[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='$u_1 \sigma_1$')
    ax.arrow(0, 0, v2_final[0], v2_final[1], head_width=0.1, head_length=0.1, fc='red', ec='red', label='$u_2 \sigma_2$')

    plt.tight_layout()
    # Output path relative to project root where script is run
    output_path = "notes/chapters/assets/ch05_svd_geometry.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    plot_svd_geometry()
