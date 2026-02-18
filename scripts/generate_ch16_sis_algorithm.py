
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_sis_algorithm_diagram():
    # Style settings
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 0
    })

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')

    # Colors
    color_particle = '#1f77b4' # Blue
    color_propagate = '#d62728' # Red arrow
    color_weight = '#2ca02c'   # Green annotation

    # Time Steps
    times = [2, 6, 10]
    labels = [r"$t=0$", r"$t=1$", r"$t=2$"]

    # Fake Particle Data (y positions)
    # t=0: uniform weights (equal size)
    p0_y = [2.5, 3.5, 4.5, 5.5, 6.5]
    w0 = [1.0] * 5

    # t=1: weights vary
    p1_y = [2.0, 3.8, 4.2, 5.8, 7.0]
    w1 = [0.5, 1.2, 0.8, 1.5, 0.4]

    # t=2: weights vary more (degeneracy starts)
    p2_y = [1.5, 4.0, 4.5, 6.0, 7.5]
    w2 = [0.05, 0.2, 0.1, 4.0, 0.05]   # one very big weight

    data = [
        (p0_y, w0),
        (p1_y, w1),
        (p2_y, w2)
    ]

    # Draw Time Labels at Bottom
    for i, t_x in enumerate(times):
        # Time Label
        ax.text(t_x, 0.0, labels[i], ha='center', va='center', fontsize=18, fontweight='bold', color='#333333')

        # Vertical dashed line separator
        if i < len(times) - 1:
            mid_x = (times[i] + times[i+1]) / 2
            ax.vlines(mid_x, 0.5, 8.0, colors='gray', linestyles=':', alpha=0.3)

    # Draw Particles
    for i, (ys, ws) in enumerate(data):
        curr_x = times[i]

        # Plot particles
        for j, y in enumerate(ys):
            weight = ws[j]
            # Size proportional to weight visualization
            # Scale for visibility
            radius_base = 0.15
            radius = radius_base * (weight**0.4)

            alpha = min(1.0, 0.3 + 0.6 * (weight / max(ws)))

            circle = patches.Circle((curr_x, y), radius=radius,
                                    facecolor=color_particle, edgecolor='black', alpha=alpha, zorder=10)
            ax.add_patch(circle)

            # Draw Transition Arrows from previous step
            if i > 0:
                prev_x = times[i-1]
                prev_ys = data[i-1][0]
                prev_y = prev_ys[j]

                # Curvy Arrow for natural flow
                # Determine curvature based on y difference
                connection_style = "arc3,rad=0.1" if abs(y - prev_y) > 0.5 else "arc3,rad=0"

                ax.annotate("", xy=(curr_x - radius - 0.1, y), xytext=(prev_x + radius_base + 0.1, prev_y),
                            arrowprops=dict(arrowstyle="->", connectionstyle=connection_style,
                                            color=color_propagate, alpha=0.4, lw=1.5))

    # --- Annotations ---

    # Step 1: Initialization (Bottom, above time label)
    ax.text(times[0], 1.0, "Initialization\n" + r"$w_0^{(i)} = 1/N$", ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='#f0f0f0', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    # Step 2: Propagation (Top, between columns)
    mid_x1 = (times[0] + times[1]) / 2
    ax.text(mid_x1, 7.8, "Prediction Step\n" + r"$z_1^{(i)} \sim q(z_1|z_0, x_1)$", ha='center', va='center', fontsize=12, color=color_propagate, fontweight='bold')

    # Step 2: Update (Bottom)
    ax.text(times[1], 1.0, "Update Step\n" + r"$w_1^{(i)} \propto w_0^{(i)} \frac{p(x_1|z_1^{(i)}) p(z_1^{(i)}|z_0^{(i)})}{q(\cdot)}$",
            ha='center', va='center', fontsize=11,
            bbox=dict(facecolor='#f0f0f0', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    # Step 3: Propagation (Top)
    mid_x2 = (times[1] + times[2]) / 2
    ax.text(mid_x2, 7.8, "Prediction Step\n" + r"$z_2^{(i)} \sim q(z_2|z_1, x_2)$", ha='center', va='center', fontsize=12, color=color_propagate, fontweight='bold')

    # Step 3: Update / Degeneracy (Bottom)
    ax.text(times[2], 1.0, "Update / Degeneracy\n" + r"$w_2^{(i)}$ Variance Increases",
            ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='#ffebee', alpha=0.9, edgecolor='#d32f2f', boxstyle='round,pad=0.5'))

    # Highlight High Weight Particle
    large_p_idx = np.argmax(w2)
    large_p_y = p2_y[large_p_idx]
    ax.annotate(r"Dominant Particle (High $w$)", xy=(times[2]+0.3, large_p_y), xytext=(times[2]+1.5, large_p_y),
                arrowprops=dict(arrowstyle="->", color=color_weight, lw=2), fontsize=12, color=color_weight, fontweight='bold')

    # Highlight Low Weight Particle
    small_p_idx = np.argmin(w2)
    small_p_y = p2_y[small_p_idx]
    ax.annotate(r"Negligible Particle (Low $w$)", xy=(times[2]+0.1, small_p_y), xytext=(times[2]+1.5, small_p_y-0.5),
                arrowprops=dict(arrowstyle="->", color='gray', lw=1.5), fontsize=12, color='gray')


    output_path = os.path.join(output_dir, 'ch16_sis_algorithm.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_sis_algorithm_diagram()
