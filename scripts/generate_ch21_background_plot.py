import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def create_energy_state_diagram():
    output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.linewidth": 0
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # --- Left side: Physical System ---
    system_center = (2.0, 4.0)
    system_radius = 1.2

    # Draw System Boundary
    circle_v = patches.Circle(system_center, system_radius, facecolor='#f0f8ff', edgecolor='#2c3e50', lw=2)
    ax.add_patch(circle_v)
    ax.text(system_center[0] + system_radius + 0.2, system_center[1], r'$V$', fontsize=16, color='#2c3e50', va='center')
    ax.text(system_center[0], system_center[1] + system_radius + 0.3, "Statistical Physics:\nA Physical System", fontsize=14, ha='center', color='#34495e', fontweight='bold')

    # Draw Particles
    np.random.seed(42)
    for _ in range(8):
        # random position within the circle
        r = np.random.uniform(0, system_radius - 0.2)
        theta = np.random.uniform(0, 2*np.pi)
        px = system_center[0] + r * np.cos(theta)
        py = system_center[1] + r * np.sin(theta)
        p_circle = patches.Circle((px, py), 0.08, facecolor='#e74c3c', edgecolor='#c0392b', lw=1)
        ax.add_patch(p_circle)

    # Annotation for particles
    ax.annotate("Particles\n(Molecules/Atoms)", xy=(system_center[0] - 0.5, system_center[1] - 0.5), xytext=(system_center[0] - 1.5, system_center[1] - 2.0),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#7f8c8d'), fontsize=12, color='#7f8c8d')


    # --- Right side: Energy and Probability ---
    ax.text(6.0, 5.2, "Boltzmann Distribution (Gibbs Distribution)", fontsize=16, ha='center', color='#2980b9', fontweight='bold')

    # Formula connecting State and Energy
    ax.text(4.5, 4.2, r"System State $\rightarrow$ Energy $E$", fontsize=14, color='#2c3e50')
    ax.text(4.5, 3.5, r"$P(\text{state}) \propto \exp \left\{ -\frac{E}{kT} \right\}$", fontsize=18, color='#8e44ad')

    # Energy vs Probability
    ax.text(4.5, 2.5, r"$E \uparrow \quad \Longrightarrow \quad P(\text{state}) \downarrow$", fontsize=14, color='#c0392b', fontweight='bold')
    ax.text(8.0, 2.5, "(Unstable)", fontsize=14, color='#c0392b', va='center')
    ax.text(4.5, 1.8, r"$E \downarrow \quad \Longrightarrow \quad P(\text{state}) \uparrow$", fontsize=14, color='#27ae60', fontweight='bold')
    ax.text(8.0, 1.8, "(Stable)", fontsize=14, color='#27ae60', va='center')


    # --- Bottom: State Transition Timeline ---
    # Draw timeline arrow
    ax.annotate("", xy=(8.5, 0.5), xytext=(1.5, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color='#34495e'))
    ax.text(5.0, 1.0, "State Transition Direction (Natural Evolution)", fontsize=14, ha='center', color='#34495e', fontweight='bold', alpha=0.9)

    # States mapped on timeline
    # Young
    ax.text(2.5, 0.3, "Young", fontsize=14, ha='center', color='#e67e22', fontweight='bold')
    ax.text(2.5, -0.1, r"High $E$ (Unstable)", fontsize=11, ha='center', color='#d35400')
    ax.text(2.5, 0.5, "|", fontsize=16, ha='center', va='center', color='#34495e')

    # Aging
    ax.text(5.0, 0.3, "Aging", fontsize=14, ha='center', color='#f39c12', fontweight='bold')
    ax.text(5.0, -0.1, r"Next State", fontsize=11, ha='center', color='#e67e22')
    ax.text(5.0, 0.5, "|", fontsize=16, ha='center', va='center', color='#34495e')

    # Death
    ax.text(7.5, 0.3, "Death", fontsize=14, ha='center', color='#7f8c8d', fontweight='bold')
    ax.text(7.5, -0.1, r"Lowest $E$ (Stable)", fontsize=11, ha='center', color='#95a5a6')
    ax.text(7.5, 0.5, "|", fontsize=16, ha='center', va='center', color='#34495e')


    output_path = os.path.join(output_dir, 'ch21_energy_state_transition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_energy_state_diagram()
