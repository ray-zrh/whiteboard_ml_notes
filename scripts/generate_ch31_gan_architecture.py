import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def draw_gan_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Set font to support Chinese
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # Define box styles
    generator_props = dict(boxstyle="round,pad=0.8", fc="#e6ffe6", ec="#00cc00", lw=2)
    discriminator_props = dict(boxstyle="round,pad=0.8", fc="#ffe6e6", ec="#cc0000", lw=2)
    data_props = dict(boxstyle="round,pad=0.5", fc="white", ec="#333333", lw=1.5)

    # Coordinates
    x_noise = 0.1
    x_gen = 0.35
    x_fake_data = 0.6
    x_real_data = 0.6
    x_disc = 0.85

    y_top = 0.7   # Path for real data
    y_bot = 0.3   # Path for generator

    # Nodes
    ax.text(x_noise, y_bot, r"潜在噪声\nLatent Noise\n$z \sim p_z(z)$", ha="center", va="center", size=14, bbox=data_props)
    ax.text(x_gen, y_bot, "生成器\nGenerator $G$", ha="center", va="center", size=16, fontweight='bold', bbox=generator_props)
    ax.text(x_fake_data, y_bot, "假数据\nFake Data\n$G(z)$", ha="center", va="center", size=14, bbox=data_props)

    ax.text(x_real_data, y_top, r"真实数据\nReal Data\n$x \sim p_{data}(x)$", ha="center", va="center", size=14, bbox=data_props)

    # Discriminator spans both paths conceptually, we place it in between at the end
    y_mid = 0.5
    ax.text(x_disc, y_mid, "判别器\nDiscriminator $D$", ha="center", va="center", size=16, fontweight='bold', bbox=discriminator_props)

    # Output probabilities
    ax.text(x_disc + 0.12, y_top, "$D(x)$ 预测为真的概率\n(目标: $\to 1$)", ha="left", va="center", size=12, color="green")
    ax.text(x_disc + 0.12, y_bot, "$D(G(z))$ 预测为真的概率\n(目标: $\to 0$)", ha="left", va="center", size=12, color="red")

    # Helper function to draw arrows
    def draw_arrow(ax, start, end, rad=0, color='#333333'):
        ax.annotate("", xy=end, xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, connectionstyle=f"arc3,rad={rad}"))

    # Arrows for Gen path
    draw_arrow(ax, (x_noise+0.1, y_bot), (x_gen-0.08, y_bot))
    draw_arrow(ax, (x_gen+0.1, y_bot), (x_fake_data-0.08, y_bot))
    draw_arrow(ax, (x_fake_data+0.08, y_bot), (x_disc-0.08, y_mid-0.05), rad=-0.1)

    # Arrows for Real Data path
    draw_arrow(ax, (x_real_data+0.1, y_top), (x_disc-0.08, y_mid+0.05), rad=0.1)

    # Arrows out of discriminator
    draw_arrow(ax, (x_disc+0.08, y_mid+0.05), (x_disc+0.11, y_top-0.02), rad=-0.2)
    draw_arrow(ax, (x_disc+0.08, y_mid-0.05), (x_disc+0.11, y_bot+0.02), rad=0.2)


    # Title
    plt.suptitle("GAN 数学架构 (GAN Mathematical Architecture)", fontsize=18, fontweight='bold', y=0.95)

    # Objective Function
    formula = r"$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))]$"
    ax.text(0.5, 0.05, formula, ha="center", va="center", size=16, bbox=dict(boxstyle="round,pad=0.5", fc="#f9f9f9", ec="#999999", lw=1))

    # Limits
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)

    output_path = os.path.join(output_dir, "ch31_gan_architecture.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    draw_gan_architecture()
