import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def draw_gan_concept():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Hide axes
    ax.axis('off')

    # Set font to support Chinese
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # Define box style
    box_props = dict(boxstyle="round,pad=0.6", fc="white", ec="#333333", lw=1.5)
    real_data_props = dict(boxstyle="round,pad=0.6", fc="#e6f3ff", ec="#0066cc", lw=1.5)
    fake_data_props = dict(boxstyle="round,pad=0.6", fc="#fff9e6", ec="#ffcc00", lw=1.5)
    discriminator_props = dict(boxstyle="round,pad=0.8", fc="#ffe6e6", ec="#cc0000", lw=2)
    generator_props = dict(boxstyle="round,pad=0.8", fc="#e6ffe6", ec="#00cc00", lw=2)

    # Base coords
    # Center lines
    y_real_path = 0.75
    y_fake_path = 0.25
    y_discriminator = 0.5

    # x coords
    x_source = 0.1
    x_db = 0.32
    x_product = 0.54
    x_judge = 0.77
    x_result = 0.98

    # Level 1: Real Data (Top)
    ax.text(x_source, y_real_path, "古人\nAncients\n($P_{data}$)", ha="center", va="center", size=14, bbox=real_data_props)
    ax.text(x_db, y_real_path, "收藏库\nCollection", ha="center", va="center", size=14, bbox=real_data_props)
    ax.text(x_product, y_real_path, "国宝\nReal Data ($x$)", ha="center", va="center", size=14, bbox=real_data_props)

    # Level 2: Generated Data (Bottom)
    ax.text(x_source, y_fake_path, "随机灵感\nNoise ($z \\sim P_z$)", ha="center", va="center", size=14, bbox=fake_data_props)
    generator_txt = ax.text(x_db, y_fake_path, "工艺大师\nGenerator\n$G(z; \\theta_g)$", ha="center", va="center", size=14, bbox=generator_props)
    ax.text(x_product, y_fake_path, "高仿工艺品\nFake Data ($G(z)$)", ha="center", va="center", size=14, bbox=fake_data_props)

    # Discriminator (Middle Right)
    discriminator_txt = ax.text(x_judge, y_discriminator, "鉴赏专家\nDiscriminator\n$D(x; \\theta_d)$", ha="center", va="center", size=16, fontweight='bold', bbox=discriminator_props)

    # Output (Far Right)
    ax.text(x_result, y_real_path, "真品！\n(Real, $D(x)$ 接近 1)", ha="center", va="center", size=14, color="green", fontweight='bold')
    ax.text(x_result, y_fake_path, "赝品！\n(Fake, $D(G(z))$ 接近 0)", ha="center", va="center", size=14, color="red", fontweight='bold')


    # Draw Arrows
    def draw_arrow(ax, start, end, color='#333333', style='-', label='', rad=0):
        # ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=1.5, ls=style, connectionstyle=f"arc3,rad={rad}"))
        ax.annotate(label, xy=end, xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, ls=style, connectionstyle=f"arc3,rad={rad}"),
                    ha='center', va='center', size=12, color=color, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # Real path arrows
    draw_arrow(ax, (x_source+0.06, y_real_path), (x_db-0.08, y_real_path))
    draw_arrow(ax, (x_db+0.06, y_real_path), (x_product-0.08, y_real_path))
    draw_arrow(ax, (x_product+0.08, y_real_path), (x_judge-0.08, y_real_path-0.04), rad=0.2)
    draw_arrow(ax, (x_judge+0.08, y_discriminator+0.06), (x_result-0.08, y_real_path-0.02), rad=-0.2)

    # Fake path arrows
    draw_arrow(ax, (x_source+0.08, y_fake_path), (x_db-0.08, y_fake_path))
    draw_arrow(ax, (x_db+0.08, y_fake_path), (x_product-0.08, y_fake_path))
    draw_arrow(ax, (x_product+0.1, y_fake_path), (x_judge-0.08, y_discriminator-0.06), rad=-0.2)
    draw_arrow(ax, (x_judge+0.08, y_discriminator-0.06), (x_result-0.08, y_fake_path+0.05), rad=0.2)

    # Feedback loops
    draw_arrow(ax, (x_judge, y_discriminator+0.12), (x_judge, y_discriminator+0.32), color='red', style='--', label='更新 $D$ 以识别真伪 (Max $V(D,G)$)')

    # Generator feedback loop (needs to go from discriminator back to generator)
    x_mid_fb = (x_db + x_judge) / 2
    y_bottom_fb = y_fake_path - 0.14

    # Multi-segment arrow for generator feedback
    ax.annotate("", xy=(x_db, y_fake_path-0.08), xytext=(x_db, y_bottom_fb), arrowprops=dict(arrowstyle="->", color='blue', lw=2, ls='--'))
    ax.plot([x_judge, x_judge], [y_discriminator-0.08, y_bottom_fb], color='blue', lw=2, ls='--')
    ax.plot([x_judge, x_db], [y_bottom_fb, y_bottom_fb], color='blue', lw=2, ls='--')
    ax.text(x_mid_fb, y_bottom_fb-0.02, '更新 $G$ 以骗过专家 (Min $V(D,G)$)', ha='center', va='bottom', size=12, color='blue', fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # Title
    plt.suptitle("生成对抗网络 (Generative Adversarial Network) - 鉴赏家与工艺大师", fontsize=20, fontweight='bold', y=0.95)

    # Value Function formula
    formula = r"$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim P_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim P_z(z)}[\log (1 - D(G(z)))]$"
    ax.text(0.5, 0.04, formula, ha="center", va="center", size=15, bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="#999999", lw=1))

    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15) # Adjust for title and formula

    output_path = os.path.join(output_dir, "ch31_gan_concept.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    draw_gan_concept()
