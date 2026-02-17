
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../notes/chapters/assets")
os.makedirs(output_dir, exist_ok=True)

def create_filtering_cycle_diagram():
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

    # Figure setup
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Colors
    c_posterior = '#e3f2fd' # Blueish
    c_prior = '#fff3e0'     # Orangeish
    c_obs = '#e0e0e0'       # Grey
    c_edge = '#333333'
    c_text = '#333333'
    c_math = '#d62728'

    # Box Geometry
    box_w = 3.5
    box_h = 1.8
    y_main = 4.0
    x_left = 2.5
    x_right = 11.5

    # 1. State Nodes (Posterior t-1, Prior t, Posterior t)

    # Node 1: Posterior t-1
    # box1 = patches.FancyBboxPatch((x_left - box_w/2, y_main - box_h/2), box_w, box_h,
    #                               boxstyle="round,pad=0.1", fc=c_posterior, ec=c_edge, lw=2)
    # ax.add_patch(box1)
    # ax.text(x_left, y_main + 0.4, "Posterior Time $t-1$", ha='center', va='center', fontweight='bold', fontsize=12)
    # ax.text(x_left, y_main - 0.3, r"$P(z_{t-1} | x_{1:t-1})$", ha='center', va='center', fontsize=16)

    # Actually, let's make it a cycle.
    # Top: Prediction
    # Bottom: Update

    # Layout:
    #      [Posterior z_t-1]  ---> (Prediction) ---> [Prior z_t]
    #                                                     |
    #                                                  (Update) <--- [Observation x_t]
    #                                                     v
    #                                              [Posterior z_t]
    #                                                     |
    #           <-----------------------------------------+

    # Coordinates
    x_post_prev = 3.0
    y_row1 = 6.0

    x_prior = 10.0
    y_row1 = 6.0

    x_post_curr = 10.0
    y_row2 = 2.0

    x_obs = 13.0 # Side input

    # Box 1: Posterior t-1 (Start)
    box1 = patches.FancyBboxPatch((x_post_prev - box_w/2, y_row1 - box_h/2), box_w, box_h,
                                  boxstyle="round,pad=0.1", fc=c_posterior, ec=c_edge, lw=2)
    ax.add_patch(box1)
    ax.text(x_post_prev, y_row1 + 0.5, "Posterior ($t-1$)", ha='center', va='center', fontweight='bold', fontsize=14, color=c_text)
    ax.text(x_post_prev, y_row1 - 0.2, r"$P(z_{t-1} | x_{1:t-1})$", ha='center', va='center', fontsize=18, color=c_text)

    # Box 2: Prior t (Prediction Result)
    box2 = patches.FancyBboxPatch((x_prior - box_w/2, y_row1 - box_h/2), box_w, box_h,
                                  boxstyle="round,pad=0.1", fc=c_prior, ec=c_edge, lw=2)
    ax.add_patch(box2)
    ax.text(x_prior, y_row1 + 0.5, "Prior / Prediction ($t$)", ha='center', va='center', fontweight='bold', fontsize=14, color=c_text)
    ax.text(x_prior, y_row1 - 0.2, r"$P(z_t | x_{1:t-1})$", ha='center', va='center', fontsize=18, color=c_text)

    # Arrow 1: Prediction Step
    # Integral formula text
    formula_pred = r"$\int P(z_t|z_{t-1}) \cdot P(z_{t-1}|x_{1:t-1}) d z_{t-1}$"

    ax.annotate("", xy=(x_prior - box_w/2 - 0.1, y_row1), xytext=(x_post_prev + box_w/2 + 0.1, y_row1),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=c_edge))
    ax.text((x_post_prev + x_prior)/2, y_row1 + 0.5, "Prediction", ha='center', va='bottom', fontsize=14, fontweight='bold', color='#1565c0')
    ax.text((x_post_prev + x_prior)/2, y_row1 - 0.8, formula_pred, ha='center', va='top', fontsize=13, color=c_edge)


    # Box 3: Posterior t (Update Result)
    box3 = patches.FancyBboxPatch((x_post_curr - box_w/2, y_row2 - box_h/2), box_w, box_h,
                                  boxstyle="round,pad=0.1", fc=c_posterior, ec=c_edge, lw=2)
    ax.add_patch(box3)
    ax.text(x_post_curr, y_row2 + 0.5, "Posterior ($t$)", ha='center', va='center', fontweight='bold', fontsize=14, color=c_text)
    ax.text(x_post_curr, y_row2 - 0.2, r"$P(z_t | x_{1:t})$", ha='center', va='center', fontsize=18, color=c_text)

    # Observation Input
    circle_obs = patches.Circle((x_post_curr + 2.5, (y_row1+y_row2)/2), 0.6, fc=c_obs, ec=c_edge, lw=2)
    ax.add_patch(circle_obs)
    ax.text(x_post_curr + 2.5, (y_row1+y_row2)/2, r"$x_t$", ha='center', va='center', fontsize=18)
    ax.text(x_post_curr + 2.5, (y_row1+y_row2)/2 + 0.9, "New\nObs", ha='center', va='center', fontsize=12)

    # Arrow 2: Update Step (Vertical Down)
    ax.annotate("", xy=(x_post_curr, y_row2 + box_h/2 + 0.1), xytext=(x_post_curr, y_row1 - box_h/2 - 0.1),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=c_edge))

    # Arrow from observation
    ax.annotate("", xy=(x_post_curr + 0.5, (y_row1+y_row2)/2), xytext=(x_post_curr + 1.9, (y_row1+y_row2)/2),
                arrowprops=dict(arrowstyle="->", lw=2, color=c_edge, ls='dashed'))

    # Formulas for Update
    mid_y = (y_row1 + y_row2) / 2
    ax.text(x_post_curr - 0.5, mid_y + 0.5, "Update", ha='right', va='center', fontsize=14, fontweight='bold', color='#2e7d32')
    ax.text(x_post_curr - 0.5, mid_y - 0.2, r"$\propto P(x_t|z_t) \cdot P(z_t|x_{1:t-1})$", ha='right', va='center', fontsize=13, color=c_edge) # Bayes

    # Arrow 3: Loop back (Next Step t -> t+1)
    # Curve from Box 3 to Box 1 is hard.
    # Conceptually, Box 3 *is* Box 1 for the next step.
    # Let's draw a dashed arrow looping back to indicate time step increase.

    path_patch = patches.FancyArrowPatch(
        (x_post_curr - box_w/2 - 0.1, y_row2),
        (x_post_prev, y_row2),
        connectionstyle="arc3,rad=0.0",
        arrowstyle="->", color=c_edge, lw=2, ls='dashed'
    )
    # Actually just a straight line left, then up

    # Draw simple lines for loop
    # Line Left
    ax.plot([x_post_curr - box_w/2, x_post_prev], [y_row2, y_row2], color=c_edge, lw=2, ls='--')
    # Line Up
    ax.annotate("", xy=(x_post_prev, y_row1 - box_h/2 - 0.1), xytext=(x_post_prev, y_row2),
                arrowprops=dict(arrowstyle="->", lw=2, color=c_edge, ls='--'))

    ax.text(x_post_prev + 0.2, (y_row1 + y_row2)/2, "Next Step\n$t \leftarrow t+1$", ha='left', va='center', fontsize=12, color='#555555')

    # Title
    ax.text(7.0, 7.5, "Recursive Filtering Process (Predict - Update)", ha='center', va='center', fontsize=20, fontweight='bold')

    output_path = os.path.join(output_dir, 'ch15_filtering_cycle.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_filtering_cycle_diagram()
