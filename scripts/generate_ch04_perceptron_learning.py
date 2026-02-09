
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
output_dir = "notes/chapters/assets"
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def perceptron_learning_process():
    np.random.seed(42)

    # 1. Generate linearly separable data
    n_samples = 20
    # Class 1: Centered at (2, 2)
    X1 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
    # Class -1: Centered at (0, 0)
    X2 = np.random.randn(n_samples, 2) * 0.5 + np.array([0, 0])

    # Combine
    X = np.vstack([X1, X2])
    # Add bias term (column of 1s)
    X_bias = np.hstack([np.ones((2*n_samples, 1)), X])

    y = np.hstack([np.ones(n_samples), -np.ones(n_samples)])

    # Shuffle
    indices = np.arange(2*n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    X_bias = X_bias[indices]
    y = y[indices]

    # Perceptron Algorithm with snapshots
    w = np.zeros(3) # [w0, w1, w2] i.e. [bias, weight_x, weight_y]
    lr = 1.0

    snapshots = []
    max_epochs = 10
    errors_log = []

    # Snapshot at initialization
    snapshots.append({'w': w.copy(), 'title': 'Initialization (w=0)', 'highlight': None})

    step_count = 0
    updates = 0

    for epoch in range(max_epochs):
        errors = 0
        for i in range(len(y)):
            # x_i includes bias at index 0
            # Prediction: sign(w^T x)
            # y in {-1, 1}
            if y[i] * np.dot(w, X_bias[i]) <= 0:
                # Mistake found!
                # Snapshot before update (showing the mistake)
                if updates in [0, 2, 5]: # Capture 1st, 3rd, 6th update
                    snapshots.append({
                        'w': w.copy(),
                        'title': f'Update #{updates+1}: Mistake Found',
                        'highlight': X[i] # Original feature vector without bias for plotting
                    })

                # Update
                w += lr * y[i] * X_bias[i]
                updates += 1
                errors += 1

                # Snapshot after update
                if updates in [1, 3, 6]:
                     snapshots.append({
                        'w': w.copy(),
                        'title': f'After Update #{updates}',
                        'highlight': X[i]
                    })

        errors_log.append(errors)
        if errors == 0:
            break

    # Final state
    snapshots.append({'w': w.copy(), 'title': f'Converged (Total {updates} updates)', 'highlight': None})

    # Plotting snapshots
    # We will select 4 key moments to show in a 2x2 grid
    # 0: Init
    # 1: After 1st update
    # 2: After 3rd update
    # 3: Converged

    # We need to filter snapshots to get distinct meaningful states
    # Let's pick: Init, After Update 1, After Update 3, Converged
    selected_indices = [0]
    # Find index for 'After Update #1'
    for idx, s in enumerate(snapshots):
        if s['title'] == 'After Update #1':
            selected_indices.append(idx)
            break
    for idx, s in enumerate(snapshots):
        if s['title'] == 'After Update #3':
            selected_indices.append(idx)
            break
    selected_indices.append(len(snapshots) - 1)


    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    x_min, x_max = -2, 4
    y_min, y_max = -2, 4
    xx = np.linspace(x_min, x_max, 100)

    for ax, idx in zip(axes, selected_indices):
        state = snapshots[idx]
        curr_w = state['w']

        # Plot data
        ax.scatter(X1[:, 0], X1[:, 1], c='blue', marker='o', label='Pos (+1)' if idx==0 else "")
        ax.scatter(X2[:, 0], X2[:, 1], c='red', marker='x', label='Neg (-1)' if idx==0 else "")

        # Plot Decision Boundary: w0 + w1*x + w2*y = 0 => y = -(w1*x + w0)/w2
        if curr_w[2] != 0:
            yy = -(curr_w[1] * xx + curr_w[0]) / curr_w[2]
            ax.plot(xx, yy, 'k-', linewidth=2)
        elif curr_w[1] != 0: # Vertical line
            x_line = -curr_w[0] / curr_w[1]
            ax.axvline(x_line, color='k', linewidth=2)

        # Highlight misclassified point if any
        if state['highlight'] is not None:
             ax.scatter(state['highlight'][0], state['highlight'][1], s=200, facecolors='none', edgecolors='green', linewidth=2, label='Mistake')
             # Draw arrow for normal vector w (optional, but w changes)

        ax.set_title(state['title'], fontsize=12, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if idx == 0:
            ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ch04_perceptron_process.png"), dpi=300)
    plt.close()
    print("Saved ch04_perceptron_process.png")

    # Plot Convergence (Errors vs Epochs)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(range(1, len(errors_log)+1), errors_log, 'bo-', linewidth=2)
    ax2.set_xlabel('Epoch (Iteration over dataset)')
    ax2.set_ylabel('Number of Misclassifications')
    ax2.set_title('Perceptron Convergence Analysis')
    ax2.set_xticks(range(1, len(errors_log)+1))
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ch04_perceptron_convergence.png"), dpi=300)
    plt.close()
    print("Saved ch04_perceptron_convergence.png")

if __name__ == "__main__":
    perceptron_learning_process()
