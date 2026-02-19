
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def f(x, a0=-0.3, a1=0.5):
    return a0 + a1 * x

def get_posterior(X, y, sigma_sq, tau_sq):
    # X: (N, 2) with bias term
    # y: (N, )
    # Prior: w ~ N(0, tau^2 I)

    N, D = X.shape
    Sigma_p = tau_sq * np.eye(D)
    Sigma_p_inv = (1/tau_sq) * np.eye(D)

    # Sigma_N_inv = (1/sigma^2) X^T X + Sigma_p_inv
    Sigma_N_inv = (1/sigma_sq) * (X.T @ X) + Sigma_p_inv
    Sigma_N = np.linalg.inv(Sigma_N_inv)

    # mu_N = Sigma_N * (1/sigma^2) * X^T y
    mu_N = Sigma_N @ ((1/sigma_sq) * X.T @ y)

    return mu_N, Sigma_N

def plot_samples(ax, w_samples, x_range, color='red', alpha=0.5, label=None):
    # w_samples: (K, 2)
    X_line = np.column_stack([np.ones_like(x_range), x_range])
    for i, w in enumerate(w_samples):
        y_line = X_line @ w
        if i == 0:
            ax.plot(x_range, y_line, color=color, alpha=alpha, label=label)
        else:
            ax.plot(x_range, y_line, color=color, alpha=alpha)

# --- Configuration ---
N_total = 25
noise_std = 0.2
sigma_sq = noise_std**2
tau_sq = 2.0 # Prior variance
alpha = 1/tau_sq

# Generate Data
x_true = np.random.uniform(-1, 1, N_total)
y_true = f(x_true) + np.random.normal(0, noise_std, N_total)

# Feature Matrix (with bias)
Phi_total = np.column_stack([np.ones(N_total), x_true])

# Grid for plotting
x_grid = np.linspace(-1, 1, 100)
Phi_grid = np.column_stack([np.ones(100), x_grid])

# --- Plot 1: Samples from Prior and Posterior ---
fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Prior
w_prior_mean = np.zeros(2)
w_prior_cov = tau_sq * np.eye(2)
# Sample 10 lines from prior
w_samples_prior = np.random.multivariate_normal(w_prior_mean, w_prior_cov, size=15)

axes[0].set_title(f'1. Prior Samples (No Data)\n$w \sim \mathcal{{N}}(0, {tau_sq}I)$')
plot_samples(axes[0], w_samples_prior, x_grid, color='#e74c3c', alpha=0.5, label='Samples from Prior')
axes[0].set_ylim(-2, 2)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()


# 2. Posterior (after 5 points)
N_obs = 5
X_obs = Phi_total[:N_obs]
y_obs = y_true[:N_obs]
x_obs_raw = x_true[:N_obs]

mu_N, Sigma_N = get_posterior(X_obs, y_obs, sigma_sq, tau_sq)
w_samples_post = np.random.multivariate_normal(mu_N, Sigma_N, size=15)

axes[1].set_title(f'2. Posterior Samples (N={N_obs})\nLikelihood + Prior')
axes[1].scatter(x_obs_raw, y_obs, s=80, facecolors='none', edgecolors='black', linewidth=1.5, label='Data', zorder=5)
plot_samples(axes[1], w_samples_post, x_grid, color='#2ecc71', alpha=0.5, label='Samples from Posterior')
axes[1].plot(x_grid, f(x_grid), 'k--', alpha=0.6, label='True Function')
axes[1].set_ylim(-2, 2)
axes[1].set_xlabel('x')
axes[1].legend()

plt.tight_layout()
out1 = "notes/chapters/assets/ch19_bayesian_samples.png"
fig1.savefig(out1, dpi=300)
print(f"Saved {out1}")


# --- Plot 2: Predictive Distribution ---
fig2, ax = plt.subplots(figsize=(10, 6))

# Use 20 points
N_pred = 20
X_pred_train = Phi_total[:N_pred]
y_pred_train = y_true[:N_pred]
x_pred_raw = x_true[:N_pred]

# Calculate Posterior
mu_N_pred, Sigma_N_pred = get_posterior(X_pred_train, y_pred_train, sigma_sq, tau_sq)

# Calculate Predictive Distribution
# y_hat = mu_N^T x_new
y_pred_mean = Phi_grid @ mu_N_pred

# sigma_star^2 = sigma^2 + x^T Sigma_N x
# Compute variance for each point in grid
pred_var = []
for i in range(len(x_grid)):
    phi_x = Phi_grid[i] # (2,)
    # sigma^2 + phi^T Sigma_N phi
    var_i = sigma_sq + phi_x.T @ Sigma_N_pred @ phi_x
    pred_var.append(var_i)
pred_var = np.array(pred_var)
pred_std = np.sqrt(pred_var)

# Plot
ax.set_title(f'3. Predictive Distribution (N={N_pred})\nMean + 2 Std Dev Region')
ax.scatter(x_pred_raw, y_pred_train, s=60, facecolors='none', edgecolors='black', linewidth=1.5, label='Data', zorder=5)
ax.plot(x_grid, f(x_grid), 'k--', label='True Function')
ax.plot(x_grid, y_pred_mean, color='#3498db', linewidth=2, label='Predictive Mean')

# Uncertainty Region
ax.fill_between(x_grid, y_pred_mean - 2*pred_std, y_pred_mean + 2*pred_std,
                color='#3498db', alpha=0.2, label='Uncertainty (Mean $\pm$ 2$\sigma$)')

ax.set_ylim(-2, 2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper left')

out2 = "notes/chapters/assets/ch19_predictive_distribution.png"
fig2.savefig(out2, dpi=300)
print(f"Saved {out2}")
