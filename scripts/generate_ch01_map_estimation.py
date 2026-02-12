import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def generate_map_plot():
    # Parameters
    # Prior P(\theta)
    mu_prior = 0
    sigma_prior = 2.0

    # Likelihood P(X|\theta) - represented as a function of \theta
    # Assume we observed x = 4
    # Likelihood is proportional to N(\theta | x, sigma^2)
    x_observed = 4.0
    sigma_likelihood = 1.0

    # Posterior P(\theta|X) parameters (analytical solution for Gaussian)
    # precision = 1/variance
    tau_prior = 1 / sigma_prior**2
    tau_likelihood = 1 / sigma_likelihood**2
    tau_posterior = tau_prior + tau_likelihood

    sigma_posterior = np.sqrt(1 / tau_posterior)
    mu_posterior = (tau_prior * mu_prior + tau_likelihood * x_observed) / tau_posterior

    # Generate distribution data
    x = np.linspace(-6, 8, 500)
    pdf_prior = stats.norm.pdf(x, mu_prior, sigma_prior)
    pdf_likelihood = stats.norm.pdf(x, x_observed, sigma_likelihood)
    pdf_posterior = stats.norm.pdf(x, mu_posterior, sigma_posterior)

    # Scaling to match the visual reference roughly
    # The reference image has Likelihood taller than Posterior which is mathematically inconsistent
    # with normalized PDFs (since posterior variance < likelihood variance).
    # However, to produce a "textbook-like" schematic that matches the user's provided image style:
    # We will just plot the normalized PDFs. The user's image might have used different variances
    # or arbitrary scaling.
    # Color scheme: Prior=Blue, Likelihood=Orange, Posterior=Purple

    plt.figure(figsize=(8, 6))

    # Plot distributions
    plt.plot(x, pdf_prior, label=r'Prior $P(\theta)$', color='#1f77b4', linewidth=2)
    plt.plot(x, pdf_likelihood, label=r'Likelihood $P(\mathbf{X}|\theta)$', color='#ff7f0e', linewidth=2)
    plt.plot(x, pdf_posterior, label=r'Posterior $P(\theta|\mathbf{X})$', color='purple', linewidth=2)

    # Plot MAP Vertical Line
    plt.axvline(mu_posterior, color='black', linestyle='--', alpha=0.8)

    # Labels and Legend
    plt.xlabel(r'Parameter $\theta$', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)

    # X-axis ticks: Only show theta_MAP
    plt.xticks([mu_posterior], [r'$\theta_{\mathrm{MAP}}$'], fontsize=14)
    plt.tick_params(axis='x', length=0) # Hide tick marks

    # Y-axis: Hide ticks and values
    plt.yticks([])

    # Legend
    plt.legend(fontsize=12, loc='upper left', framealpha=1.0)

    # Adjust layout
    plt.ylim(bottom=0)
    plt.xlim(-6, 8)
    plt.grid(False)

    # Remove top and right spines
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)

    # Create output directory if it doesn't exist
    import os
    os.makedirs('notes/chapters/assets', exist_ok=True)

    # Save
    output_path = 'notes/chapters/assets/ch01_map_estimation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_map_plot()
