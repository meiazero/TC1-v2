import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gaussian_kde

def plot_residuals(y_true, y_pred, model_name=None, split_name=None, ax=None, figsize=(8, 6)):
    """Plot histogram of residuals with normal distribution overlay."""
    residuals = y_true - y_pred
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    # histogram of residuals
    ax.hist(residuals, bins=30, density=True, alpha=0.6,
            color='#0033b0', edgecolor='black', label='Residuals')

    # perfect fit line
    ax.axvline(0, color='magenta', linestyle='--', linewidth=2)

    # overlay normal distribution PDF
    mu, std = np.mean(residuals), np.std(residuals)
    x_vals = np.linspace(residuals.min(), residuals.max(), 100)
    p = norm.pdf(x_vals, mu, std)
    ax.plot(x_vals, p, color='orange', linewidth=2, label='Normal distribution ($\\mu$=0, $\\sigma$=1)')

    # overlay Gaussian KDE
    kde = gaussian_kde(residuals)
    x_vals = np.linspace(residuals.min(), residuals.max(), 100)
    p = kde(x_vals)
    ax.plot(x_vals, p, color='green', linewidth=2, label='Gaussian KDE')

    # formatting
    ax.set_xlim(residuals.min(), residuals.max())
    ax.legend()
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')

    # title
    if model_name and split_name:
        title = f"{model_name} - {split_name.capitalize()} Set\nResiduals Histogram"
    else:
        title = "Residuals Histogram"
    ax.set_title(title)
    return fig, ax