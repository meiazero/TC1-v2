import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def plot_actual_vs_predicted(y_true, y_pred, model_name=None, split_name=None, ax=None, figsize=(8, 6)):
    """Plot Actual vs Predicted scatter with perfect-fit line, trend line and R2 score."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # scatter points
    ax.scatter(y_true, y_pred, alpha=0.5, label="Data points")

    # perfect fit line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val],
            color="red", linestyle="--", label="Perfect fit")

    # trend line
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    x_line = np.array([min_val, max_val])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="green", linestyle="-",
            label=f"Trend: $y={slope:.2f}x+{intercept:.2f}$")

    # R2 annotation
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.8))

    # formatting
    ax.legend(loc="upper right")

    # labels
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")

    # title
    if model_name and split_name:
        title = f"{model_name} - {split_name.capitalize()} Set\nActual vs Predicted"
    else:
        title = "Actual $\\times$ Predicted"
    ax.set_title(title)

    plt.tight_layout()

    return fig, ax