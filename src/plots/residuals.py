import matplotlib.pyplot as plt

def plot_residuals(y_true, y_pred, ax=None, figsize=(8, 6)):
    """Plot residuals vs. predicted values."""
    residuals = y_true - y_pred
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    return fig, ax