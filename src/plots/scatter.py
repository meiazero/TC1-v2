import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_true, y_pred, ax=None, figsize=(8, 6)):
    """Plot actual vs. predicted values scatter with y=x line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Actual vs Predicted")
    return fig, ax