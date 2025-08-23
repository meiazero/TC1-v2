import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_qq(residuals, model_name=None, split_name=None, ax=None, figsize=(6, 6)):
    """Create a QQ-plot of residuals against a normal distribution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Ordered Values")
    # title
    if model_name and split_name:
        title = f"{model_name} - {split_name.capitalize()} Set\nQQ-Plot of Residuals"
    else:
        title = "QQ-Plot of Residuals"
    ax.set_title(title)
    return fig, ax