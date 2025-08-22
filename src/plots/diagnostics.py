import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_qq(residuals, ax=None, figsize=(6, 6)):
    """Create a QQ-plot of residuals against a normal distribution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("QQ-Plot of Residuals")
    return fig, ax