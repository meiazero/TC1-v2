import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

# pretty name mapping for metrics
_PRETTY_NAMES = {
    "mae_test": "MAE",
    "mape_test": "MAPE",
    "mse_test": "MSE",
    "medae_test": "MedAE",
    "pearson_test": "Pearson",
    "r2_test": "R2",
    "res_mean_test": "Residual Mean",
    "res_var_test": "Residual Variance",
    "res_skew_test": "Residual Skewness",
    "res_kurt_test": "Residual Kurtosis",
    "rmse_test": "RMSE",
    "spearman_test": "Spearman"
}


def plot_summary_boxplots(df, output_dir, metrics=None, model_col='model'):
    """
    For each metric in `metrics`, plot a boxplot grouping values by model and
    save the PNGs to `output_dir`.
    If metrics is None, selects all numeric columns ending with '_test'.
    """
    # determine which metrics to plot
    if metrics is None:
        # select all numeric test metrics
        metrics = [c for c in df.columns if c.endswith('_test')
                   and np.issubdtype(df[c].dtype, np.number)]

    for metric in metrics:
        # get display name for metric
        default_name = metric.replace('_', ' ').title()
        display_name = _PRETTY_NAMES.get(metric, default_name)

        groups = []
        labels = []
        for model_name, grp in df.groupby(model_col):
            vals = grp[metric].dropna().tolist()
            if vals:
                groups.append(vals)
                labels.append(model_name)
        if not groups:
            continue

        # create figure
        fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.5), 6))
        # boxplot with means
        meanprops = {'marker': 'd', 'markerfacecolor': 'orange'}
        ax.boxplot(groups, labels=labels, showmeans=True, meanprops=meanprops)
        # titles and labels
        ax.set_title(f"{display_name} by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel(str(display_name))
        # legend for mean marker
        mean_handle = mlines.Line2D([], [], color='orange', marker='d', linestyle='None',
                                    markersize=6, label='Mean')
        ax.legend(handles=[mean_handle], loc='upper right', fontsize='small')

        # descriptive annotation
        fig.text(
            0.5, 0.01,
            'Box: IQR (25th-75th percentile); Whiskers: 1.5$\\times$IQR; Marker: Mean',
            ha='center', va='bottom', fontsize=8
        )
        # adjust layout to fit annotation
        fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
        # save
        fname = f"{metric}_boxplot.png"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=300)
        plt.close(fig)