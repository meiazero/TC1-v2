import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

# pretty name mapping for base metrics
_PRETTY_NAMES = {
    "mae": "MAE",
    "mape": "MAPE",
    "mse": "MSE",
    "medae": "MedAE",
    "pearson": "Pearson",
    "r2": "R2",
    "res_mean": "Residual Mean",
    "res_var": "Residual Variance",
    "res_skew": "Residual Skewness",
    "res_kurt": "Residual Kurtosis",
    "rmse": "RMSE",
    "spearman": "Spearman"
}


def plot_summary_boxplots(df, output_dir, metrics=None, model_col='model'):
    """
    For each metric in `metrics`, plot a boxplot grouping values by model and
    save the PNGs to `output_dir`.
    If metrics is None, selects all numeric columns ending with '_train' or '_test'.
    """
    # determine which metrics to plot
    if metrics is None:
        # select all numeric test and train metrics
        metrics = [
            c for c in df.columns
            if (c.endswith('_test') or c.endswith('_train'))
               and np.issubdtype(df[c].dtype, np.number)
        ]

    for metric in metrics:
        # determine display name using pretty mapping and suffix
        if metric in _PRETTY_NAMES:
            display_name = _PRETTY_NAMES[metric]
        else:
            display_name = None
            # check for train/test suffix
            if metric.endswith('_train'):
                suffix = 'Train'
                base = metric[:-6]
            elif metric.endswith('_test'):
                suffix = 'Test'
                base = metric[:-5]
            else:
                suffix = None
                base = None

            # if base metric has pretty name, append suffix
            if suffix and base in _PRETTY_NAMES:
                display_name = f"{_PRETTY_NAMES[base]} ({suffix})"

            # fallback to default formatting
            if display_name is None:
                display_name = metric.replace('_', ' ').title()

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
        fname = f"{metric}_boxplot.pdf"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=300)
        plt.close(fig)