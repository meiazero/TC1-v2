import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


def plot_learning_curve(
    estimator,
    X,
    y,
    title: str = '',
    cv=5,
    n_jobs=None,
    train_sizes=None
):
    """
    Generate a plot of the learning curve for a given estimator.

    Args:
        estimator: scikit-learn estimator
        X: feature matrix
        y: target vector
        title: Title for the chart
        cv: cross-validation strategy or integer
        n_jobs: number of jobs for parallel computation
        train_sizes: relative or absolute numbers of training examples
    Returns:
        fig: matplotlib Figure object
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    train_sizes_abs, train_scores, val_scores, fit_times, score_times = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring='r2',
        return_times=True
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    ax.grid()
    ax.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color='r'
    )
    ax.fill_between(
        train_sizes_abs,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.1,
        color='g'
    )
    ax.plot(
        train_sizes_abs,
        train_scores_mean,
        'o-',
        color='r',
        label='Training score'
    )
    ax.plot(
        train_sizes_abs,
        val_scores_mean,
        'o-',
        color='g',
        label='Cross-validation score'
    )
    ax.legend(loc='best')
    fig.tight_layout()
    return fig