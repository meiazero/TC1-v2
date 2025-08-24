import numpy as np
from scipy.stats import kurtosis, pearsonr, skew, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def r2(y_true, y_pred):
    """Coefficient of determination"""
    return r2_score(y_true, y_pred)

def mse(y_true, y_pred):
    """Mean squared error"""
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    """Root mean squared error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean absolute error"""
    return mean_absolute_error(y_true, y_pred)

def medae(y_true, y_pred):
    """Median absolute error"""
    return median_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    """Mean absolute percentage error"""
    try:
        return mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        return np.nan

def pearson_corr(y_true, y_pred):
    """Pearson correlation coefficient"""
    try:
        corr, _ = pearsonr(y_true, y_pred)
    except Exception:
        corr = np.nan
    return corr

def spearman_corr(y_true, y_pred):
    """Spearman correlation coefficient"""
    try:
        corr, _ = spearmanr(y_true, y_pred)
    except Exception:
        corr = np.nan
    return corr

def residual_stats(residuals):
    """Compute basic statistics of residuals: mean, variance, skewness, kurtosis"""
    return {
        "mean": np.mean(residuals),
        "var": np.var(residuals),
        "skew": skew(residuals),
        "kurtosis": kurtosis(residuals)
    }
