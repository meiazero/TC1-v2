import numpy as np
import pandas as pd
from scipy.stats import t


def significance_test(model, X, y, feature_names=None):
    """
    Perform significance testing for model coefficients using t-test.

    Parameters:
        model: fitted linear model with attributes coef_ and intercept_.
        X: array-like, shape (n_samples, n_features), design matrix used for fitting.
        y: array-like, shape (n_samples,), target vector.
        feature_names: list of feature names of length n_features.

    Returns:
        DataFrame with columns: feature, coef, std_err, t_stat, p_value.
    """
    # Ensure numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape
    # Check model has coefficients
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not have coef_ attribute for significance testing.")
    # Design matrix with intercept
    X_design = np.hstack((np.ones((n_samples, 1)), X))
    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    # Degrees of freedom: n_samples - (n_features + 1)
    df = n_samples - n_features - 1
    if df <= 0:
        raise ValueError("Degrees of freedom <= 0 for significance test.")
    # Residual variance estimate
    sigma2 = np.sum(residuals**2) / df
    # Compute covariance matrix of coefficients
    XtX_inv = np.linalg.inv(X_design.T.dot(X_design))
    cov_beta = sigma2 * XtX_inv
    # Standard errors (first is intercept)
    std_err = np.sqrt(np.diag(cov_beta))
    # Coefficients including intercept
    coef = np.concatenate(([float(model.intercept_)], np.asarray(model.coef_, dtype=float).ravel()))
    # t-statistics
    t_stats = coef / std_err
    # two-sided p-values
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df))
    # Feature names
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]
    features = ["intercept"] + list(feature_names)
    # Build DataFrame
    df_stats = pd.DataFrame({
        "feature": features,
        "coef": coef,
        "std_err": std_err,
        "t_stat": t_stats,
        "p_value": p_values
    })
    return df_stats