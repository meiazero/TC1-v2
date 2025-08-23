import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error
)
from scipy.stats import pearsonr, spearmanr, skew, kurtosis
from sklearn.model_selection import cross_validate, RepeatedKFold


def train_and_evaluate(model, name, params, X_train, y_train, X_test, y_test):
    results = {
        "model": name,
        "params": params,
        "train": {},
        "test": {}
    }

    try:
        # --- Repeated K-Fold cross-validation for R2 and RMSE ---
        try:
            rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
            scoring = {'r2': 'r2', 'rmse': 'neg_root_mean_squared_error'}
            cv_results = cross_validate(
                model, X_train, y_train,
                cv=rkf, scoring=scoring, return_train_score=True
            )
            # Extract and invert negative RMSE scores
            r2_train = cv_results.get('train_r2', [])
            r2_test = cv_results.get('test_r2', [])
            rmse_train = -cv_results.get('train_rmse', [])
            rmse_test = -cv_results.get('test_rmse', [])
            results['cv'] = {
                'r2_cv_train_mean': float(np.mean(r2_train)),
                'r2_cv_train_std': float(np.std(r2_train)),
                'r2_cv_test_mean': float(np.mean(r2_test)),
                'r2_cv_test_std': float(np.std(r2_test)),
                'rmse_cv_train_mean': float(np.mean(rmse_train)),
                'rmse_cv_train_std': float(np.std(rmse_train)),
                'rmse_cv_test_mean': float(np.mean(rmse_test)),
                'rmse_cv_test_std': float(np.std(rmse_test)),
            }
        except Exception:
            # If CV fails, continue without CV metrics
            results['cv'] = {}
        # --- Treinamento ---
        model.fit(X_train, y_train)

        # --- Predições ---
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # --- Resíduos ---
        residuals_train = y_train - y_pred_train
        residuals_test = y_test - y_pred_test

        # --- Métricas treino/teste ---
        for split, y_true, y_pred, residuals in [
            ("train", y_train, y_pred_train, residuals_train),
            ("test", y_test, y_pred_test, residuals_test)
        ]:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            medae = median_absolute_error(y_true, y_pred)

            try:
                mape = mean_absolute_percentage_error(y_true, y_pred)
            except Exception:
                mape = np.nan

            r2 = r2_score(y_true, y_pred)

            # Correlações
            try:
                pearson, _ = pearsonr(y_true, y_pred)
            except Exception:
                pearson = np.nan
            try:
                spearman, _ = spearmanr(y_true, y_pred)
            except Exception:
                spearman = np.nan

            # Estatísticas de resíduos
            res_mean = np.mean(residuals) # mean
            res_var = np.var(residuals) # variance
            res_skew = skew(residuals) # skewness
            res_kurt = kurtosis(residuals) # kurtosis

            results[split] = {
                "r2": r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "medae": medae,
                "mape": mape,
                "pearson": pearson,
                "spearman": spearman,
                "res_mean": res_mean,
                "res_var": res_var,
                "res_skew": res_skew,
                "res_kurt": res_kurt,
                "y_true": y_true,
                "y_pred": y_pred,
                "residuals": residuals
            }

    except Exception as e:
        results["error"] = str(e)

    return results
