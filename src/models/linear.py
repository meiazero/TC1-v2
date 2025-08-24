from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor,
)

LINEAR_MODELS = {
    "LinearRegression": LinearRegression,
    "LogisticRegression": LogisticRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "SGDRegressor": SGDRegressor,
}

def get_linear_model(name: str, **params):
    """Instantiate a linear model by name."""
    if name not in LINEAR_MODELS:
        raise ValueError(f"Unknown linear model: {name}")
    return LINEAR_MODELS[name](**params)