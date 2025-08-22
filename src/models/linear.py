from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor

LINEAR_MODELS = {
    "LinearRegression": LinearRegression,
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