from sklearn.neural_network import MLPRegressor

def get_mlp_model(**params):
    """Instantiate an MLPRegressor model."""
    return MLPRegressor(**params)