from sklearn.neural_network import MLPClassifier, MLPRegressor


def get_mlp_model(**params):
    """Instantiate an MLPRegressor model."""
    return MLPRegressor(**params)

def get_mlp_classifier_model(**params):
    """Instantiate an MLPClassifier model."""
    return MLPClassifier(**params)