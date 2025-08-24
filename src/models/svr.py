from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVC


def get_svr_model(**params):
    """Instantiate an SVR model."""
    return SVR(**params)

def get_kernel_ridge_model(**params):
    """Instantiate a Kernel Ridge model."""
    return KernelRidge(**params)

def get_svc_model(**params):
    """Instantiate an SVC model."""
    return LinearSVC(**params)
