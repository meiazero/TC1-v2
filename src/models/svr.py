from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

def get_svr_model(**params):
    """Instantiate an SVR model."""
    return SVR(**params)

def get_kernel_ridge_model(**params):
    """Instantiate a Kernel Ridge model."""
    return KernelRidge(**params)