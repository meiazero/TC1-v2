from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def get_decision_tree_model(**params):
    """Instantiate a Decision Tree Regressor."""
    return DecisionTreeRegressor(**params)

def get_random_forest_model(**params):
    """Instantiate a Random Forest Regressor."""
    return RandomForestRegressor(**params)

def get_gradient_boosting_model(**params):
    """Instantiate a Gradient Boosting Regressor."""
    return GradientBoostingRegressor(**params)

if HAS_XGB:
    def get_xgb_model(**params):
        """Instantiate an XGBoost Regressor."""
        return XGBRegressor(**params)