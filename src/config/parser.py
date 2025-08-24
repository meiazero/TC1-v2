try:
    import yaml
except ImportError:
    yaml = None
import itertools

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


MODEL_REGISTRY = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "LogisticRegression": LogisticRegression,
    "SGDRegressor": SGDRegressor,
    "MLPRegressor": MLPRegressor,
    "SVR": SVR,
    "LinearSVR": LinearSVR,
    "KernelRidge": KernelRidge,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "ExtraTreeRegressor": ExtraTreeRegressor,
    "AdaBoostRegressor": AdaBoostRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}

if HAS_XGB:
    MODEL_REGISTRY["XGBRegressor"] = XGBRegressor


def load_model_configs(path: str):
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load model configurations. "
            "Install it via pip install pyyaml"
        )
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    experiments = []
    for model_cfg in config["models"]:
        name = model_cfg["name"]
        mode = model_cfg["mode"]

        if name not in MODEL_REGISTRY:
            raise ValueError(f"Modelo {name} não suportado.")

        if mode == "fixed":
            params = model_cfg.get("params", {})
            model = MODEL_REGISTRY[name](**params)
            experiments.append({"name": name, "params": params, "model": model})

        elif mode == "grid":
            params = model_cfg.get("params", {}) or {}
            # ensure each param value is iterable for grid combinations
            items = []
            for key, val in params.items():
                if isinstance(val, (list, tuple)):
                    items.append((key, val))
                else:
                    items.append((key, [val]))
            if items:
                keys, values = zip(*items)
            else:
                keys, values = ([], [])
            for combination in itertools.product(*values):
                param_dict = dict(zip(keys, combination))
                model = MODEL_REGISTRY[name](**param_dict)
                experiments.append({"name": name, "params": param_dict, "model": model})

        elif mode == "sequential":
            for conf in model_cfg.get("configs", []):
                model = MODEL_REGISTRY[name](**conf)
                experiments.append({"name": name, "params": conf, "model": model})

        else:
            raise ValueError(f"Modo desconhecido: {mode}")

    return experiments
