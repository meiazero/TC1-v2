from config.parser import MODEL_REGISTRY, load_model_configs


def load_experiments(config_path: str):
    """Load model configurations and instantiate experiments."""
    return load_model_configs(config_path)

def create_model(name: str, params: dict):
    """Create a model by name with given parameters."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](**params)