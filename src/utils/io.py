import os
import pickle

def make_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def save_dataframe(df, path):
    """Save a pandas DataFrame to CSV at the given path."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    df.to_csv(path, index=False)

def save_model(model, path):
    """Serialize and save a model or object to the given path using pickle."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)