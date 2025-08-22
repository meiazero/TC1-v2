import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(df: pd.DataFrame, target_col: str = "price", test_size: float = 0.2, random_state: int = 42):
    """
    Split DataFrame into training and testing sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def preprocess(df: pd.DataFrame, target_col: str = "price", test_size: float = 0.2, random_state: int = 42):
    """
    Perform train/test split and feature scaling.
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = split_data(df, target_col, test_size, random_state)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler