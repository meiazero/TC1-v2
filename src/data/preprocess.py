import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

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

def remove_outliers(df: pd.DataFrame, method: str = "iqr", factor: float = 1.5, columns=None):
    """
    Remove outliers from DataFrame using the IQR method.

    Args:
        df: DataFrame to process.
        method: Outlier removal method. Currently only 'iqr' is supported.
        factor: The multiplier for the IQR to define the bounds.
        columns: List of columns to consider for outlier removal. Defaults to all numeric columns.

    Returns:
        A DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    mask = pd.Series(True, index=df.index)

    if method == "iqr":
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            mask &= df[col].between(lower, upper)
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")
    return df[mask].copy()