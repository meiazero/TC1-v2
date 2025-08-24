import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    """
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data: rename columns, drop duplicates and missing values.
    """
    df = df.copy()
    # Rename columns for clarity
    df = df.rename(columns={
        "X1 transaction date": "transaction_date",
        "X2 house age": "house_age",
        "X3 distance to the nearest MRT station": "distance_to_mrt",
        "X4 number of convenience stores": "num_convenience",
        "X5 latitude": "latitude",
        "X6 longitude": "longitude",
        "Y house price of unit area": "price"
    })
    # Drop unnecessary columns
    if "No" in df.columns:
        df = df.drop(columns=["No"])
    df = df.drop_duplicates()
    df = df.dropna()
    return df