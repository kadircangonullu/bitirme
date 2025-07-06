import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, on_bad_lines='skip')  # problematic rows will be skipped

    # Drop unnecessary columns
    df.drop(columns=["vin", "saledate", "seller"], inplace=True, errors="ignore")

    # If target column is missing, drop that row
    df.dropna(subset=["sellingprice"], inplace=True)

    # Drop all other missing values
    df.dropna(inplace=True)

    # Separate target and features
    y = df["sellingprice"]
    X = df.drop(columns=["sellingprice", "mmr"], errors="ignore")

    return X, y


def preprocess_data(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    return preprocessor, cat_cols, num_cols


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
