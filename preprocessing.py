# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, on_bad_lines='skip')  # problemli satırları atla

    # Gereksiz sütunları kaldır
    df.drop(columns=["vin", "saledate", "seller"], inplace=True, errors="ignore")

    # Hedef sütun eksikse o satırı at
    df.dropna(subset=["sellingprice"], inplace=True)

    # Genel eksik verileri kaldır
    df.dropna(inplace=True)

    # Hedef ve girdileri ayır
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
