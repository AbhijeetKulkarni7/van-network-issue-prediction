# ============================================================
# preprocessing.py
# VAN Network Issue Prediction
# ============================================================

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

TARGET_COLUMN = "Network_Issue"

# These are identifiers and are not used directly by the model.
DROP_COLUMNS = [
    "Vehicle_ID",
    "Timestamp"
]

CATEGORICAL_FEATURES = [
    "Network_Type",
    "Topology"
]


# ============================================================
# GET FEATURE COLUMNS
# ============================================================

def get_feature_columns(df):
    """
    Returns the feature columns used by the ML model.
    """

    columns_to_drop = DROP_COLUMNS + [TARGET_COLUMN]

    feature_columns = [
        column
        for column in df.columns
        if column not in columns_to_drop
    ]

    return feature_columns


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(numeric_features):
    """
    Creates the preprocessing pipeline.

    Numerical:
        Missing values -> Median
        Scaling -> StandardScaler

    Categorical:
        Missing values -> Most frequent
        Encoding -> OneHotEncoder
    """

    # Numerical preprocessing
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """
    Separates input features (X) and target (y).
    """

    feature_columns = get_feature_columns(df)

    X = df[feature_columns].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# GET NUMERICAL FEATURES
# ============================================================

def get_numeric_features(X):
    """
    Identifies numerical features.
    """

    numeric_features = [
        column
        for column in X.columns
        if column not in CATEGORICAL_FEATURES
    ]

    return numeric_features


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    DATA_PATH = "data/van_network_issue_dataset.csv"

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    X, y = prepare_data(df)

    numeric_features = get_numeric_features(X)

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nNumerical features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(CATEGORICAL_FEATURES)

    print("\nTarget:")
    print(TARGET_COLUMN)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nPreprocessing module working successfully!")