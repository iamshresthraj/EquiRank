"""
preprocess.py — Data loading and sklearn preprocessing pipeline for ED-01.

Responsibilities:
  - Load student-mat.csv (semicolon-separated)
  - Construct target: support_needed = 1 if G3 < 10, else 0
  - Drop G3 from features (never used as input)
  - Build a ColumnTransformer (OneHotEncoder for categoricals, StandardScaler for numerics)
  - All transformers are designed to be fit ONLY on training data
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ── Column definitions ──────────────────────────────────────────────────────

# Nominal / binary categorical columns
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

# Numeric columns (including G1, G2 which are permitted early-checkpoint grades)
NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2",
]

# Protected attribute columns (used for fairness evaluation, not dropped)
PROTECTED_ATTRS = ["sex", "school"]

# Target column used only for label construction during development
TARGET_SOURCE_COL = "G3"
TARGET_COL = "support_needed"

# All permitted feature columns (everything except G3 and support_needed)
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def load_data(data_path: str) -> pd.DataFrame:
    """Load student-mat.csv and return raw DataFrame."""
    df = pd.read_csv(data_path, sep=";")
    return df


def construct_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add support_needed column and drop G3.

    support_needed = 1 if G3 < 10, else 0
    G3 is NEVER used as an input feature.
    """
    df = df.copy()
    df[TARGET_COL] = (df[TARGET_SOURCE_COL] < 10).astype(int)
    df = df.drop(columns=[TARGET_SOURCE_COL])
    return df


def load_and_prepare(data_path: str):
    """Full data loading pipeline.

    Returns
    -------
    X : DataFrame of features (FEATURE_COLS only)
    y : Series of support_needed labels
    protected : DataFrame with sex and school columns
    """
    df = load_data(data_path)
    df = construct_target(df)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    protected = df[PROTECTED_ATTRS].copy()

    return X, y, protected


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer for the feature pipeline.

    Must be fit ONLY on training data, never on the full dataset.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract feature names after fitting the preprocessor."""
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn versions
        cat_features = list(
            preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
        )
        num_features = NUMERIC_COLS
        return cat_features + num_features


if __name__ == "__main__":
    # Quick validation
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
    X, y, protected = load_and_prepare(data_path)

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"\nProtected attributes:")
    print(f"  sex: {protected['sex'].value_counts().to_dict()}")
    print(f"  school: {protected['school'].value_counts().to_dict()}")
    print(f"\nFeature columns: {list(X.columns)}")

    # Verify G3 is NOT in the features
    assert "G3" not in X.columns, "G3 must not be in features!"
    assert TARGET_COL not in X.columns, "support_needed must not be in features!"
    print("\n[+] G3 and support_needed correctly excluded from features.")
