"""
predict.py — Standalone inference script for ED-01.

Loads the saved pipeline artifact, accepts ONLY permitted feature columns
+ an opaque 'id' column, and outputs:
  - ranking.csv: all rows with predicted probabilities, sorted descending
  - selected_students.csv: top-20% selected rows

This script has NO dependency on G3, support_needed, or any evaluator-only
metadata. It does NOT load, query, reconstruct, fingerprint, or otherwise
recover evaluation targets.
"""

import os
import sys
import math
import argparse
import pandas as pd
import numpy as np
import joblib

# Add parent to path for local imports
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import FEATURE_COLS

# Forbidden columns — must never appear in inference input
FORBIDDEN_COLS = {"G3", "support_needed"}

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pipeline.joblib")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_pipeline(model_path: str):
    """Load the saved pipeline artifact."""
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        print("  Run train.py first to generate the pipeline.")
        sys.exit(1)
    return joblib.load(model_path)


def validate_input(df: pd.DataFrame, id_col: str = "id"):
    """Validate that input contains only permitted columns."""
    # Check for forbidden columns
    for col in FORBIDDEN_COLS:
        if col in df.columns:
            raise ValueError(
                f"Forbidden column '{col}' found in input. "
                f"Inference must not use target-related columns."
            )

    # Check that required features exist
    missing = set(FEATURE_COLS) - set(df.columns) - {id_col}
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    return True


def predict(input_path: str, model_path: str, output_dir: str, id_col: str = "id",
            sep: str = ";"):
    """Run inference and generate outputs.

    Parameters
    ----------
    input_path : path to CSV with permitted features + id column
    model_path : path to saved pipeline.joblib
    output_dir : directory for output CSVs
    id_col : name of the opaque ID column
    sep : CSV separator (default semicolon for student-mat format)
    """
    print("\n" + "=" * 60)
    print("  ED-01: INFERENCE")
    print("=" * 60)

    # Load data
    print(f"\n  Loading input from: {input_path}")
    df = pd.read_csv(input_path, sep=sep)
    print(f"  Rows: {len(df)}")

    # Extract or create ID column
    if id_col not in df.columns:
        print(f"  No '{id_col}' column found — generating sequential IDs.")
        df[id_col] = range(len(df))

    # Validate
    validate_input(df, id_col)
    print("  [+] Input validation passed (no forbidden columns)")

    # Extract features
    X = df[FEATURE_COLS].copy()
    ids = df[id_col].values

    # Load model
    print(f"  Loading model from: {model_path}")
    pipeline = load_pipeline(model_path)

    # Predict probabilities
    print("  Generating predictions...")
    probs = pipeline.predict_proba(X)[:, 1]

    n = len(X)
    k = math.ceil(0.20 * n)

    # Build ranking
    ranking_df = pd.DataFrame({
        "id": ids,
        "predicted_probability": probs,
    })
    ranking_df = ranking_df.sort_values("predicted_probability", ascending=False)
    ranking_df["rank"] = range(1, len(ranking_df) + 1)

    # Save ranking
    os.makedirs(output_dir, exist_ok=True)
    ranking_path = os.path.join(output_dir, "ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)
    print(f"  [+] Saved ranking.csv ({len(ranking_df)} rows)")

    # Selected students (top-20%)
    selected_df = ranking_df.head(k).copy()
    selected_df["selected"] = 1

    selected_path = os.path.join(output_dir, "selected_students.csv")
    selected_df.to_csv(selected_path, index=False)
    print(f"  [+] Saved selected_students.csv ({len(selected_df)} rows, top-20%)")

    print(f"\n  Probability summary:")
    print(f"    Mean:   {probs.mean():.4f}")
    print(f"    Std:    {probs.std():.4f}")
    print(f"    Min:    {probs.min():.4f}")
    print(f"    Max:    {probs.max():.4f}")
    print(f"    Budget: k = {k} out of {n}")

    print("\n  [+] Inference complete!")
    return ranking_df, selected_df


def main():
    parser = argparse.ArgumentParser(
        description="ED-01: Predict student support-needed probability"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV (permitted features + id column)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Path to saved pipeline.joblib"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Output directory for ranking.csv and selected_students.csv"
    )
    parser.add_argument(
        "--id-col", default="id",
        help="Name of the opaque ID column (default: 'id')"
    )
    parser.add_argument(
        "--sep", default=";",
        help="CSV separator (default: ';')"
    )

    args = parser.parse_args()
    predict(args.input, args.model, args.output_dir, args.id_col, args.sep)


if __name__ == "__main__":
    main()
