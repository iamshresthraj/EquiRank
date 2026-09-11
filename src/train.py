"""
train.py — Model training, comparison, calibration, and pipeline saving for ED-01.

Workflow:
  1. Load data, construct target, split stratified
  2. Multi-seed evaluation loop (10 seeds) for stability
  3. Train LogisticRegression (balanced) and GradientBoosting
  4. Compare using composite rubric score
  5. Calibrate probabilities (CalibratedClassifierCV)
  6. Integrate fairness mitigation
  7. Save final pipeline to models/pipeline.joblib
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Add parent to path for local imports
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import (
    load_and_prepare, build_preprocessor,
    FEATURE_COLS, CATEGORICAL_COLS, NUMERIC_COLS
)
from evaluate import evaluate, build_fairness_table
from fairness import apply_fairness_postprocessing, diagnose_and_mitigate

RANDOM_SEED = 42
N_STABILITY_SEEDS = 10
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pipeline.joblib")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def build_candidate_pipelines():
    """Return dict of candidate model pipelines."""
    return {
        "logistic_balanced": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_SEED,
            )),
        ]),
    }


def evaluate_model(pipeline, X_val, y_val, protected_val, verbose=False):
    """Evaluate a fitted pipeline on validation data."""
    probs = pipeline.predict_proba(X_val)[:, 1]
    results = evaluate(
        y_true=y_val.values,
        probabilities=probs,
        sex_labels=protected_val["sex"].values,
        school_labels=protected_val["school"].values,
        verbose=verbose,
    )
    return results, probs


def multi_seed_stability(X, y, protected, n_seeds=N_STABILITY_SEEDS):
    """Run model training across multiple seeds to check metric stability.

    This is especially important for the small MS-school subgroup.
    """
    print("\n" + "=" * 60)
    print("  MULTI-SEED STABILITY ANALYSIS")
    print("=" * 60)

    candidates = list(build_candidate_pipelines().keys())
    seed_results = {name: [] for name in candidates}

    for seed in range(n_seeds):
        X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
            X, y, protected,
            test_size=0.25,
            stratify=y,
            random_state=seed,
        )

        for name in candidates:
            # Build fresh pipeline for each seed
            pipelines = build_candidate_pipelines()
            pipe = pipelines[name]
            pipe.fit(X_train, y_train)
            res, _ = evaluate_model(pipe, X_val, y_val, prot_val, verbose=False)
            seed_results[name].append(res)

    # Summarize
    for name in candidates:
        scores = [r["composite_score"] for r in seed_results[name]]
        recalls = [r["overall_recall"] for r in seed_results[name]]
        gaps = [r["fairness_gap"] for r in seed_results[name]]
        briers = [r["brier"] for r in seed_results[name]]

        print(f"\n  Model: {name}")
        print(f"    Composite : {np.mean(scores):.2f} ± {np.std(scores):.2f}  "
              f"(range {np.min(scores):.2f}–{np.max(scores):.2f})")
        print(f"    Recall    : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"    Fair. Gap : {np.mean(gaps):.4f} ± {np.std(gaps):.4f}")
        print(f"    Brier     : {np.mean(briers):.4f} ± {np.std(briers):.4f}")

    return seed_results


def train_and_select_best(X, y, protected):
    """Train candidates, compare on composite score, return best.

    Uses a fixed seed for the primary train/val split.
    """
    X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
        X, y, protected,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print(f"\n  Train: {len(X_train)} rows, Val: {len(X_val)} rows")
    print(f"  Train positives: {y_train.sum()} ({y_train.mean():.1%})")
    print(f"  Val positives:   {y_val.sum()} ({y_val.mean():.1%})")

    pipelines = build_candidate_pipelines()
    best_name = None
    best_score = -1
    best_pipeline = None
    all_results = {}

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    for name, pipe in pipelines.items():
        print(f"\n  Training: {name}")
        pipe.fit(X_train, y_train)
        results, probs = evaluate_model(pipe, X_val, y_val, prot_val, verbose=True)
        all_results[name] = results

        if results["composite_score"] > best_score:
            best_score = results["composite_score"]
            best_name = name
            best_pipeline = pipe

    print(f"\n  ✓ Best model: {best_name}  (composite = {best_score:.2f})")

    return best_pipeline, best_name, X_train, X_val, y_train, y_val, prot_train, prot_val


def calibrate_model(best_pipeline, best_name, X_train, X_val, y_train, y_val, prot_val):
    """Calibrate the best model using CalibratedClassifierCV.

    Re-runs evaluation to confirm Brier improves.
    """
    print("\n" + "=" * 60)
    print("  PROBABILITY CALIBRATION")
    print("=" * 60)

    # Get pre-calibration metrics
    pre_results, _ = evaluate_model(best_pipeline, X_val, y_val, prot_val, verbose=False)
    print(f"\n  Pre-calibration Brier: {pre_results['brier']:.4f}")

    # Build calibrated pipeline
    # We calibrate the entire pipeline (preprocessor + classifier)
    calibrated = CalibratedClassifierCV(
        estimator=best_pipeline,
        method="sigmoid",
        cv=5,
    )
    calibrated.fit(X_train, y_train)

    # Re-evaluate
    post_results, probs = evaluate_model(calibrated, X_val, y_val, prot_val, verbose=False)
    print(f"  Post-calibration Brier: {post_results['brier']:.4f}")
    print(f"  Brier change: {post_results['brier'] - pre_results['brier']:+.4f}")

    # Check if calibration helped
    if post_results['brier'] < pre_results['brier']:
        print("  ✓ Calibration improved Brier score — using calibrated model.")
        # Verify recall and fairness not materially hurt
        recall_diff = post_results['overall_recall'] - pre_results['overall_recall']
        gap_diff = post_results['fairness_gap'] - pre_results['fairness_gap']
        print(f"  Recall change:  {recall_diff:+.4f}")
        print(f"  Gap change:     {gap_diff:+.4f}")
        return calibrated, post_results
    else:
        print("  ✗ Calibration did not improve Brier — keeping uncalibrated model.")
        return best_pipeline, pre_results


def main():
    """Full training pipeline."""
    print("\n" + "#" * 60)
    print("  ED-01: FAIR STUDENT-SUPPORT PRIORITIZATION")
    print("  Training Pipeline")
    print("#" * 60)

    # 1. Load data
    print("\n  Loading data...")
    X, y, protected = load_and_prepare(DATA_PATH)
    print(f"  Dataset: {len(X)} rows, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.1%} (support_needed=1)")
    print(f"  Schools: {protected['school'].value_counts().to_dict()}")
    print(f"  Sex:     {protected['sex'].value_counts().to_dict()}")

    # 2. Multi-seed stability analysis
    seed_results = multi_seed_stability(X, y, protected)

    # 3–4. Train candidates and select best
    best_pipeline, best_name, X_train, X_val, y_train, y_val, prot_train, prot_val = \
        train_and_select_best(X, y, protected)

    # 5. Calibrate
    calibrated_pipeline, cal_results = calibrate_model(
        best_pipeline, best_name, X_train, X_val, y_train, y_val, prot_val
    )

    # 6. Fairness diagnosis and mitigation
    final_pipeline, fairness_info = diagnose_and_mitigate(
        calibrated_pipeline, X, y, protected,
        X_train, X_val, y_train, y_val, prot_train, prot_val,
        outputs_dir=OUTPUTS_DIR,
    )

    # 7. Final evaluation on validation set
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION (validation set)")
    print("=" * 60)
    final_results, final_probs = evaluate_model(
        final_pipeline, X_val, y_val, prot_val, verbose=True
    )

    # 8. Save fairness table
    fairness_df = build_fairness_table(final_results)
    fairness_path = os.path.join(OUTPUTS_DIR, "fairness_table.csv")
    fairness_df.to_csv(fairness_path, index=False)
    print(f"\n  ✓ Saved fairness_table.csv to {fairness_path}")

    # 9. Save final pipeline
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"  ✓ Saved pipeline to {MODEL_PATH}")

    # 10. Generate ranking and selection on full validation set
    generate_outputs(final_pipeline, X_val, y_val, prot_val)

    print("\n  ✓ Training pipeline complete!")
    return final_pipeline, final_results


def generate_outputs(pipeline, X_val, y_val, prot_val):
    """Generate ranking.csv and selected_students.csv on validation data."""
    import math

    probs = pipeline.predict_proba(X_val)[:, 1]
    n = len(X_val)
    k = math.ceil(0.20 * n)

    # Build ranking DataFrame
    ranking_df = pd.DataFrame({
        "id": range(len(X_val)),
        "predicted_probability": probs,
    })
    ranking_df = ranking_df.sort_values("predicted_probability", ascending=False)
    ranking_df["rank"] = range(1, len(ranking_df) + 1)

    ranking_path = os.path.join(OUTPUTS_DIR, "ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)
    print(f"  ✓ Saved ranking.csv ({len(ranking_df)} rows)")

    # Selected students (top-20%)
    selected_df = ranking_df.head(k).copy()
    selected_df["selected"] = 1

    selected_path = os.path.join(OUTPUTS_DIR, "selected_students.csv")
    selected_df.to_csv(selected_path, index=False)
    print(f"  ✓ Saved selected_students.csv ({len(selected_df)} rows)")


if __name__ == "__main__":
    main()
