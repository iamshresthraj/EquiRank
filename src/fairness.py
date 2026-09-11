"""
fairness.py — Fairness diagnosis, mitigation, and Pareto tradeoff analysis for ED-01.

Techniques implemented:
  1. Group-aware sample reweighting during training
  2. Post-hoc per-group probability recalibration before global top-k ranking

The module sweeps multiple fairness-strength hyperparameters, records
recall vs. fairness gap tradeoffs, and generates a Pareto chart.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no UI
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import (
    load_and_prepare, build_preprocessor,
    FEATURE_COLS, CATEGORICAL_COLS, NUMERIC_COLS
)
from evaluate import evaluate, select_top_k

RANDOM_SEED = 42


class FairnessPostProcessor:
    """Post-hoc per-group probability adjustment wrapper.

    Wraps a fitted pipeline and applies per-group probability
    recalibration (additive shift) to equalize recall across groups
    before the global top-k selection.

    Parameters
    ----------
    base_pipeline : fitted sklearn pipeline with predict_proba
    group_adjustments : dict mapping (attribute, group) -> additive shift
        e.g. {("school", "MS"): 0.05} means add 0.05 to MS students' probs
    """

    def __init__(self, base_pipeline, group_adjustments=None):
        self.base_pipeline = base_pipeline
        self.group_adjustments = group_adjustments or {}

    def predict_proba(self, X):
        """Return adjusted probabilities."""
        probs = self.base_pipeline.predict_proba(X)
        # Apply adjustments if we have group info in X
        adjusted = probs.copy()
        for (attr, group), shift in self.group_adjustments.items():
            if attr in X.columns:
                mask = X[attr].values == group
                adjusted[mask, 1] = np.clip(adjusted[mask, 1] + shift, 0, 1)
                adjusted[mask, 0] = 1.0 - adjusted[mask, 1]
        return adjusted

    def fit(self, X, y):
        """No-op — base pipeline is already fitted."""
        return self

    def get_params(self, deep=True):
        return {
            "base_pipeline": self.base_pipeline,
            "group_adjustments": self.group_adjustments,
        }


def compute_group_sample_weights(y_train, protected_train, strength=1.0):
    """Compute sample weights for group-aware reweighting.

    Higher strength → more aggressive reweighting toward underrepresented
    group-label combinations.

    Weight for sample i in group g with label l:
      w_i = (N / (n_groups * n_gl)) ^ strength
    where n_gl = count of samples with same group and label.
    """
    n = len(y_train)
    weights = np.ones(n)

    for attr in ["sex", "school"]:
        groups = protected_train[attr].values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        for g in unique_groups:
            for label in [0, 1]:
                mask = (groups == g) & (y_train.values == label)
                n_gl = mask.sum()
                if n_gl > 0:
                    w = (n / (n_groups * n_gl)) ** strength
                    weights[mask] *= w

    # Normalize so weights sum to n
    weights = weights * (n / weights.sum())
    return weights


def diagnose_fairness(pipeline, X_val, y_val, prot_val, label="Baseline"):
    """Diagnose fairness gap and print summary."""
    probs = pipeline.predict_proba(X_val)[:, 1]
    results = evaluate(
        y_true=y_val.values,
        probabilities=probs,
        sex_labels=prot_val["sex"].values,
        school_labels=prot_val["school"].values,
        verbose=False,
    )

    print(f"\n  [{label}]")
    print(f"    Overall Recall: {results['overall_recall']:.4f}")
    print(f"    R_min:          {results['r_min']:.4f}")
    print(f"    Fairness Gap:   {results['fairness_gap']:.4f}")
    print(f"    Brier:          {results['brier']:.4f}")
    print(f"    Composite:      {results['composite_score']:.2f}")
    print(f"    Sex recalls:    {results['sex_recalls']}")
    print(f"    School recalls: {results['school_recalls']}")

    return results


def sweep_reweighting(X, y, protected, strengths=None):
    """Sweep group-reweighting strengths and return results."""
    if strengths is None:
        strengths = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    sweep_results = []

    for strength in strengths:
        X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
            X, y, protected,
            test_size=0.25,
            stratify=y,
            random_state=RANDOM_SEED,
        )

        # Build fresh pipeline
        from preprocess import build_preprocessor
        from sklearn.ensemble import GradientBoostingClassifier

        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_SEED,
            )),
        ])

        # Compute sample weights
        weights = compute_group_sample_weights(y_train, prot_train, strength=strength)

        # Fit with weights
        pipe.fit(X_train, y_train, classifier__sample_weight=weights)

        # Evaluate
        probs = pipe.predict_proba(X_val)[:, 1]
        results = evaluate(
            y_true=y_val.values,
            probabilities=probs,
            sex_labels=prot_val["sex"].values,
            school_labels=prot_val["school"].values,
            verbose=False,
        )

        results["method"] = f"reweight_{strength}"
        results["strength"] = strength
        sweep_results.append(results)

    return sweep_results


def sweep_posthoc(pipeline, X_val, y_val, prot_val, shifts=None):
    """Sweep post-hoc probability adjustments for underrepresented groups."""
    if shifts is None:
        shifts = np.arange(-0.10, 0.15, 0.02)

    sweep_results = []

    # Find the underperforming group from baseline
    baseline_probs = pipeline.predict_proba(X_val)[:, 1]
    baseline = evaluate(
        y_true=y_val.values,
        probabilities=baseline_probs,
        sex_labels=prot_val["sex"].values,
        school_labels=prot_val["school"].values,
        verbose=False,
    )

    # Determine which groups need boosting
    all_recalls = {**baseline["sex_recalls"], **baseline["school_recalls"]}
    if all_recalls:
        min_group = min(all_recalls, key=all_recalls.get)
        # Determine attribute
        if min_group in baseline["sex_recalls"]:
            attr = "sex"
        else:
            attr = "school"
    else:
        return sweep_results

    for shift in shifts:
        wrapper = FairnessPostProcessor(
            base_pipeline=pipeline,
            group_adjustments={(attr, min_group): shift}
        )

        probs = wrapper.predict_proba(X_val)[:, 1]
        results = evaluate(
            y_true=y_val.values,
            probabilities=probs,
            sex_labels=prot_val["sex"].values,
            school_labels=prot_val["school"].values,
            verbose=False,
        )

        results["method"] = f"posthoc_{attr}_{min_group}_{shift:.2f}"
        results["shift"] = shift
        results["target_group"] = f"{attr}={min_group}"
        sweep_results.append(results)

    return sweep_results


def plot_pareto(all_results, output_path):
    """Generate Pareto tradeoff chart: overall recall vs. fairness gap."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Separate by method type
    reweight_results = [r for r in all_results if r["method"].startswith("reweight")]
    posthoc_results = [r for r in all_results if r["method"].startswith("posthoc")]

    if reweight_results:
        gaps = [r["fairness_gap"] for r in reweight_results]
        recalls = [r["overall_recall"] for r in reweight_results]
        scores = [r["composite_score"] for r in reweight_results]
        sc = ax.scatter(gaps, recalls, c=scores, cmap="viridis", s=100,
                        label="Group Reweighting", marker="o", edgecolors="black", zorder=3)
        # Annotate with strength
        for r in reweight_results:
            ax.annotate(f's={r["strength"]:.1f}',
                        (r["fairness_gap"], r["overall_recall"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, alpha=0.7)

    if posthoc_results:
        gaps = [r["fairness_gap"] for r in posthoc_results]
        recalls = [r["overall_recall"] for r in posthoc_results]
        scores = [r["composite_score"] for r in posthoc_results]
        sc2 = ax.scatter(gaps, recalls, c=scores, cmap="viridis", s=100,
                         label="Post-hoc Recalibration", marker="s", edgecolors="black", zorder=3)

    ax.set_xlabel("Fairness Gap (lower is better)", fontsize=12)
    ax.set_ylabel("Overall Recall (higher is better)", fontsize=12)
    ax.set_title("ED-01: Recall vs. Fairness Gap — Pareto Tradeoff", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Colorbar for composite score
    all_scores = [r["composite_score"] for r in all_results]
    if all_scores:
        norm = plt.Normalize(min(all_scores), max(all_scores))
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Composite Score", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Pareto tradeoff chart to {output_path}")


def diagnose_and_mitigate(calibrated_pipeline, X, y, protected,
                           X_train, X_val, y_train, y_val, prot_train, prot_val,
                           outputs_dir=None):
    """Full fairness workflow: diagnose, sweep mitigations, select best, plot Pareto.

    Returns
    -------
    final_pipeline : the best pipeline (possibly with fairness post-processing)
    fairness_info : dict with before/after metrics for the report
    """
    print("\n" + "=" * 60)
    print("  FAIRNESS DIAGNOSIS & MITIGATION")
    print("=" * 60)

    # 1. Baseline diagnosis
    baseline = diagnose_fairness(calibrated_pipeline, X_val, y_val, prot_val, "Baseline")

    # 2. Sweep reweighting strengths
    print("\n  Sweeping group-reweighting strengths...")
    reweight_results = sweep_reweighting(X, y, protected)

    # 3. Sweep post-hoc adjustments
    print("  Sweeping post-hoc probability adjustments...")
    posthoc_results = sweep_posthoc(calibrated_pipeline, X_val, y_val, prot_val)

    # 4. Combine all results
    all_results = reweight_results + posthoc_results

    # Add baseline
    baseline["method"] = "reweight_0.0"
    baseline["strength"] = 0.0

    # 5. Plot Pareto
    if outputs_dir:
        os.makedirs(outputs_dir, exist_ok=True)
        pareto_path = os.path.join(outputs_dir, "pareto_tradeoff.png")
        plot_pareto(all_results, pareto_path)

    # 6. Select best by composite score
    best = max(all_results, key=lambda r: r["composite_score"])
    print(f"\n  Best configuration: {best['method']}")
    print(f"    Composite: {best['composite_score']:.2f}")
    print(f"    Recall:    {best['overall_recall']:.4f}")
    print(f"    R_min:     {best['r_min']:.4f}")
    print(f"    Gap:       {best['fairness_gap']:.4f}")
    print(f"    Brier:     {best['brier']:.4f}")

    # 7. Rebuild the best pipeline
    final_pipeline = calibrated_pipeline  # default
    best_method = best["method"]

    if best_method.startswith("reweight_"):
        strength = best.get("strength", 0.0)
        if strength > 0:
            # Retrain with reweighting
            from sklearn.ensemble import GradientBoostingClassifier
            pipe = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=RANDOM_SEED,
                )),
            ])
            weights = compute_group_sample_weights(y_train, prot_train, strength=strength)
            pipe.fit(X_train, y_train, classifier__sample_weight=weights)

            # Calibrate the reweighted model
            cal_pipe = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=5)
            cal_pipe.fit(X_train, y_train)
            final_pipeline = cal_pipe
    elif best_method.startswith("posthoc_"):
        # Apply post-hoc adjustment
        shift = best.get("shift", 0.0)
        target = best.get("target_group", "")
        if "=" in target:
            attr, group = target.split("=")
            final_pipeline = FairnessPostProcessor(
                base_pipeline=calibrated_pipeline,
                group_adjustments={(attr, group): shift}
            )

    # 8. Store before/after for report
    fairness_info = {
        "baseline_recall": baseline["overall_recall"],
        "baseline_r_min": baseline["r_min"],
        "baseline_gap": baseline["fairness_gap"],
        "baseline_brier": baseline["brier"],
        "baseline_composite": baseline["composite_score"],
        "baseline_sex_recalls": baseline["sex_recalls"],
        "baseline_school_recalls": baseline["school_recalls"],
        "best_method": best_method,
        "best_recall": best["overall_recall"],
        "best_r_min": best["r_min"],
        "best_gap": best["fairness_gap"],
        "best_brier": best["brier"],
        "best_composite": best["composite_score"],
        "best_sex_recalls": best.get("sex_recalls", {}),
        "best_school_recalls": best.get("school_recalls", {}),
    }

    return final_pipeline, fairness_info


def apply_fairness_postprocessing(pipeline, X, protected):
    """Apply any fairness post-processing to predictions.

    If pipeline is a FairnessPostProcessor, it handles adjustments internally.
    Otherwise, just return raw predictions.
    """
    probs = pipeline.predict_proba(X)
    return probs[:, 1]


if __name__ == "__main__":
    # Quick test
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
    X, y, protected = load_and_prepare(data_path)

    X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
        X, y, protected, test_size=0.25, stratify=y, random_state=RANDOM_SEED,
    )

    from sklearn.ensemble import GradientBoostingClassifier
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=RANDOM_SEED)),
    ])
    pipe.fit(X_train, y_train)

    diagnose_fairness(pipe, X_val, y_val, prot_val, "Test Baseline")
