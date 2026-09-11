"""
evaluate.py — Implements all 9 rubric formulas for ED-01 exactly.

Formulas implemented:
  1. Support-budget selection (top-20%)
  2. Protected groups (sex: F/M, school: GP/MS)
  3. Group eligibility (>=10 rows AND >=3 positives)
  4. Per-group recall R_g
  5. Worst-group recall R_min
  6. Fairness gap (max pairwise diff per attribute, then max across)
  7. Overall top-20% recall
  8. Brier score
  9. Composite rubric score (100 points)
"""

import math
import warnings
from itertools import combinations
import numpy as np
import pandas as pd


def select_top_k(probabilities: np.ndarray, n: int) -> np.ndarray:
    """Return boolean mask for top-k rows where k = ceil(0.20 * n).

    Ties are broken by index order (argsort is stable).
    """
    k = math.ceil(0.20 * n)
    # Descending sort by probability — use negative for ascending argsort
    ranked_indices = np.argsort(-probabilities, kind="mergesort")
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[ranked_indices[:k]] = True
    return selected_mask


def group_eligibility(y_true: np.ndarray, group_labels: np.ndarray):
    """Return dict mapping group_label -> bool (eligible if rows>=10 AND positives>=3)."""
    eligibility = {}
    for g in np.unique(group_labels):
        mask = group_labels == g
        n_rows = mask.sum()
        n_pos = y_true[mask].sum()
        eligibility[g] = (n_rows >= 10) and (n_pos >= 3)
    return eligibility


def per_group_recall(y_true: np.ndarray, selected_mask: np.ndarray,
                     group_labels: np.ndarray, eligibility: dict) -> dict:
    """Compute recall R_g for each eligible group.

    R_g = (positive rows in group g that are selected) / (total positive rows in group g)
    """
    recalls = {}
    for g, is_eligible in eligibility.items():
        if not is_eligible:
            continue
        mask = group_labels == g
        positives_in_group = y_true[mask].sum()
        if positives_in_group == 0:
            # Should not happen for eligible groups, but guard anyway
            recalls[g] = 0.0
        else:
            selected_positives = (y_true[mask] & selected_mask[mask]).sum()
            recalls[g] = float(selected_positives) / float(positives_in_group)
    return recalls


def worst_group_recall(group_recalls: dict):
    """R_min = min(R_g) over eligible groups. Returns None if no eligible groups."""
    if not group_recalls:
        return None
    return min(group_recalls.values())


def attribute_fairness_gap(group_recalls: dict) -> float | None:
    """Max absolute pairwise difference in R_g for one attribute.

    Returns None if fewer than 2 eligible groups for this attribute.
    """
    if len(group_recalls) < 2:
        return None
    max_diff = 0.0
    for (g1, r1), (g2, r2) in combinations(group_recalls.items(), 2):
        max_diff = max(max_diff, abs(r1 - r2))
    return max_diff


def fairness_gap(sex_recalls: dict, school_recalls: dict) -> float:
    """Compute overall fairness gap across both protected attributes.

    fairness_gap = max(attribute_gap) over attributes that contribute.
    If neither attribute has >=2 eligible groups, fairness_gap = 0.
    """
    sex_gap = attribute_fairness_gap(sex_recalls)
    school_gap = attribute_fairness_gap(school_recalls)

    contributing = []
    if sex_gap is not None:
        contributing.append(sex_gap)
    if school_gap is not None:
        contributing.append(school_gap)

    if not contributing:
        return 0.0
    return max(contributing)


def overall_recall(y_true: np.ndarray, selected_mask: np.ndarray) -> float:
    """Overall top-20% recall = positives in selected / total positives."""
    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0
    return float((y_true & selected_mask).sum()) / float(total_positives)


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Brier = (1/N) * sum((p_i - y_i)^2)."""
    n = len(y_true)
    return float(np.mean((probabilities - y_true.astype(float)) ** 2))


def composite_score(overall_rec: float, r_min: float, fair_gap: float,
                    brier: float, reproducibility: float = 1.0) -> float:
    """100-point composite rubric score.

    score = 40*overall_recall + 25*R_min + 20*(1-fairness_gap) + 10*(1-Brier) + 5*reproducibility
    """
    return (40.0 * overall_rec
            + 25.0 * r_min
            + 20.0 * (1.0 - fair_gap)
            + 10.0 * (1.0 - brier)
            + 5.0 * reproducibility)


def evaluate(y_true: np.ndarray, probabilities: np.ndarray,
             sex_labels: np.ndarray, school_labels: np.ndarray,
             reproducibility: float = 1.0, verbose: bool = True) -> dict:
    """Run the full evaluation pipeline and return all metrics.

    Parameters
    ----------
    y_true : array of 0/1 (support_needed)
    probabilities : predicted probabilities
    sex_labels : array of 'F'/'M'
    school_labels : array of 'GP'/'MS'
    reproducibility : 1.0 if reproducible, else 0.0
    verbose : if True, print formatted breakdown

    Returns
    -------
    dict with all metrics
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    sex_labels = np.asarray(sex_labels)
    school_labels = np.asarray(school_labels)

    n = len(y_true)

    # 1. Select top-k
    selected = select_top_k(probabilities, n)
    k = math.ceil(0.20 * n)

    # 2–4. Group eligibility and per-group recall for sex
    sex_elig = group_eligibility(y_true, sex_labels)
    sex_recalls = per_group_recall(y_true, selected, sex_labels, sex_elig)

    # 2–4. Group eligibility and per-group recall for school
    school_elig = group_eligibility(y_true, school_labels)
    school_recalls = per_group_recall(y_true, selected, school_labels, school_elig)

    # 5. Worst-group recall
    all_recalls = {**sex_recalls, **school_recalls}
    r_min = worst_group_recall(all_recalls)
    if r_min is None:
        warnings.warn("No eligible groups found — R_min is undefined.")
        r_min = 0.0

    # 6. Fairness gap
    f_gap = fairness_gap(sex_recalls, school_recalls)

    # 7. Overall recall
    o_recall = overall_recall(y_true, selected)

    # 8. Brier
    brier = brier_score(y_true, probabilities)

    # 9. Composite score
    score = composite_score(o_recall, r_min, f_gap, brier, reproducibility)

    results = {
        "n": n,
        "k": k,
        "overall_recall": o_recall,
        "r_min": r_min,
        "fairness_gap": f_gap,
        "brier": brier,
        "composite_score": score,
        "reproducibility": reproducibility,
        "sex_eligibility": sex_elig,
        "school_eligibility": school_elig,
        "sex_recalls": sex_recalls,
        "school_recalls": school_recalls,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("  ED-01 EVALUATION — 100-POINT RUBRIC BREAKDOWN")
        print("=" * 60)
        print(f"  N = {n},  k = {k}  (top-20% budget)")
        print()
        print("  Per-group recalls:")
        for attr_name, recalls, elig in [("sex", sex_recalls, sex_elig),
                                          ("school", school_recalls, school_elig)]:
            for g in sorted(elig.keys()):
                status = f"R={recalls[g]:.4f}" if elig[g] else "INELIGIBLE"
                print(f"    {attr_name}={g:>3s}:  {status}")
        print()
        print(f"  Overall Recall  : {o_recall:.4f}   (x40 -> {40 * o_recall:.2f} pts)")
        print(f"  Worst-Group R   : {r_min:.4f}   (x25 -> {25 * r_min:.2f} pts)")
        print(f"  Fairness Gap    : {f_gap:.4f}   (x20 -> {20 * (1 - f_gap):.2f} pts)")
        print(f"  Brier Score     : {brier:.4f}   (x10 -> {10 * (1 - brier):.2f} pts)")
        print(f"  Reproducibility : {reproducibility:.1f}      (x 5 -> {5 * reproducibility:.2f} pts)")
        print("-" * 60)
        print(f"  COMPOSITE SCORE : {score:.2f} / 100")
        print("=" * 60 + "\n")

    return results


def build_fairness_table(results: dict) -> pd.DataFrame:
    """Build a DataFrame suitable for saving as fairness_table.csv."""
    rows = []
    for attr, recalls_key, elig_key in [("sex", "sex_recalls", "sex_eligibility"),
                                         ("school", "school_recalls", "school_eligibility")]:
        elig = results[elig_key]
        recalls = results[recalls_key]
        for g in sorted(elig.keys()):
            rows.append({
                "attribute": attr,
                "group": g,
                "eligible": elig[g],
                "recall": recalls.get(g, None),
            })

    df = pd.DataFrame(rows)
    # Add summary metrics as extra columns on the first row
    df["R_min"] = None
    df["fairness_gap"] = None
    df["brier"] = None
    df["composite_score"] = None
    df.loc[0, "R_min"] = results["r_min"]
    df.loc[0, "fairness_gap"] = results["fairness_gap"]
    df.loc[0, "brier"] = results["brier"]
    df.loc[0, "composite_score"] = results["composite_score"]
    return df


if __name__ == "__main__":
    import os
    import joblib
    from sklearn.model_selection import train_test_split

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "pipeline.joblib")
    data_path = os.path.join(base_dir, "data", "student-mat.csv")

    if os.path.exists(model_path) and os.path.exists(data_path):
        from preprocess import load_and_prepare
        print(f"Evaluating final trained pipeline from: {model_path}")
        X, y, protected = load_and_prepare(data_path)
        _, X_val, _, y_val, _, prot_val = train_test_split(
            X, y, protected,
            test_size=0.25,
            stratify=y,
            random_state=42,
        )
        pipeline = joblib.load(model_path)
        probs = pipeline.predict_proba(X_val)[:, 1]
        results = evaluate(
            y_true=y_val.values,
            probabilities=probs,
            sex_labels=prot_val["sex"].values,
            school_labels=prot_val["school"].values,
            reproducibility=1.0,
            verbose=True,
        )
    else:
        # Quick smoke test with dummy data
        np.random.seed(42)
        n = 100
        y = np.random.binomial(1, 0.3, n)
        p = np.clip(y + np.random.normal(0, 0.3, n), 0, 1)
        sex = np.random.choice(["F", "M"], n)
        school = np.random.choice(["GP", "MS"], n, p=[0.8, 0.2])
        res = evaluate(y, p, sex, school)

