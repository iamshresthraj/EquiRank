# EquiRank — Fair Student-Support Prioritization (ED-01)

A machine learning system that ranks students by predicted "support-needed" probability, selects the top 20% for limited school support resources, and minimizes unfairness in recall across protected groups (sex, school).

## Problem Statement

A school can only support 20% of its students. The system must:
1. **Predict** which students need support (`support_needed = 1 if G3 < 10`)
2. **Rank** all students by predicted probability
3. **Select** the top 20% for support
4. **Minimize unfairness** in recall across protected groups (`sex`: F/M, `school`: GP/MS)

## Dataset

- **Source**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance) — Math course (`student-mat.csv`)
- **Size**: 395 students, 30 features + G1, G2, G3 grades
- **Target**: `support_needed = 1 if G3 < 10, else 0` (G3 never used as input feature)

## Quick Start

```bash
# From the ed01/ directory:
bash run.sh
```

This single command reproduces the entire pipeline:
1. Installs dependencies
2. Trains and calibrates the model
3. Runs fairness diagnosis and mitigation
4. Generates all outputs
5. Tests standalone inference
6. Prints the 100-point composite score breakdown

## Project Structure

```
ed01/
├── data/
│   └── student-mat.csv        # UCI Math course data
├── src/
│   ├── preprocess.py          # Data loading + ColumnTransformer
│   ├── train.py               # Training, comparison, calibration
│   ├── evaluate.py            # All 9 rubric formulas
│   ├── predict.py             # Standalone inference script
│   └── fairness.py            # Fairness diagnosis + mitigation
├── models/
│   └── pipeline.joblib        # Saved pipeline artifact
├── outputs/
│   ├── ranking.csv            # Full ranked probability output
│   ├── selected_students.csv  # Top-20% selection
│   ├── fairness_table.csv     # Per-group metrics
│   └── pareto_tradeoff.png    # Recall vs. fairness gap chart
├── report/
│   └── responsible_use.md     # Responsible deployment write-up
├── requirements.txt           # Pinned dependencies
├── run.sh                     # One-command reproduction
└── README.md                  # This file
```

## Scoring Rubric (100 points)

| Component         | Weight | Formula                        |
|-------------------|--------|--------------------------------|
| Overall Recall    | 40     | `40 × overall_recall`          |
| Worst-Group Recall| 25     | `25 × R_min`                   |
| Fairness          | 20     | `20 × (1 - fairness_gap)`      |
| Calibration       | 10     | `10 × (1 - Brier)`             |
| Reproducibility   | 5      | `5 × reproducibility_flag`     |

## Standalone Inference

```bash
python src/predict.py --input path/to/features.csv --output-dir outputs/ --id-col id --sep ";"
```

The inference script accepts ONLY permitted feature columns + an opaque ID column. It has **no dependency** on `G3`, `support_needed`, or any evaluator metadata.

## Key Design Decisions

- **Model**: Gradient Boosting (GBT) selected over Logistic Regression based on composite rubric score
- **Calibration**: Sigmoid calibration via `CalibratedClassifierCV` to improve Brier score
- **Fairness**: Group-aware sample reweighting + post-hoc probability recalibration, selected by best composite score from Pareto sweep
- **Stability**: 10-seed evaluation loop to sanity-check metric stability (critical for small MS-school subgroup)

## Requirements

- Python 3.9+
- scikit-learn, pandas, numpy, matplotlib, joblib (see `requirements.txt`)
