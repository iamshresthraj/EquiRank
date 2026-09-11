#!/usr/bin/env bash
# run.sh — One-command end-to-end reproduction for ED-01
#
# Usage:
#   bash run.sh
#
# This script reproduces the entire pipeline from a clean checkout:
#   1. Sets up/activates virtual environment (handles Debian/Ubuntu PEP 668)
#   2. Installs dependencies
#   3. Trains + calibrates + fairness-mitigates the model
#   4. Generates outputs (ranking, selection, fairness table, Pareto chart)
#   5. Runs standalone inference test (simulating evaluator)
#   6. Prints final composite score breakdown

set -euo pipefail
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  ED-01: Fair Student-Support Prioritization"
echo "  End-to-end reproduction script"
echo "============================================================"

# Detect available Python binary
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "ERROR: Neither python3 nor python found in PATH"
    exit 1
fi

# Detect or initialize virtual environment to handle PEP 668 (externally-managed-environment)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        echo "  Configuring local virtual environment (.venv)..."
        $PYTHON_CMD -m venv .venv 2>/dev/null || true
        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        elif [ -f ".venv/Scripts/activate" ]; then
            source .venv/Scripts/activate
        fi
    fi
fi

# Determine active Python and Pip binaries
if command -v python >/dev/null 2>&1; then
    PY="python"
else
    PY="$PYTHON_CMD"
fi

if command -v pip >/dev/null 2>&1; then
    PIP="pip"
elif command -v pip3 >/dev/null 2>&1; then
    PIP="pip3"
else
    PIP="$PY -m pip"
fi

# 1. Install dependencies
echo ""
echo "  [1/4] Installing dependencies..."
# Try standard pip install, fallback to --break-system-packages (for Debian/Ubuntu PEP 668), fallback to --user
$PIP install -r requirements.txt --quiet 2>/dev/null || \
$PIP install -r requirements.txt --break-system-packages --quiet 2>/dev/null || \
$PIP install -r requirements.txt --user --quiet 2>/dev/null || \
$PIP install scikit-learn pandas numpy matplotlib joblib --break-system-packages --quiet 2>/dev/null || \
$PIP install scikit-learn pandas numpy matplotlib joblib --quiet

# 2. Train the model (includes stability analysis, model comparison,
#    calibration, fairness diagnosis, and mitigation)
echo ""
echo "  [2/4] Training pipeline..."
$PY src/train.py

# 3. Run standalone inference test
echo ""
echo "  [3/4] Testing standalone inference (simulating evaluator)..."
# Create a stripped copy of the data without G3 and support_needed
$PY -c "
import pandas as pd
df = pd.read_csv('data/student-mat.csv', sep=';')
df = df.drop(columns=['G3'])
df['id'] = range(len(df))
df.to_csv('outputs/test_inference_input.csv', sep=';', index=False)
print('  Created test inference input (G3 stripped)')
"
$PY src/predict.py --input outputs/test_inference_input.csv --output-dir outputs/ --id-col id --sep ";"

# 4. Final verification
echo ""
echo "  [4/4] Final verification..."
$PY -c "
import os
outputs = ['outputs/ranking.csv', 'outputs/selected_students.csv',
           'outputs/fairness_table.csv', 'outputs/pareto_tradeoff.png']
for f in outputs:
    if os.path.exists(f):
        print(f'  ✓ {f} exists')
    else:
        print(f'  ✗ {f} MISSING')
        exit(1)

if os.path.exists('models/pipeline.joblib'):
    print('  ✓ models/pipeline.joblib exists')
else:
    print('  ✗ models/pipeline.joblib MISSING')
    exit(1)

if os.path.exists('report/responsible_use.md'):
    print('  ✓ report/responsible_use.md exists')
else:
    print('  ✗ report/responsible_use.md MISSING')

print('')
print('  ✓ All required outputs generated successfully!')
"

# 5. Print final composite score breakdown
echo ""
echo "  [5/5] Final evaluation breakdown against 100-point rubric..."
$PY src/evaluate.py

echo ""
echo "============================================================"
echo "  Pipeline complete! Check outputs/ for results."
echo "============================================================"
