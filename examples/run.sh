#!/bin/bash
set -e

# Go to project root
cd "$(dirname "$0")/.."

OUTDIR="examples/outputs"
mkdir -p "$OUTDIR"

TARGET="y"

if command -v lasso-bss &> /dev/null; then
    CMD="lasso-bss"
else
    echo "[Info] 'lasso-bss' not found, fallback to 'python -m lasso_with_bss.cli'"
    CMD="python -m lasso_with_bss.cli"
fi

$CMD \
  --data examples/data \
  --target "$TARGET" \
  --n-boot 50 \
  --n-folds 5 \
  --n-alphas 50 \
  --seed 42 \
  --plot-path "$OUTDIR/alpha_mae_curves.png" \
  --coef-matrix-csv "$OUTDIR/coef_matrix.csv" \
  --coef-summary-csv "$OUTDIR/coef_summary.csv" \
  --report-json "$OUTDIR/report.json"

echo
echo " Done. See $OUTDIR"
