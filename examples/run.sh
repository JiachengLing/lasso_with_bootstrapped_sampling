#!/usr/bin/env bash
set -euo pipefail

# 切到仓库根目录（run.sh 所在目录是 examples/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATADIR="examples/all_data"
OUTROOT="examples/outputs"

mkdir -p "$OUTROOT"

CMD="python -m lasso_with_bss.cli"

echo "Running double-mode Lasso for all datasets..."
echo

for f in "$DATADIR"/*.csv; do
    # 防止目录里没有 csv 的情况
    [ -e "$f" ] || continue

    FILE="$(basename "$f")"       # Col.csv
    BASENAME="${FILE%.csv}"       # Col

    echo "------------------------------------------"
    echo "Processing $FILE"
    echo "------------------------------------------"

    # ==========================
    # ====== Mode E (drop) =====
    # ==========================
    OUTDIR="$OUTROOT/${BASENAME}_E"
    mkdir -p "$OUTDIR"

    $CMD \
        --data "$DATADIR" \
        --pattern "$FILE" \
        --target y \
        --explicit-drop \
        --n-boot 1000 \
        --n-folds 10 \
        --n-alphas 50 \
        --plot-path "$OUTDIR/alpha_mae.png" \
        --coef-matrix-csv "$OUTDIR/coef_matrix.csv" \
        --coef-summary-csv "$OUTDIR/coef_summary.csv" \
        --report-json "$OUTDIR/report.json"

    # ==========================
    # ====== Mode G (keep) =====
    # ==========================
    OUTDIR="$OUTROOT/${BASENAME}_G"
    mkdir -p "$OUTDIR"

    $CMD \
        --data "$DATADIR" \
        --pattern "$FILE" \
        --target y \
        --n-boot 1000 \
        --n-folds 10 \
        --n-alphas 50 \
        --plot-path "$OUTDIR/alpha_mae.png" \
        --coef-matrix-csv "$OUTDIR/coef_matrix.csv" \
        --coef-summary-csv "$OUTDIR/coef_summary.csv" \
        --report-json "$OUTDIR/report.json"

done

echo
echo "All done!"
