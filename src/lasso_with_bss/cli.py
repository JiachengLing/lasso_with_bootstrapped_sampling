from __future__ import annotations
import argparse
from pathlib import Path
import json

from .io import load_folder_as_df
from .lasso import bootstrap_lasso



def main():
    p = argparse.ArgumentParser(
        description="Bootstrap Lasso stability analysis (MAE vs alpha curves + effect-size exports)."
    )
    p.add_argument("--data", required=True, help="Folder containing CSV files")
    p.add_argument("--pattern", default="*.csv", help="Glob pattern for CSVs, default '*.csv'")
    p.add_argument("--target", required=True, help="Target column name")

    p.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap iterations")
    p.add_argument("--n-alphas", type=int, default=100, help="Number of alphas if grid not provided")
    p.add_argument("--n-folds", type=int, default=5, help="KFold splits")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--alpha-grid", type=str, default=None,
                   help="Comma-separated alpha list (overrides n-alphas), e.g. '0.0001,0.001,0.01,0.1,1'")
    p.add_argument("--plot-path", type=str, default="alpha_mae_curves.png",
                   help="Output path for MAE~alpha plot (set to '' to disable)")
    p.add_argument("--coef-matrix-csv", type=str, default="coef_bootstrap_matrix.csv",
                   help="Path to save coefficient matrix CSV ('' to disable)")
    p.add_argument("--coef-summary-csv", type=str, default="coef_summary.csv",
                   help="Path to save coefficient summary CSV ('' to disable)")
    p.add_argument("--report-json", type=str, default="report.json",
                   help="Path to save a small JSON report")
    p.add_argument("--explicit-drop", action="store_true",
                   help="Enable explicit column dropping (default: False)")

    args = p.parse_args()

    # load data
    df = load_folder_as_df(args.data, args.pattern)

    # ----------------------------------------------------------
    # Auto-drop columns based on pattern rules + explicit rules
    # ----------------------------------------------------------
    drop_config = {}

    # Explicit columns to delete
    explicit_drop = {
        "soiltype_cls5_p",
        "soiltype_mean",
        "forest_mean",
        'forest_cls1_p',
        'soiltype_cls4_p'
    }

    for col in df.columns:
        col_low = col.lower()

        # rule 1: drop *_msr, *_shdi, *_lsi
        if (
                col_low.endswith("_msr") or
                col_low.endswith("_shdi") or
                col_low.endswith("_lsi")
        ):
            drop_config[col] = False
        # rule 2: explicit column drop
        elif col in explicit_drop:
            drop_config[col] = args.explicit_drop
        else:
            drop_config[col] = False

    # alpha grid
    alpha_grid = None
    if args.alpha_grid:
        parts = [s.strip() for s in args.alpha_grid.split(",") if s.strip()]
        alpha_grid = [float(x) for x in parts]

    # normalize empty strings -> None for outputs
    plot_path = args.plot_path if args.plot_path else None
    coef_mat_csv = args.coef_matrix_csv if args.coef_matrix_csv else None
    coef_sum_csv = args.coef_summary_csv if args.coef_summary_csv else None

    res = bootstrap_lasso(
        df=df,
        target=args.target,
        n_boot=args.n_boot,
        n_alphas=args.n_alphas,
        n_folds=args.n_folds,
        random_state=args.seed,
        alpha_grid=alpha_grid,
        plot_path=plot_path,
        save_coef_matrix_csv=coef_mat_csv,
        save_coef_summary_csv=coef_sum_csv,
        drop_config=drop_config,
    )

    # small JSON summary (no big arrays)
    summary = {
        "n_boot": int(args.n_boot),
        "n_alphas": int(len(res["alpha_grid"])),
        "best_alpha_mean": float(res["best_alphas"].mean()),
        "best_alpha_std": float(res["best_alphas"].std()),
        "top_features_by_vote": res["coef_summary"].head(10)["vote_count"].to_dict(),
    }

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

