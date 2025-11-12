from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def bootstrap_lasso(
    df: pd.DataFrame,
    target: str,
    n_boot: int = 1000,
    n_alphas: int = 100,
    n_folds: int = 5,
    random_state: int = 42,
    alpha_grid: np.ndarray | None = None,   # user-fixed alpha range (optional)
    plot_path: str | None = "alpha_mae_curves.png",
    save_coef_matrix_csv: str | None = "coef_bootstrap_matrix.csv",
    save_coef_summary_csv: str | None = "coef_summary.csv",
):
    """
    Bootstrap Lasso stability analysis (MAE vs alpha curves + effect-size exports).

    - Each bootstrap sample is drawn WITH replacement from the full dataset.
    - For each bootstrap sample, we run K-fold CV for every alpha in `alpha_grid`
      and compute the mean MAE across folds (this yields one MAE curve per sample).
    - We pick the alpha minimizing MAE on that bootstrap, refit on the whole
      bootstrap sample (with StandardScaler), and record non-zero coefficients as one vote.
    - We also store the final refit coefficients per bootstrap to build an
      effect-size matrix and summary (mean/std/Q1/Q3/vote).

    Returns
    -------
    dict with keys:
        alpha_grid: np.ndarray
        mae_matrix: (n_boot, n_alpha) array of MAE curves
        avg_mae:    (n_alpha,) average MAE curve
        best_alphas:(n_boot,) array of best alpha per bootstrap
        feature_votes: pd.Series (feature -> vote count, sorted desc)
        coef_matrix:  pd.DataFrame (n_boot x n_features) final refit coefficients
        coef_summary: pd.DataFrame per-feature summary (mean/std/q25/q75/vote/nonzero_rate)
    """

    rng = np.random.RandomState(random_state)
    # ---- Select numeric features (exclude target and any known meta-columns) ----
    exclude = {target, "__source__"}
    if target not in df.columns:
        raise ValueError(f"target '{target}' not in columns")
    X_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if len(X_cols) == 0:
        raise ValueError("No numeric feature columns found after exclusions.")

    # drop all NA rows
    cols_for_model = [target] + X_cols
    df_clean = df[cols_for_model].dropna(how="any").copy()
    print("[DEBUG] rows_after_dropna:", len(df_clean), "features:", len(X_cols))
    print("[DEBUG] top_nan_cols:", df_clean[X_cols].isnull().sum().sort_values(ascending=False).head(10).to_dict())
    if df_clean.empty:
        raise ValueError("All rows dropped after NaN removal; check CSVs or missing rates.")

    X = df_clean[X_cols].to_numpy()
    y = df_clean[target].to_numpy()

    # 🔹 新增：标准化目标变量 y
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    if not np.isfinite(X).all():
        raise ValueError("X still has NaN/inf after dropna; check io cleaning.")
    if not np.isfinite(y).all():
        raise ValueError("y has NaN/inf after dropna; check target column.")

    # ---- Fix alpha grid (log-spaced by default) so curves align across bootstraps ----
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, 1, num=n_alphas)
    else:
        alpha_grid = np.asarray(alpha_grid)
        n_alphas = len(alpha_grid)

    # ---- Containers for outputs ----
    mae_matrix   = np.empty((n_boot, n_alphas), dtype=float)  # MAE curve per bootstrap
    best_alphas  = np.empty(n_boot, dtype=float)
    feature_votes = pd.Series(0, index=X_cols, dtype=int)     # non-zero coefficient votes
    coef_matrix  = pd.DataFrame(0.0, index=range(n_boot), columns=X_cols)  # effect-size matrix

    print(f"Running {n_boot} bootstrap iterations on {len(X_cols)} features...")
    for i in tqdm(range(n_boot), desc="Bootstrap Lasso"):
        # --- Bootstrap sample WITH replacement ---
        Xb, yb = resample(X, y, replace=True, random_state=random_state + i)

        # --- For this bootstrap, evaluate MAE across alphas using K-fold CV ---
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state + i)
        mae_per_alpha = []

        for a in alpha_grid:
            fold_mae = []
            for tr_idx, va_idx in kf.split(Xb):
                Xtr, Xva = Xb[tr_idx], Xb[va_idx]
                ytr, yva = yb[tr_idx], yb[va_idx]

                # IMPORTANT: Standardize INSIDE each fold to avoid data leakage.
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xva = scaler.transform(Xva)

                model = Lasso(alpha=a, max_iter=10000, random_state=random_state + i)
                model.fit(Xtr, ytr)
                pred = model.predict(Xva)
                fold_mae.append(mean_absolute_error(yva, pred))

            mae_per_alpha.append(np.mean(fold_mae))

        mae_per_alpha = np.asarray(mae_per_alpha)
        mae_matrix[i, :] = mae_per_alpha

        # --- Choose best alpha on this bootstrap (min MAE) ---
        best_idx = int(np.argmin(mae_per_alpha))
        best_alpha = float(alpha_grid[best_idx])
        best_alphas[i] = best_alpha

        # --- Refit on the FULL bootstrap sample with the best alpha (and scaling) ---
        scaler = StandardScaler()
        Xb_scaled = scaler.fit_transform(Xb)
        final_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=random_state + i)
        final_model.fit(Xb_scaled, yb)

        # --- Record non-zero coefficients (vote) and store coefficients (effect sizes) ---
        nonzero = np.array(X_cols)[final_model.coef_ != 0]
        feature_votes[nonzero] += 1
        coef_matrix.loc[i, :] = final_model.coef_

    # ---- Aggregate curves ----
    avg_mae = mae_matrix.mean(axis=0)

    # ---- Build per-feature effect-size summary ----
    coef_summary = pd.DataFrame({
        "mean_coef":  coef_matrix.mean(axis=0),
        "std_coef":   coef_matrix.std(axis=0),
        "q25":        coef_matrix.quantile(0.25, axis=0),
        "q75":        coef_matrix.quantile(0.75, axis=0),
        "vote_count": feature_votes,
    })
    coef_summary["nonzero_rate"] = coef_summary["vote_count"] / n_boot
    coef_summary = coef_summary.sort_values("vote_count", ascending=False)

    # ---- Save CSVs if requested ----
    if save_coef_matrix_csv:
        coef_matrix.to_csv(save_coef_matrix_csv, index_label="bootstrap_id")
    if save_coef_summary_csv:
        coef_summary.to_csv(save_coef_summary_csv, index_label="feature")

    result = {
        "alpha_grid": alpha_grid,
        "mae_matrix": mae_matrix,
        "avg_mae": avg_mae,
        "best_alphas": best_alphas,
        "feature_votes": feature_votes.sort_values(ascending=False),
        "coef_matrix": coef_matrix,
        "coef_summary": coef_summary,
    }

    # ---- Visualization: 1000 transparent curves + solid mean curve ----
    if plot_path:
        plt.figure(figsize=(8, 6))
        for i in range(n_boot):
            plt.plot(alpha_grid, mae_matrix[i], alpha=0.05)
        plt.plot(alpha_grid, avg_mae, linewidth=2, label="Mean MAE")
        plt.xscale("log")
        plt.xlabel("Alpha (λ)")
        plt.ylabel("MAE")
        plt.title(f"Lasso Bootstrap ({n_boot} runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

    return result
