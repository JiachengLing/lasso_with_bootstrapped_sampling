from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    drop_config: dict[str, bool] | None = None,
):
    """
    Bootstrap Lasso stability analysis (MAE vs alpha curves + effect-size exports).

    Workflow
    --------
    1. Select numeric feature columns from `df`, excluding:
       - the target column
       - the special column "__source__"
       - any columns explicitly marked as "to drop" in `drop_config`.

    2. Drop rows with any NaN in target or feature columns.

    3. Scale the target `y` to [0, 1] with MinMaxScaler.

    4. For each bootstrap iteration:
       - Draw a bootstrap sample (with replacement) from (X, y).
       - For each alpha in `alpha_grid`, run K-fold CV and compute the mean MAE.
       - Select the alpha with the lowest MAE.
       - Refit Lasso on the full bootstrap sample (with StandardScaler for X).
       - Record:
         * the MAE curve for this bootstrap
         * the chosen alpha
         * non-zero coefficients (as votes for features)
         * the full coefficient vector for effect-size statistics.

    5. Aggregate across bootstraps to obtain:
       - average MAE curve vs alpha
       - per-feature coefficient statistics and selection frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame containing the target and predictors.
    target : str
        Name of the target column in `df`.
    n_boot : int, default=1000
        Number of bootstrap iterations.
    n_alphas : int, default=100
        Number of alpha values (ignored if `alpha_grid` is provided).
    n_folds : int, default=5
        Number of folds for K-fold cross-validation.
    random_state : int, default=42
        Base random seed for reproducibility.
    alpha_grid : np.ndarray or None, default=None
        Optional explicit array of alpha values. If None, a log-spaced grid
        between 1e-6 and 1e0 is created with length `n_alphas`.
    plot_path : str or None, default="alpha_mae_curves.png"
        If not None, save the MAE vs alpha curves plot to this path.
    save_coef_matrix_csv : str or None, default="coef_bootstrap_matrix.csv"
        If not None, save the bootstrap coefficient matrix to this CSV path.
    save_coef_summary_csv : str or None, default="coef_summary.csv"
        If not None, save the per-feature summary statistics to this CSV path.
    drop_config : dict[str, bool] or None, default=None
        Optional mapping "column_name -> drop_flag".
        If drop_flag is True, the column is **excluded** from the model.
        If drop_flag is False, the entry is ignored and the column is handled
        by the default logic (numeric + not in the base exclude set).

    Returns
    -------
    dict
        A dictionary with the following entries:
        - "alpha_grid"     : np.ndarray, alpha values used.
        - "mae_matrix"     : np.ndarray, shape (n_boot, n_alphas),
                             MAE curve per bootstrap.
        - "avg_mae"        : np.ndarray, shape (n_alphas,),
                             average MAE across bootstraps.
        - "best_alphas"    : np.ndarray, shape (n_boot,),
                             best alpha selected for each bootstrap.
        - "feature_votes"  : pd.Series, feature -> non-zero count (sorted desc).
        - "coef_matrix"    : pd.DataFrame, shape (n_boot, n_features),
                             Lasso coefficients per bootstrap.
        - "coef_summary"   : pd.DataFrame, per-feature summary:
                             mean_coef, std_coef, q25, q75,
                             vote_count, nonzero_rate.
    """

    rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # 1) Build the base exclude set (always removed from features)
    # ------------------------------------------------------------------
    if target not in df.columns:
        raise ValueError(f"target '{target}' not in dataframe columns.")

    exclude: set[str] = {target, "__source__"}

    # Apply user-defined drop configuration
    # If drop_config[col] is True -> column is excluded
    if drop_config is not None:
        for col, to_drop in drop_config.items():
            if to_drop:
                exclude.add(col)

    # ------------------------------------------------------------------
    # 2) Select numeric feature columns after exclusions
    # ------------------------------------------------------------------
    X_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(X_cols) == 0:
        raise ValueError("No numeric feature columns found after exclusions.")

    # Keep only target + selected feature columns, then drop rows with NaN
    cols_for_model = [target] + X_cols
    df_clean = df[cols_for_model].dropna(how="any").copy()

    print("[DEBUG] rows_after_dropna:", len(df_clean), "features:", len(X_cols))
    print("[DEBUG] top_nan_cols:",
          df_clean[X_cols].isnull().sum().sort_values(ascending=False).head(10).to_dict())

    if df_clean.empty:
        raise ValueError("All rows dropped after NaN removal; check input data and missing rates.")

    # ------------------------------------------------------------------
    # 3) Extract X, y and scale y to [0, 1]
    # ------------------------------------------------------------------
    X = df_clean[X_cols].to_numpy()
    y = df_clean[target].to_numpy()

    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    if not np.isfinite(X).all():
        raise ValueError("X still has NaN/inf after dropna; check preprocessing.")
    if not np.isfinite(y).all():
        raise ValueError("y has NaN/inf after dropna or scaling; check target column.")

    # ------------------------------------------------------------------
    # 4) Fix alpha grid so curves are aligned across bootstraps
    # ------------------------------------------------------------------
    if alpha_grid is None:
        alpha_grid = np.logspace(-6, 0, num=n_alphas)
    else:
        alpha_grid = np.asarray(alpha_grid)
        n_alphas = len(alpha_grid)

    # ------------------------------------------------------------------
    # 5) Allocate containers for outputs
    # ------------------------------------------------------------------
    mae_matrix = np.empty((n_boot, n_alphas), dtype=float)        # MAE curve per bootstrap
    best_alphas = np.empty(n_boot, dtype=float)                   # best alpha per bootstrap
    feature_votes = pd.Series(0, index=X_cols, dtype=int)         # non-zero coefficient votes
    coef_matrix = pd.DataFrame(0.0, index=range(n_boot), columns=X_cols)  # effect-size matrix

    print(f"Running {n_boot} bootstrap iterations on {len(X_cols)} features...")

    # ------------------------------------------------------------------
    # 6) Main bootstrap loop
    # ------------------------------------------------------------------
    for i in tqdm(range(n_boot), desc="Bootstrap Lasso"):
        # 6.1) Bootstrap sample with replacement
        Xb, yb = resample(X, y, replace=True, random_state=random_state + i)

        # 6.2) Evaluate MAE across alphas using K-fold CV on this bootstrap
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state + i)
        mae_per_alpha = []

        for a in alpha_grid:
            fold_mae = []

            for tr_idx, va_idx in kf.split(Xb):
                Xtr, Xva = Xb[tr_idx], Xb[va_idx]
                ytr, yva = yb[tr_idx], yb[va_idx]

                # Standardize features *inside* each fold to avoid data leakage
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

        # 6.3) Choose best alpha (minimum MAE) for this bootstrap
        best_idx = int(np.argmin(mae_per_alpha))
        best_alpha = float(alpha_grid[best_idx])
        best_alphas[i] = best_alpha

        # 6.4) Refit Lasso on the full bootstrap sample with the best alpha
        scaler = StandardScaler()
        Xb_scaled = scaler.fit_transform(Xb)

        final_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=random_state + i)
        final_model.fit(Xb_scaled, yb)

        # 6.5) Record non-zero coefficients as votes and store coefficients
        nonzero = np.array(X_cols)[final_model.coef_ != 0]
        feature_votes[nonzero] += 1
        coef_matrix.loc[i, :] = final_model.coef_

    # ------------------------------------------------------------------
    # 7) Aggregate MAE curves
    # ------------------------------------------------------------------
    avg_mae = mae_matrix.mean(axis=0)

    # ------------------------------------------------------------------
    # 8) Build per-feature effect-size summary
    # ------------------------------------------------------------------
    coef_summary = pd.DataFrame({
        "mean_coef":  coef_matrix.mean(axis=0),
        "std_coef":   coef_matrix.std(axis=0),
        "q25":        coef_matrix.quantile(0.25, axis=0),
        "q75":        coef_matrix.quantile(0.75, axis=0),
        "vote_count": feature_votes,
    })
    coef_summary["nonzero_rate"] = coef_summary["vote_count"] / n_boot
    coef_summary = coef_summary.sort_values("vote_count", ascending=False)

    # ------------------------------------------------------------------
    # 9) Save CSV outputs if requested
    # ------------------------------------------------------------------
    if save_coef_matrix_csv:
        coef_matrix.to_csv(save_coef_matrix_csv, index_label="bootstrap_id")
    if save_coef_summary_csv:
        coef_summary.to_csv(save_coef_summary_csv, index_label="feature")

    # ------------------------------------------------------------------
    # 10) Plot MAE vs alpha curves if requested
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 11) Pack and return results
    # ------------------------------------------------------------------
    result = {
        "alpha_grid": alpha_grid,
        "mae_matrix": mae_matrix,
        "avg_mae": avg_mae,
        "best_alphas": best_alphas,
        "feature_votes": feature_votes.sort_values(ascending=False),
        "coef_matrix": coef_matrix,
        "coef_summary": coef_summary,
    }

    return result


