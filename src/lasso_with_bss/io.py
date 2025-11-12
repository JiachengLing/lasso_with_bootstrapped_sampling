# src/lasso_with_bss/io.py
from __future__ import annotations
import pandas as pd
from pathlib import Path


def load_folder_as_df(folder: str | Path, pattern: str = "*.csv") -> pd.DataFrame:

    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    # === Read and merge all CSV files ===
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["__source__"] = f.name
        dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # === Basic cleaning ===
    # 1. Remove completely empty rows
    df = df.dropna(how="all")

    # 2. Standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # 3. Convert columns to numeric (skip siteID and __source__)
    skip_cols = {"siteid", "__source__"}
    for c in df.columns:
        if c not in skip_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. Drop rows with NaN in any numeric column
    num_cols = [c for c in df.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])]
    before = len(df)
    df = df.dropna(subset=num_cols)
    after = len(df)
    print(f"[Info] Dropped {before - after} rows with NaN in numeric columns ({after} rows remain).")

    # 5. Return the cleaned DataFrame
    return df
