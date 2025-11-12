import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def explore_distribution(df: pd.DataFrame, outdir: str = "plots"):
    import os
    os.makedirs(outdir, exist_ok=True)

    num_cols = df.select_dtypes("number").columns
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue

        # plot histogram
        plt.hist(series, bins=20, color="lightblue", edgecolor="k")
        plt.title(f"{col} (mean={series.mean():.2f}, std={series.std():.2f})")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.savefig(f"{outdir}/{col}_hist.png")
        plt.clf()

        # Check normality (Shapiro-Wilk)
        stat, p = stats.shapiro(series)
        print(f"{col:20s} Shapiro-Wilk p={p:.4f}")

