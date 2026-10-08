"""Reference metrics of the trivial "forecast zero everywhere" model, computed from the stored actuals.

Why: with extremely intermittent demand (here ~92% of series-days are zero) WAPE and MAE are minimised by predicting nothing.
For an all-zero forecast WAPE is exactly 1 by definition, so any model with pooled WAPE >= 1 is not better than predicting zero on that
metric. Reads the validation predictions written by `retailfc backtest validation`; no model is rerun.

    python scripts/zero_forecast_reference.py <experiment_dir> [validation|holdout]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(exp_dir: str, stage: str = "validation") -> None:
    p = Path(exp_dir) / stage / "predictions"
    ref = pd.read_parquet(p / "naive.parquet", columns=["origin", "series_idx", "h", "actual"])
    a = ref["actual"].to_numpy(float)

    def row(df: pd.DataFrame, label) -> dict:
        x = df["actual"].to_numpy(float)
        return {
            "scope": label,
            "n": len(x),
            "wape": 1.0 if x.sum() > 0 else np.nan,
            "mae": x.mean(),
            "rmse": float(np.sqrt((x**2).mean())),
            "zero_share_of_actuals": float((x == 0).mean()),
        }

    rows = [row(ref, "all origins")] + [row(g, f"origin {int(o)}") for o, g in ref.groupby("origin")]
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(
        f"\n(sum of actuals = {a.sum():.0f}; a model with pooled WAPE >= 1 is not better than predicting zero on WAPE)"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
