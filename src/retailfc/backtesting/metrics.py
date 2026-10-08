"""Forecast accuracy metrics. Exact definitions (also in docs/experiment_plan.md):

Let a = actual, f = forecast, e = f - a  (sign convention: **positive bias = over-forecast**).

* MAE   = mean(|e|)
* RMSE  = sqrt(mean(e^2))
* WAPE  = sum(|e|) / sum(|a|)       -- undefined (NaN, ``wape_undefined`` flagged) if sum(|a|) == 0.
          Pooled WAPE weights series by volume; high-volume series dominate (a documented limitation).
* sMAPE = mean( 200 * |e| / (|a| + |f|) ) in percent (range 0..200); terms with a == f == 0 are defined as 0
          and counted (``n_both_zero``). Because retail demand is intermittent this convention makes sMAPE optimistic.
* bias  = mean(e)                    (units), ``bias_pct`` = sum(e) / sum(|a|) (NaN if denominator 0)

Three aggregation views:
* pooled          -> stats over all (series, origin, h) rows in the group,
* per_series_mean -> metric per series, then unweighted mean over series (series with undefined metric excluded, counted),
* aggregated      -> forecasts and actuals summed over series per (origin, target day), then metric over those totals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["mae", "rmse", "wape", "smape", "bias", "bias_pct"]


def _stats(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level building blocks; summed per group they are sufficient for every pooled metric."""
    a = df["actual"].to_numpy(np.float64)
    f = df["forecast"].to_numpy(np.float64)
    e = f - a
    den = np.abs(a) + np.abs(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        sm = np.where(den == 0, 0.0, 200.0 * np.abs(e) / den)
    return pd.DataFrame(
        {
            "n": 1,
            "abs_e": np.abs(e),
            "sq_e": e * e,
            "sum_e": e,
            "abs_a": np.abs(a),
            "smape": sm,
            "both_zero": (den == 0).astype(np.int64),
        },
        index=df.index,
    )


def metrics_from_sums(s: pd.DataFrame) -> pd.DataFrame:
    n = s["n"].replace(0, np.nan)
    out = pd.DataFrame(index=s.index)
    out["mae"] = s["abs_e"] / n
    out["rmse"] = np.sqrt(s["sq_e"] / n)
    out["wape"] = np.where(s["abs_a"] > 0, s["abs_e"] / s["abs_a"].replace(0, np.nan), np.nan)
    out["smape"] = s["smape"] / n
    out["bias"] = s["sum_e"] / n
    out["bias_pct"] = np.where(s["abs_a"] > 0, s["sum_e"] / s["abs_a"].replace(0, np.nan), np.nan)
    out["wape_undefined"] = (s["abs_a"] == 0).astype(int)
    out["n"] = s["n"].astype(int)
    out["n_both_zero"] = s["both_zero"].astype(int)
    return out


def pooled(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    st = _stats(df)
    st = pd.concat([df[by or []], st], axis=1) if by else st
    if by:
        g = st.groupby(by, observed=True).sum(numeric_only=True)
    else:
        g = st.sum(numeric_only=True).to_frame().T
    return metrics_from_sums(g)


def per_series_mean(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Unweighted mean over series of the per-series metric (WAPE-undefined series excluded and counted)."""
    keys = [*(by or []), "series_idx"]
    per = pooled(df, keys).reset_index()
    grp = per.groupby(by, observed=True) if by else per.assign(_k=0).groupby("_k")
    rows = []
    for k, g in grp:
        rows.append(
            {
                **(dict(zip(by, k if isinstance(k, tuple) else (k,), strict=True)) if by else {}),
                "mae": g["mae"].mean(),
                "rmse": g["rmse"].mean(),
                "wape": g["wape"].mean(),
                "smape": g["smape"].mean(),
                "bias": g["bias"].mean(),
                "bias_pct": g["bias_pct"].mean(),
                "n_series": len(g),
                "n_series_wape_undefined": int(g["wape"].isna().sum()),
                "n": int(g["n"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.set_index(by) if by else out


def aggregated(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Metrics on demand summed across series per (origin, target day) within each group."""
    keys = [*(by or []), "origin", "target_day"]
    tot = df.groupby(keys, observed=True)[["actual", "forecast"]].sum().reset_index()
    return pooled(tot, by)


def add_smape_note() -> str:
    return "sMAPE: 200*|e|/(|a|+|f|), both-zero -> 0, reported in % (0..200)."
