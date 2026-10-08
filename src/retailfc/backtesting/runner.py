"""Temporal backtest runner + metric-record generation.

Every model is evaluated on the same origins, series and actuals; predictions are persisted at row level
so aggregate metrics can be recomputed from the parquet files alone.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..panel import Panel
from ..segments import fit_volume_edges, segment_frame
from . import metrics as M

H_RANGES = {"1-7": (1, 7), "8-14": (8, 14), "15-21": (15, 21), "22-28": (22, 28), "1-28": (1, 28)}


@dataclass
class ModelSpec:
    """name/version + a ``fit(panel, origin)`` returning an object with ``predict(panel, origin, sel, horizons)``."""

    name: str
    version: str
    fit: Callable
    restricted_to: np.ndarray | None = None  # restrict to a series subset (e.g. SARIMA)
    timings: list[dict] = field(default_factory=list)


def _demand(panel: Panel, ix: np.ndarray, lo: int, hi: int) -> float:
    return float(panel.sales[ix, lo:hi].sum()) if len(ix) and hi > lo else 0.0


def universe_coverage(
    panel: Panel, series_idx: np.ndarray, origins: list[int], horizons: list[int]
) -> pd.DataFrame:
    """Per origin: how many series are eligible (product observed by the origin), how many are excluded as not-yet-launched, and
    how much of the realised target-window demand those excluded series carry (cold-start demand NOT scored here)."""
    rows = []
    for o in origins:
        elig = np.intersect1d(series_idx, panel.eligible_at(o))
        out = np.setdiff1d(series_idx, elig)
        lo, hi = o, min(o + max(horizons), panel.n_days)
        d_in, d_out = _demand(panel, elig, lo, hi), _demand(panel, out, lo, hi)
        rows.append(
            {
                "origin": o,
                "n_series_universe": int(len(series_idx)),
                "n_eligible": int(len(elig)),
                "n_not_yet_launched": int(len(out)),
                "target_demand_eligible": d_in,
                "target_demand_not_yet_launched": d_out,
                "share_demand_not_yet_launched": d_out / (d_in + d_out) if d_in + d_out > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def attach_truth_and_segments(
    pred: pd.DataFrame,
    panel: Panel,
    origin: int,
    cfg: dict,
    edges: list[float],
    seg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join actuals (only where observed) and origin-time segments."""
    pred = pred.copy()
    pred["origin"] = np.int32(origin)
    pred["target_day"] = (origin + pred["h"].astype(np.int32)).astype(np.int32)
    valid = pred["target_day"] <= panel.n_days
    pred = pred[valid].copy()
    pred["actual"] = panel.sales[pred["series_idx"].to_numpy(), pred["target_day"].to_numpy() - 1].astype(
        np.float32
    )
    seg = segment_frame(panel, origin, cfg, edges) if seg is None else seg
    pred = pred.merge(
        seg[["series_idx", "volume_segment", "intermittency_segment"]], on="series_idx", how="left"
    )
    ids = panel.ids[["store_id", "state_id", "cat_id"]].copy()
    ids["series_idx"] = np.arange(len(ids), dtype=np.int32)
    return pred.merge(ids, on="series_idx", how="left")


def run_backtest(
    panel: Panel,
    cfg: dict,
    specs: list[ModelSpec],
    origins: list[int],
    series_idx: np.ndarray,
    out_dir: Path,
    *,
    volume_edges: list[float],
    horizons: list[int] | None = None,
) -> dict:
    horizons = horizons or list(range(1, cfg["horizon"] + 1))
    out_dir = Path(out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    segs = {o: segment_frame(panel, o, cfg, volume_edges) for o in origins}
    timing = []
    coverage = universe_coverage(panel, series_idx, origins, horizons)
    coverage.to_csv(out_dir / "universe_coverage.csv", index=False)
    for spec in specs:
        parts = []
        sel_base = (
            series_idx if spec.restricted_to is None else np.intersect1d(series_idx, spec.restricted_to)
        )
        for o in origins:
            sel_m = np.intersect1d(
                sel_base, panel.eligible_at(o)
            )  # same eligible universe for every compared model
            t0 = time.perf_counter()
            predictor = spec.fit(panel, o)
            t_fit = time.perf_counter() - t0
            t1 = time.perf_counter()
            pf = predictor.predict(panel, o, sel_m, horizons)
            t_pred = time.perf_counter() - t1
            if "negative_raw" not in pf:
                pf["negative_raw"] = np.int8(0)
            df = attach_truth_and_segments(pf, panel, o, cfg, volume_edges, segs[o])
            df["model_name"], df["model_version"] = spec.name, spec.version
            parts.append(df)
            info = getattr(predictor, "info", None)
            timing.append(
                {
                    "model": spec.name,
                    "origin": o,
                    "fit_seconds": t_fit,
                    "predict_seconds": t_pred,
                    "n_predictions": int(len(pf)),
                    "best_iteration": getattr(info, "best_iteration", None),
                    "n_train_rows": getattr(info, "n_train_rows", None),
                    "peak_mem_mb": getattr(info, "peak_mem_mb", None),
                }
            )
        pd.concat(parts, ignore_index=True).to_parquet(
            out_dir / "predictions" / f"{spec.name}.parquet", index=False
        )
    pd.DataFrame(timing).to_csv(out_dir / "timings.csv", index=False)
    return {"models": [s.name for s in specs], "origins": origins, "n_series": int(len(series_idx))}


# ------------------------------------------------------------------------- metric records
SEGMENT_COLS = {
    "store": "store_id",
    "state": "state_id",
    "category": "cat_id",
    "volume": "volume_segment",
    "intermittency": "intermittency_segment",
}


def _rows(
    mdf: pd.DataFrame,
    base: dict,
    seg_type: str,
    seg_val_col: str | None,
    horizon,
    aggregation: str,
    origin,
    extra_n: str = "n",
) -> list[dict]:
    recs = []
    for idx, r in mdf.iterrows():
        sv = idx if seg_val_col is None else idx
        if isinstance(sv, tuple):
            sv = sv[0]
        for metric in M.METRICS:
            v = r.get(metric)
            recs.append(
                {
                    **base,
                    "forecast_origin": origin,
                    "horizon": horizon,
                    "segment_type": seg_type,
                    "segment_value": str(sv) if seg_type != "all" else "all",
                    "aggregation": aggregation,
                    "metric": metric,
                    "value": None if pd.isna(v) else float(v),
                    "observation_count": int(r[extra_n]) if extra_n in r else int(r.get("n", 0)),
                    "undefined": bool(pd.isna(v)),
                }
            )
    return recs


def metric_records(pred: pd.DataFrame, base: dict) -> pd.DataFrame:
    """pred: concatenated prediction frames for ONE model (all origins). ``base`` carries ids/versions."""
    recs: list[dict] = []
    pred = pred.copy()
    for rng_name, (lo, hi) in H_RANGES.items():
        sub = pred[(pred["h"] >= lo) & (pred["h"] <= hi)]
        for o_label, o_df in [(None, sub), *[(int(o), g) for o, g in sub.groupby("origin")]]:
            if o_df.empty:
                continue
            recs += _rows(M.pooled(o_df), base, "all", None, rng_name, "pooled", o_label)
            recs += _rows(M.aggregated(o_df), base, "all", None, rng_name, "aggregated", o_label)
            recs += _rows(
                M.per_series_mean(o_df), base, "all", None, rng_name, "per_series_mean", o_label, "n"
            )
            for st, col in SEGMENT_COLS.items():
                recs += _rows(M.pooled(o_df, [col]), base, st, col, rng_name, "pooled", o_label)
                if rng_name == "1-28":
                    recs += _rows(
                        M.per_series_mean(o_df, [col]), base, st, col, rng_name, "per_series_mean", o_label
                    )
                    recs += _rows(M.aggregated(o_df, [col]), base, st, col, rng_name, "aggregated", o_label)
    for o_label, o_df in [(None, pred), *[(int(o), g) for o, g in pred.groupby("origin")]]:
        mh = M.pooled(o_df, ["h"])
        for h, r in mh.iterrows():
            for metric in M.METRICS:
                v = r[metric]
                recs.append(
                    {
                        **base,
                        "forecast_origin": o_label,
                        "horizon": str(int(h)),
                        "segment_type": "all",
                        "segment_value": "all",
                        "aggregation": "pooled",
                        "metric": metric,
                        "value": None if pd.isna(v) else float(v),
                        "observation_count": int(r["n"]),
                        "undefined": bool(pd.isna(v)),
                    }
                )
    return pd.DataFrame(recs)


def compute_all_metrics(
    out_dir: Path, base: dict, *, scope_by_model: dict[str, str] | None = None
) -> pd.DataFrame:
    frames = []
    for f in sorted((Path(out_dir) / "predictions").glob("*.parquet")):
        pred = pd.read_parquet(f)
        name = pred["model_name"].iloc[0]
        b = {
            **base,
            "model_name": name,
            "model_version": pred["model_version"].iloc[0],
            "evaluation_scope": (scope_by_model or {}).get(name, "full_panel"),
        }
        frames.append(metric_records(pred, b))
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(Path(out_dir) / "metrics.parquet", index=False)
    return res


def load_predictions(out_dir: Path, models: list[str] | None = None) -> pd.DataFrame:
    parts = []
    for f in sorted((Path(out_dir) / "predictions").glob("*.parquet")):
        if models is None or f.stem in models:
            parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def edges_for(panel: Panel, first_val_origin: int, cfg: dict) -> list[float]:
    return fit_volume_edges(panel, first_val_origin, cfg)
