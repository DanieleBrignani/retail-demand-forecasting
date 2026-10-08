"""Batch forecast generation from an explicitly selected registry version, persisted with exact versions."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..features.build import build_examples
from ..panel import Panel
from ..registry import mlflow_utils as ML
from . import store


class ModelAfterOriginError(RuntimeError):
    """The selected model version was trained with data later than the requested forecast origin."""


def generate_forecast_batch(
    conn,
    panel: Panel,
    model_name: str,
    model_version: str,
    *,
    origin: int | None = None,
    mode: str = "production",
    run_label: str = "default",
    series_idx: np.ndarray | None = None,
    horizons: list[int] | None = None,
) -> dict:
    """Forecast origin = last day with sales in ``panel`` unless an earlier origin is given (historical replay).
    Targets are origin+1..origin+28; only ``panel.as_of(origin)`` information is used."""
    horizons = horizons or list(range(1, 29))
    origin = origin or panel.n_days
    if origin > panel.n_days:
        raise ValueError("origin beyond observed sales")
    if mode == "production" and origin < panel.n_days:
        raise ValueError(
            "production batches must use the latest origin; use mode='historical_replay' for earlier origins"
        )
    sel = (
        panel.eligible_at(origin)
        if series_idx is None
        else np.intersect1d(series_idx, panel.eligible_at(origin))
    )
    point, bundle, info = ML.load_version(model_name, model_version)
    train_origin = info["tags"].get("train_origin")
    if train_origin and int(train_origin) > origin:
        raise ModelAfterOriginError(
            f"model v{model_version} was trained on data through day {train_origin}, after the forecast origin {origin}: "
            "the forecast (and any replay scoring) would use information from after its origin"
        )
    t0 = time.perf_counter()
    pred = point.predict(panel, origin, sel, horizons)
    infer_s = time.perf_counter() - t0
    if bundle is not None:
        iv = bundle.apply(bundle.predict_raw(panel, origin, sel, horizons))
        pred = pred.merge(iv[["series_idx", "h", "lower", "upper"]], on=["series_idx", "h"], how="left")
        # fallback forecasts carry no model interval
        fb = pred["fallback_used"] != ""
        pred.loc[fb, ["lower", "upper"]] = np.nan
        pred["lower"] = np.minimum(pred["lower"], pred["forecast"])
        pred["upper"] = np.maximum(pred["upper"], pred["forecast"])
    rec = store.forecast_records(panel, pred, origin)
    origin_date = str(pd.Timestamp(panel.date_of(origin)).date())
    res = store.write_batch(
        conn,
        rec,
        model_name=model_name,
        model_version=str(model_version),
        dataset_version=panel.dataset_version,
        feature_version=point.feature_version,
        origin_date=origin_date,
        origin_day=origin,
        mode=mode,
        run_label=run_label,
        price_scenario=point.price_scenario,
        generation_time=datetime.now(UTC),
        data_available_at=datetime.now(UTC),
    )
    record_generation_stats(conn, panel, point, origin, rec, res["forecast_batch_id"])
    conn.commit()
    return {
        **res,
        "origin": origin,
        "origin_date": origin_date,
        "inference_seconds": infer_s,
        "fallback_share": float((rec["fallback_used"] != "").mean()),
        "n_series": int(len(sel)),
    }


def record_generation_stats(conn, panel: Panel, point, origin: int, rec: pd.DataFrame, batch_id: str) -> None:
    """Feature missingness + volume + fallback stats, written to PostgreSQL (detail) at generation time."""
    X = build_examples(
        panel,
        origin,
        panel.eligible_at(origin),
        [1],
        point.groups,
        point.enc,
        with_target=False,
        price_scenario=point.price_scenario,
    )
    rows = []
    for c in point.feature_cols:
        rows.append(("data_quality", f"feature_missing:{c}", float(X[c].isna().mean()), {"batch": batch_id}))
    rows.append(("volume", "forecast_rows", float(len(rec)), {"batch": batch_id}))
    rows.append(("volume", "fallback_share", float((rec["fallback_used"] != "").mean()), {"batch": batch_id}))
    for k, v in rec["fallback_used"].value_counts().items():
        if k:
            rows.append(("volume", f"fallback:{k}", float(v), {"batch": batch_id}))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO app.monitoring_snapshots (kind, name, value, detail) VALUES (%s,%s,%s,%s::jsonb)",
            [(k, n, v, __import__("json").dumps(d)) for k, n, v, d in rows],
        )
