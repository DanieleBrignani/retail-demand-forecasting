"""Forecast generation records, idempotent transactional batch writes, actuals ingestion, error reconciliation."""

from __future__ import annotations

import hashlib
import io
import uuid
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import psycopg

from ..panel import Panel

NS = uuid.UUID("5d7a6c52-6c0e-4f7e-9a53-0e1f4f6b2a11")


class ForecastConflict(RuntimeError):
    """A batch id exists as complete with different content; forecasts are never overwritten."""


def make_batch_id(
    model_version: str,
    dataset_version: str,
    origin_date: str,
    run_label: str = "default",
    *,
    model_name: str | None = None,
    granularity: str = "day",
) -> str:
    """Daily ids keep their original formula (existing vintages stay addressable). Weekly ids also hash the model name and
    granularity, because daily and weekly registry versions both start at 1."""
    key = f"{model_version}|{dataset_version}|{origin_date}|{run_label}"
    if granularity != "day":
        key = f"{granularity}|{model_name}|{key}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"fb-{origin_date}-{h}" if granularity == "day" else f"fbw-{origin_date}-{h}"


def forecast_records(panel: Panel, pred: pd.DataFrame, origin: int) -> pd.DataFrame:
    """Join model output (series_idx, h, forecast[, lower, upper, fallback_used]) to dates/keys."""
    ids = panel.ids[["item_id", "store_id"]].iloc[pred["series_idx"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "item_id": ids["item_id"],
            "store_id": ids["store_id"],
            "target_date": pd.to_datetime(panel.dates[origin + pred["h"].to_numpy() - 1]).strftime(
                "%Y-%m-%d"
            ),
            "horizon": pred["h"].to_numpy().astype(np.int16),
            "horizon_unit": "day",
            "forecast_value": pred["forecast"].to_numpy().astype(np.float64),
            "lower_bound": pred["lower"].to_numpy() if "lower" in pred else np.nan,
            "upper_bound": pred["upper"].to_numpy() if "upper" in pred else np.nan,
            "fallback_used": pred["fallback_used"].fillna("").to_numpy() if "fallback_used" in pred else "",
        }
    )
    out["target_period_start"] = out["target_date"]
    out["target_period_end"] = out["target_date"]
    return out.sort_values(["item_id", "store_id", "target_date"]).reset_index(drop=True)


def content_hash(rec: pd.DataFrame) -> str:
    cols = ["item_id", "store_id", "target_date", "horizon", "forecast_value", "fallback_used"]
    if (
        "horizon_unit" in rec and (rec["horizon_unit"] != "day").any()
    ):  # weekly content also covers the period and unit
        cols = [*cols, "horizon_unit", "target_period_end"]
    x = rec[cols].copy()
    x["forecast_value"] = x["forecast_value"].round(6)
    return hashlib.sha256(x.to_csv(index=False).encode()).hexdigest()


def write_batch(
    conn: psycopg.Connection,
    rec: pd.DataFrame,
    *,
    model_name: str,
    model_version: str,
    dataset_version: str,
    feature_version: str,
    origin_date: str,
    origin_day: int,
    mode: str,
    run_label: str = "default",
    price_scenario: str = "observed",
    data_available_at: datetime | None = None,
    generation_time: datetime | None = None,
    granularity: str = "day",
    protocol_version: str = "",
) -> dict:
    """One transaction. Same batch id + same content -> no-op. Same id + different content -> ForecastConflict."""
    if (rec["forecast_value"] < 0).any() or rec["forecast_value"].isna().any():
        raise ValueError("refusing to store negative/NaN forecasts (clip and flag upstream)")
    if rec.duplicated(["item_id", "store_id", "target_date"]).any():
        raise ValueError("duplicate (item, store, target_date) in batch")
    if granularity not in ("day", "week"):
        raise ValueError(f"unknown granularity {granularity!r}")
    unit = rec["horizon_unit"].iloc[0] if "horizon_unit" in rec and len(rec) else granularity
    if unit != granularity or ("horizon_unit" in rec and (rec["horizon_unit"] != granularity).any()):
        raise ValueError(
            "every row's horizon_unit must equal the batch granularity (daily and weekly are never mixed)"
        )
    if "target_period_start" not in rec:
        rec = rec.assign(
            horizon_unit=granularity,
            target_period_start=rec["target_date"],
            target_period_end=rec["target_date"],
        )
    bid = make_batch_id(
        model_version, dataset_version, origin_date, run_label, model_name=model_name, granularity=granularity
    )
    chash = content_hash(rec)
    gen = generation_time or datetime.now(UTC)
    with conn.transaction():
        row = conn.execute(
            "SELECT status, content_hash FROM app.forecast_batches WHERE forecast_batch_id=%s FOR UPDATE",
            (bid,),
        ).fetchone()
        if row:
            status, old = row
            if status == "complete":
                if old == chash:
                    return {"forecast_batch_id": bid, "status": "noop_identical", "n_rows": len(rec)}
                raise ForecastConflict(
                    f"batch {bid} already complete with different content; use a new run_label"
                )
            conn.execute("DELETE FROM app.forecasts WHERE forecast_batch_id=%s", (bid,))
            conn.execute("DELETE FROM app.forecast_batches WHERE forecast_batch_id=%s", (bid,))
        conn.execute(
            "INSERT INTO app.forecast_batches (forecast_batch_id, model_name, model_version, dataset_version, "
            "feature_version, forecast_origin, forecast_origin_day, generation_time, data_available_at, mode, "
            "price_scenario, run_label, granularity, protocol_version, status) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'writing')",
            (
                bid,
                model_name,
                model_version,
                dataset_version,
                feature_version,
                origin_date,
                origin_day,
                gen,
                data_available_at,
                mode,
                price_scenario,
                run_label,
                granularity,
                protocol_version,
            ),
        )
        ids = [
            str(uuid.uuid5(NS, f"{bid}|{i}|{s}|{t}"))
            for i, s, t in zip(rec.item_id, rec.store_id, rec.target_date, strict=True)
        ]
        df = pd.DataFrame(
            {
                "forecast_id": ids,
                "forecast_batch_id": bid,
                "model_version": model_version,
                "dataset_version": dataset_version,
                "feature_version": feature_version,
                "forecast_origin": origin_date,
                "generation_time": gen.isoformat(),
                "item_id": rec.item_id,
                "store_id": rec.store_id,
                "target_date": rec.target_date,
                "horizon": rec.horizon,
                "horizon_unit": rec.horizon_unit,
                "target_period_start": rec.target_period_start,
                "target_period_end": rec.target_period_end,
                "forecast_value": rec.forecast_value,
                "lower_bound": rec.lower_bound,
                "upper_bound": rec.upper_bound,
                "fallback_used": rec.fallback_used,
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, header=False, index=False, na_rep="\\N")
        cols = list(df.columns)
        with (
            conn.cursor() as cur,
            cur.copy(
                f"COPY app.forecasts ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')"
            ) as cp,
        ):
            cp.write(buf.getvalue())
        conn.execute(
            "UPDATE app.forecast_batches SET status='complete', n_rows=%s, content_hash=%s WHERE forecast_batch_id=%s",
            (len(df), chash, bid),
        )
    return {"forecast_batch_id": bid, "status": "written", "n_rows": len(df)}


# --------------------------------------------------------------------------- actuals
def ingest_actuals(conn: psycopg.Connection, df: pd.DataFrame, source: str) -> dict:
    """df: item_id, store_id, date, sales. Idempotent; differing values for an existing key go to actuals_conflicts."""
    if (df["sales"] < 0).any() or df["sales"].isna().any():
        raise ValueError("invalid sales in actuals (negative/NaN); refusing the whole batch")
    if df.duplicated(["item_id", "store_id", "date"]).any():
        raise ValueError("duplicate keys within the actuals batch")
    with conn.transaction():
        conn.execute(
            "CREATE TEMP TABLE _act (item_id TEXT, store_id TEXT, date DATE, sales INT) ON COMMIT DROP"
        )
        buf = io.StringIO()
        df[["item_id", "store_id", "date", "sales"]].astype({"sales": int}).to_csv(
            buf, header=False, index=False
        )
        with conn.cursor() as cur, cur.copy("COPY _act FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(buf.getvalue())
        conflicts = conn.execute(
            "INSERT INTO app.actuals_conflicts (item_id, store_id, date, existing_sales, incoming_sales) "
            "SELECT a.item_id, a.store_id, a.date, e.sales, a.sales FROM _act a JOIN app.actuals e "
            "USING (item_id, store_id, date) WHERE e.sales <> a.sales"
        ).rowcount
        inserted = conn.execute(
            "INSERT INTO app.actuals (item_id, store_id, date, sales, source) SELECT item_id, store_id, date, sales, %s "
            "FROM _act ON CONFLICT (item_id, store_id, date) DO NOTHING",
            (source,),
        ).rowcount
    return {
        "received": len(df),
        "inserted": inserted,
        "already_present": len(df) - inserted - conflicts,
        "conflicts": conflicts,
    }


def replay_actuals_frame(panel: Panel, from_day: int, to_day: int) -> pd.DataFrame:
    """HISTORICAL REPLAY: release observed days from_day..to_day as if they arrived now. Days beyond the
    observed range are never produced."""
    if to_day > panel.n_days:
        raise ValueError(f"cannot replay day {to_day}: observed sales end at day {panel.n_days}")
    cols = slice(from_day - 1, to_day)
    m = panel.sales[:, cols]
    n, T = m.shape
    return pd.DataFrame(
        {
            "item_id": np.repeat(panel.ids.item_id.to_numpy(), T),
            "store_id": np.repeat(panel.ids.store_id.to_numpy(), T),
            "date": np.tile(pd.to_datetime(panel.dates[cols]).strftime("%Y-%m-%d"), n),
            "sales": m.reshape(-1).astype(int),
        }
    )


def reconcile(conn: psycopg.Connection) -> dict:
    """Score only forecasts whose actual exists; derive errors without touching forecasts."""
    with conn.transaction():
        n = conn.execute(
            "INSERT INTO app.forecast_errors (forecast_id, forecast_batch_id, model_version, horizon, target_date, actual, "
            "error, abs_error, sq_error, in_interval, horizon_unit, target_period_end) "
            "SELECT f.forecast_id, f.forecast_batch_id, f.model_version, f.horizon, f.target_date, a.sales, "
            "f.forecast_value - a.sales, abs(f.forecast_value - a.sales), power(f.forecast_value - a.sales, 2), "
            "CASE WHEN f.lower_bound IS NULL THEN NULL ELSE a.sales BETWEEN f.lower_bound AND f.upper_bound END, "
            "'day', f.target_date "
            "FROM app.forecasts f JOIN app.actuals a ON a.item_id=f.item_id AND a.store_id=f.store_id AND a.date=f.target_date "
            "WHERE f.horizon_unit='day' ON CONFLICT (forecast_id) DO NOTHING"
        ).rowcount
        # weekly forecasts are scored against the SUM of the daily actuals of their period, and only when every day is present
        n += conn.execute(
            "INSERT INTO app.forecast_errors (forecast_id, forecast_batch_id, model_version, horizon, target_date, actual, "
            "error, abs_error, sq_error, in_interval, horizon_unit, target_period_end) "
            "SELECT f.forecast_id, f.forecast_batch_id, f.model_version, f.horizon, f.target_date, w.actual, "
            "f.forecast_value - w.actual, abs(f.forecast_value - w.actual), power(f.forecast_value - w.actual, 2), "
            "CASE WHEN f.lower_bound IS NULL THEN NULL ELSE w.actual BETWEEN f.lower_bound AND f.upper_bound END, "
            "'week', f.target_period_end "
            "FROM app.forecasts f JOIN LATERAL (SELECT sum(a.sales)::float AS actual, count(*) AS n_days FROM app.actuals a "
            "WHERE a.item_id=f.item_id AND a.store_id=f.store_id AND a.date BETWEEN f.target_period_start AND f.target_period_end) w "
            "ON w.n_days = (f.target_period_end - f.target_period_start + 1) "
            "WHERE f.horizon_unit='week' ON CONFLICT (forecast_id) DO NOTHING"
        ).rowcount
        conn.execute("DELETE FROM app.monitoring_aggregates")
        conn.execute(
            "INSERT INTO app.monitoring_aggregates (forecast_batch_id, model_version, horizon, n, sum_abs_error, sum_error, "
            "sum_sq_error, sum_actual, n_in_interval, n_with_interval) "
            "SELECT forecast_batch_id, model_version, horizon, count(*), sum(abs_error), sum(error), sum(sq_error), sum(actual), "
            "count(*) FILTER (WHERE in_interval), count(in_interval) FROM app.forecast_errors GROUP BY 1,2,3"
        )
        pending = conn.execute(
            "SELECT count(*) FROM app.forecasts f WHERE NOT EXISTS (SELECT 1 FROM app.forecast_errors e "
            "WHERE e.forecast_id=f.forecast_id)"
        ).fetchone()[0]
    return {"new_errors": n, "unscored_forecasts": pending}
