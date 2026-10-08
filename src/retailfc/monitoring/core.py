"""Monitoring computations. Three distinct concerns are never conflated:

* drift                      -- input/target distribution shift (PSI); does NOT by itself imply lower accuracy,
* data-quality failures      -- missing actuals, feature missingness, conflicting actuals, stale data,
* forecast performance       -- WAPE / bias / interval coverage on *matured* forecasts only.

Detailed aggregates are kept in PostgreSQL; Prometheus receives only bounded-cardinality gauges.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from ..panel import Panel


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population stability index. NaN if either side has fewer than 50 finite values.

    Zero-inflated data (e.g. ~90% zero sales) need care: quantile bins of the raw reference collapse into one bin and the index
    is then 0 for ANY shift. When more than 5% of the reference is exactly zero, zero is its own class and the positive values are
    binned by their own reference quantiles (probabilities of the positive bins are scaled by the non-zero share).
    """
    ref = np.asarray(ref, float)
    cur = np.asarray(cur, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 50 or len(cur) < 50:
        return float("nan")

    def probs(x: np.ndarray, edges: np.ndarray, zero_class: bool) -> np.ndarray:
        if not zero_class:
            return np.histogram(x, edges)[0] / len(x)
        pos = x[x > 0]
        h = np.histogram(pos, edges)[0] / len(x)  # share of ALL values falling in each positive bin
        return np.concatenate([[(x == 0).sum() / len(x)], h])

    zero_class = bool((ref == 0).mean() > 0.05 and (ref > 0).sum() >= 50)
    base = ref[ref > 0] if zero_class else ref
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(probs(ref, edges, zero_class), 1e-4, None)
    c = np.clip(probs(cur, edges, zero_class), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def performance_by_batch(conn) -> pd.DataFrame:
    q = (
        "SELECT m.forecast_batch_id, m.model_version, b.model_name, b.granularity, sum(m.n) AS n, "
        "sum(m.sum_abs_error)/NULLIF(sum(m.sum_actual),0) AS wape, "
        "sum(m.sum_error)/NULLIF(sum(m.sum_actual),0) AS bias_pct, "
        "sum(m.n_in_interval)::float/NULLIF(sum(m.n_with_interval),0) AS coverage "
        "FROM app.monitoring_aggregates m JOIN app.forecast_batches b USING (forecast_batch_id) GROUP BY 1,2,3,4 ORDER BY 1"
    )
    return pd.read_sql_query(q, conn)


def performance_by_horizon(conn, model_version: str | None = None) -> pd.DataFrame:
    q = (
        "SELECT horizon, sum(n) AS n, sum(sum_abs_error)/NULLIF(sum(sum_actual),0) AS wape, "
        "sum(sum_error)/NULLIF(sum(sum_actual),0) AS bias_pct FROM app.monitoring_aggregates "
        "WHERE (%(v)s::text IS NULL OR model_version=%(v)s) GROUP BY 1 ORDER BY 1"
    )
    return pd.read_sql_query(q, conn, params={"v": model_version})


def actuals_coverage(conn, as_of: date) -> dict:
    """Of forecasts whose target date <= as_of (matured), the share with an actual (missing actuals => lower)."""
    # an actual exists for a forecast when EVERY day of its target period (1 day, or the 7-day block) is present
    r = conn.execute(
        "SELECT count(*), count(*) FILTER (WHERE w.n_days = f.target_period_end - f.target_period_start + 1) "
        "FROM app.forecasts f LEFT JOIN LATERAL (SELECT count(*) AS n_days FROM app.actuals a WHERE a.item_id=f.item_id "
        "AND a.store_id=f.store_id AND a.date BETWEEN f.target_period_start AND f.target_period_end) w ON true "
        "WHERE f.target_period_end <= %s",
        (as_of,),
    ).fetchone()
    matured, with_actual = int(r[0]), int(r[1])
    return {
        "matured_forecasts": matured,
        "with_actual": with_actual,
        "coverage": (with_actual / matured) if matured else None,
        "missing_actuals_fraction": ((matured - with_actual) / matured) if matured else None,
    }


def data_freshness_days(conn, as_of: date) -> float | None:
    r = conn.execute("SELECT max(date) FROM app.actuals").fetchone()[0]
    return None if r is None else float((as_of - r).days)


def fallback_share(conn) -> float | None:
    r = conn.execute(
        "SELECT avg((f.fallback_used <> '')::int) FROM app.forecasts f JOIN app.forecast_batches b USING (forecast_batch_id) "
        "JOIN app.approved_models m ON m.model_name=b.model_name AND m.model_version=f.model_version AND m.is_current"
    ).fetchone()[0]
    return None if r is None else float(r)


def compute_drift(panel: Panel, train_origin: int, current_from: int, current_to: int) -> dict:
    """PSI of demand (positive-and-zero sales) and listed prices: reference = 364 days ending at train_origin,
    current = days current_from..current_to. Prices compared per (series, week) listings."""
    ref = panel.sales[:, max(0, train_origin - 364) : train_origin].astype(float).ravel()
    cur = panel.sales[:, current_from - 1 : current_to].astype(float).ravel()
    w_ref = panel.day_week_idx[max(0, train_origin - 364) : train_origin]
    w_cur = panel.day_week_idx[current_from - 1 : current_to]
    pr = panel.price_weekly[:, np.unique(w_ref)].ravel()
    pc = panel.price_weekly[:, np.unique(w_cur)].ravel()
    return {
        "demand_psi": psi(ref, cur),
        "price_psi": psi(pr[np.isfinite(pr)], pc[np.isfinite(pc)]),
        "price_missing_share_current": float(np.isnan(pc).mean()) if len(pc) else None,
        "price_missing_share_reference": float(np.isnan(pr).mean()) if len(pr) else None,
    }


def record_snapshots(conn, rows: list[tuple[str, str, float | None, dict]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO app.monitoring_snapshots (kind, name, value, detail) VALUES (%s,%s,%s,%s::jsonb)",
            [(k, n, None if (v is None or v != v) else float(v), json.dumps(d)) for k, n, v, d in rows],
        )
    conn.commit()


def latest_snapshot(conn, kind: str, name: str) -> float | None:
    r = conn.execute(
        "SELECT value FROM app.monitoring_snapshots WHERE kind=%s AND name=%s ORDER BY taken_at DESC, id DESC LIMIT 1",
        (kind, name),
    ).fetchone()
    return None if r is None else r[0]


def refresh(conn, as_of: date, panel: Panel | None = None, train_origin: int | None = None) -> dict:
    """Recompute aggregates (reconcile) and write performance / data-quality / drift snapshots."""
    from ..forecasting.store import reconcile

    rec = reconcile(conn)
    out: dict = {"reconcile": rec}
    cov = actuals_coverage(conn, as_of)
    out["actuals"] = cov
    rows: list[tuple[str, str, float | None, dict]] = []
    rows.append(("data_quality", "missing_actuals_fraction", cov["missing_actuals_fraction"], cov))
    fr = data_freshness_days(conn, as_of)
    rows.append(("freshness", "actuals_age_days", fr, {}))
    cflt = conn.execute("SELECT count(*) FROM app.actuals_conflicts").fetchone()[0]
    rows.append(("data_quality", "actuals_conflicts", float(cflt), {}))
    perf = performance_by_batch(conn)
    for _, r in perf.iterrows():
        rows.append(
            (
                "performance",
                "wape",
                r["wape"],
                {"batch": r["forecast_batch_id"], "model_version": r["model_version"], "n": int(r["n"])},
            )
        )
        rows.append(
            (
                "performance",
                "bias_pct",
                r["bias_pct"],
                {"batch": r["forecast_batch_id"], "model_version": r["model_version"]},
            )
        )
    if panel is not None and train_origin is not None:
        last_actual = conn.execute("SELECT max(date) FROM app.actuals").fetchone()[0]
        if last_actual is not None:
            days = np.searchsorted(pd.to_datetime(panel.dates), np.datetime64(last_actual), side="right")
            frm = max(train_origin + 1, days - 27)
            if days >= frm and days <= panel.n_days:
                d = compute_drift(panel, train_origin, int(frm), int(days))
                for k, v in d.items():
                    rows.append(("drift", k, v, {"window": [int(frm), int(days)]}))
                out["drift"] = d
    record_snapshots(conn, rows)
    out["performance"] = perf.to_dict("records")
    return out
