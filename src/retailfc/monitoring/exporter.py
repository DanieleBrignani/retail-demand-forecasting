"""Prometheus exporter: bounded-cardinality gauges derived from the PostgreSQL monitoring tables.

Labels are limited to a small closed set (``model_version`` of the current/previous approved model, ``kind``).
Never labelled by item, store, forecast id or timestamp.
"""

from __future__ import annotations

import time
from datetime import date

from prometheus_client import CollectorRegistry, Gauge, generate_latest, start_http_server

from ..warehouse import db
from . import core

REG = CollectorRegistry()
G = {
    "wape": Gauge(
        "retailfc_forecast_wape",
        "Pooled WAPE on matured forecasts",
        ["model_version", "granularity"],
        registry=REG,
    ),
    "bias": Gauge(
        "retailfc_forecast_bias_pct",
        "Signed bias (forecast-actual)/sum(actual)",
        ["model_version", "granularity"],
        registry=REG,
    ),
    "coverage_actuals": Gauge(
        "retailfc_actuals_coverage_ratio", "Matured forecasts that have an actual", registry=REG
    ),
    "freshness": Gauge("retailfc_actuals_age_days", "Days since the latest actual", registry=REG),
    "fallback": Gauge(
        "retailfc_fallback_share", "Share of current-model forecasts using a fallback", registry=REG
    ),
    "rows": Gauge("retailfc_forecast_rows", "Stored forecast rows", registry=REG),
    "drift": Gauge(
        "retailfc_drift_psi", "PSI vs training reference (drift, not accuracy)", ["signal"], registry=REG
    ),
    "dq": Gauge("retailfc_data_quality_failures", "Data-quality failure indicators", ["check"], registry=REG),
    "interval_cov": Gauge(
        "retailfc_interval_coverage",
        "Empirical coverage of stored intervals",
        ["model_version", "granularity"],
        registry=REG,
    ),
    "pipeline_fail": Gauge(
        "retailfc_pipeline_failed_runs_24h", "Failed pipeline runs in the last 24h", registry=REG
    ),
    "up": Gauge("retailfc_exporter_db_up", "Exporter can reach PostgreSQL", registry=REG),
}


def collect(dsn: str | None = None, as_of: date | None = None) -> None:
    try:
        with db.connect(dsn) as conn:
            as_of = as_of or date.today()
            perf = core.performance_by_batch(conn)
            for g in (G["wape"], G["bias"], G["interval_cov"]):
                g.clear()  # drop label sets of models that are no longer approved
            current = conn.execute(
                "SELECT model_name, model_version FROM app.approved_models WHERE revoked_at IS NULL ORDER BY approved_at DESC LIMIT 4"
            ).fetchall()
            for name, v in current:
                sub = perf[(perf.model_version == v) & (perf.model_name == name)]
                if len(sub):
                    gran = str(sub["granularity"].iloc[0])
                    n = sub["n"].sum()
                    G["wape"].labels(v, gran).set(
                        float((sub["wape"] * sub["n"]).sum() / n) if n else float("nan")
                    )
                    G["bias"].labels(v, gran).set(
                        float((sub["bias_pct"] * sub["n"]).sum() / n) if n else float("nan")
                    )
                    cv = sub["coverage"].dropna()
                    if len(cv):
                        G["interval_cov"].labels(v, gran).set(float(cv.mean()))
            cov = core.actuals_coverage(conn, as_of)
            G["coverage_actuals"].set(cov["coverage"] if cov["coverage"] is not None else float("nan"))
            fr = core.data_freshness_days(conn, as_of)
            G["freshness"].set(fr if fr is not None else float("nan"))
            fs = core.fallback_share(conn)
            G["fallback"].set(fs if fs is not None else float("nan"))
            G["rows"].set(conn.execute("SELECT count(*) FROM app.forecasts").fetchone()[0])
            for sig in ("demand_psi", "price_psi"):
                v = core.latest_snapshot(conn, "drift", sig)
                G["drift"].labels(sig).set(v if v is not None else float("nan"))
            for chk in ("missing_actuals_fraction", "actuals_conflicts"):
                v = core.latest_snapshot(conn, "data_quality", chk)
                G["dq"].labels(chk).set(v if v is not None else float("nan"))
            G["pipeline_fail"].set(
                conn.execute(
                    "SELECT count(*) FROM app.pipeline_runs WHERE status='failed' AND started_at > now() - interval '24 hours'"
                ).fetchone()[0]
            )
            G["up"].set(1)
    except Exception:
        G["up"].set(0)


def metrics_text(dsn: str | None = None) -> bytes:
    collect(dsn)
    return generate_latest(REG)


def serve(port: int = 9108, interval: int = 30, dsn: str | None = None) -> None:
    start_http_server(port, registry=REG)
    while True:
        collect(dsn)
        time.sleep(interval)
