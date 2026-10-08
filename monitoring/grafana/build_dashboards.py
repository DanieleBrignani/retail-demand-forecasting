"""Regenerates monitoring/grafana/dashboards/*.json (kept as code so the JSON is reviewable)."""

import json
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prom"}
PG = {"type": "grafana-postgresql-datasource", "uid": "pg"}


def ts(id_, title, exprs, x, y, w=8, h=7, unit=None):
    return {
        "id": id_,
        "type": "timeseries",
        "title": title,
        "datasource": PROM,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": lf} for i, (e, lf) in enumerate(exprs)],
        "fieldConfig": {"defaults": {"unit": unit or "short"}, "overrides": []},
    }


def stat(id_, title, expr, x, y, unit="percentunit", w=4, h=4):
    return {
        "id": id_,
        "type": "stat",
        "title": title,
        "datasource": PROM,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": "A", "expr": expr}],
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
    }


def sql(id_, title, q, x, y, w=12, h=8, fmt="table", viz="table"):
    return {
        "id": id_,
        "type": viz,
        "title": title,
        "datasource": PG,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": "A", "rawSql": q, "format": fmt, "rawQuery": True, "editorMode": "code"}],
    }


serving = {
    "uid": "retailfc-serving",
    "title": "RetailFC - Serving and operations",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"},
    "tags": ["retailfc"],
    "panels": [
        ts(
            1,
            "Forecast requests / s by status",
            [('sum(rate(retailfc_api_requests_total{endpoint="/forecast"}[5m])) by (status)', "{{status}}")],
            0,
            0,
            unit="reqps",
        ),
        ts(
            2,
            "API latency p50 / p95 (stored-forecast retrieval)",
            [
                (
                    'histogram_quantile(0.5, sum(rate(retailfc_api_request_seconds_bucket{endpoint="/forecast"}[5m])) by (le))',
                    "p50",
                ),
                (
                    'histogram_quantile(0.95, sum(rate(retailfc_api_request_seconds_bucket{endpoint="/forecast"}[5m])) by (le))',
                    "p95",
                ),
            ],
            8,
            0,
            unit="s",
        ),
        ts(
            3,
            "DB retrieval latency p95",
            [
                (
                    "histogram_quantile(0.95, sum(rate(retailfc_api_retrieval_seconds_bucket[5m])) by (le))",
                    "p95",
                )
            ],
            16,
            0,
            unit="s",
        ),
        stat(4, "Exporter DB up", "retailfc_exporter_db_up", 0, 7, unit="short"),
        stat(5, "Stored forecast rows", "retailfc_forecast_rows", 4, 7, unit="short"),
        stat(6, "Fallback share (current model)", "retailfc_fallback_share", 8, 7),
        stat(7, "Actuals age (days)", "retailfc_actuals_age_days", 12, 7, unit="d"),
        stat(8, "Matured forecasts with actual", "retailfc_actuals_coverage_ratio", 16, 7),
        stat(9, "Failed pipeline runs (24h)", "retailfc_pipeline_failed_runs_24h", 20, 7, unit="short"),
        sql(
            10,
            "Pipeline runs",
            "SELECT run_id, flow, status, started_at, finished_at FROM app.pipeline_runs ORDER BY run_id DESC LIMIT 20",
            0,
            11,
            24,
            7,
        ),
    ],
}

quality = {
    "uid": "retailfc-forecast-quality",
    "title": "RetailFC - Forecast quality, drift and data quality",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "1m",
    "time": {"from": "now-7d", "to": "now"},
    "tags": ["retailfc"],
    "panels": [
        ts(
            1,
            "WAPE on matured forecasts (by model version)",
            [("retailfc_forecast_wape", "{{granularity}} v{{model_version}}")],
            0,
            0,
            8,
            unit="percentunit",
        ),
        ts(
            2,
            "Bias (forecast - actual) / demand",
            [("retailfc_forecast_bias_pct", "{{granularity}} v{{model_version}}")],
            8,
            0,
            8,
            unit="percentunit",
        ),
        ts(
            3,
            "Empirical interval coverage",
            [("retailfc_interval_coverage", "{{granularity}} v{{model_version}}")],
            16,
            0,
            8,
            unit="percentunit",
        ),
        ts(
            4,
            "DRIFT: PSI vs training reference (not an accuracy signal)",
            [("retailfc_drift_psi", "{{signal}}")],
            0,
            7,
            12,
        ),
        ts(5, "DATA QUALITY indicators", [("retailfc_data_quality_failures", "{{check}}")], 12, 7, 12),
        sql(
            6,
            "WAPE by horizon (matured forecasts, all batches)",
            "SELECT horizon, sum(sum_abs_error)/NULLIF(sum(sum_actual),0) AS wape, sum(sum_error)/NULLIF(sum(sum_actual),0) AS bias_pct, sum(n) AS n "
            "FROM app.monitoring_aggregates GROUP BY 1 ORDER BY 1",
            0,
            14,
            12,
            8,
        ),
        sql(
            7,
            "Performance by forecast batch (vintage)",
            "SELECT forecast_batch_id, model_version, sum(n) AS n, sum(sum_abs_error)/NULLIF(sum(sum_actual),0) AS wape "
            "FROM app.monitoring_aggregates GROUP BY 1,2 ORDER BY 1 DESC",
            12,
            14,
            12,
            8,
        ),
        sql(
            8,
            "Retraining decision audit trail",
            "SELECT decision_id, decided_at, decision, reasons->'triggers' AS triggers, challenger_version FROM app.retraining_decisions ORDER BY decision_id DESC LIMIT 20",
            0,
            22,
            24,
            8,
        ),
    ],
}

out = Path(__file__).parent / "dashboards"
out.mkdir(exist_ok=True)
for d in (serving, quality):
    (out / f"{d['uid']}.json").write_text(json.dumps(d, indent=1))
print("wrote", [p.name for p in out.glob("*.json")])
