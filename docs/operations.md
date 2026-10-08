# Operations

## Services (Docker Compose, `docker-compose.yml`)
| Service | Role | Host port (default, `.env`) | Depends on (health-gated) |
|---|---|---|---|
| `postgres` | warehouse, forecasts, monitoring, MLflow & Prefect metadata (DBs `retailfc`, `mlflow`, `prefect`) | 5452 | – |
| `migrate` | one-shot `retailfc db migrate` (checksummed, ordered, advisory-locked) | – | postgres |
| `mlflow` | tracking server + Model Registry (Postgres backend, artifact proxy volume) | 5050 | postgres |
| `prefect-server` / `prefect-worker` | single orchestrator; worker serves the scheduled deployments | 4210 | postgres / server, mlflow, migrate |
| `api` | FastAPI `/forecast` (stored-forecast retrieval) | 8010 | migrate |
| `exporter` | PostgreSQL → Prometheus gauges | 9118 | migrate |
| `prometheus` / `grafana` | scrape + alert rules / provisioned datasources and dashboards | 9095 / 3005 | exporter-independent / prometheus, postgres |
| `pipeline` (profile `tools`) | run any CLI command inside the image: `docker compose --profile tools run --rm pipeline retailfc <cmd>` | – | migrate, mlflow |

Ports are offset from the usual defaults and bound to 127.0.0.1. Copy `.env.example` to `.env` (placeholder passwords for local use only). Grafana reads PostgreSQL through the
read-only role `grafana_ro` (SELECT on schema `app`).

## Serving design
`POST /forecast {item_id, store_id, horizon}` (horizon 1..28 for a daily line, **1..4 for the weekly line**) **retrieves precomputed forecasts** of the *currently approved* model version of the served registry name (`serving.api_model_name`; Compose sets `RETAILFC_CONFIG=/app/configs/serving_weekly.yaml` → `retailfc_weekly`) (`app.approved_models`, set only by the promotion rules)
from the latest complete batch. No model is loaded in the API process and no sales history is read, so the API cannot return a forecast from an unapproved model or an unvalidated snapshot.
Response: forecasts with explicit `target_period_start/end` (+ interval bounds when stored), `granularity` and `horizon_unit` (`day` | `week`), `model_version`, `dataset_version`, `feature_version`, `forecast_origin`, `generated_at`, `forecast_batch_id`,
`fallback{used, reasons, horizons}` and the batch `mode` (`production` | `historical_replay`).

| Condition | HTTP | code |
|---|---|---|
| missing or wrong `X-API-Key` | 401 | `unauthorized` (checked before body validation, so nothing about series existence leaks) |
| no API key configured and auth not explicitly disabled | 503 | `auth_not_configured` (fail-closed; `/health/ready` also 503) |
| invalid body / horizon outside 1..28 | 422 | `invalid_request` |
| horizon above the batch's range (weekly: > 4) | 422 | `horizon_out_of_range` |
| unknown (item, store) in the active batch | 404 | `unknown_series` |
| nothing approved | 503 | `model_unavailable` |
| approved model without a complete batch | 503 | `no_forecast_batch` |
| DB down / pool exhausted | 503 | `database_unavailable` |
| more than `max_concurrency` (32) in flight | 503 | `overloaded` (no unbounded queue) |

Every response carries `x-request-id` (client value echoed, else generated); logs are one JSON object per request. Per-request DB statement timeout = `request_timeout_seconds` (5 s).
Probes: `/health/live` (process only), `/health/ready` (DB + approved model + complete batch + auth configured), `/metrics`.

**Authentication.** `POST /forecast` requires header `X-API-Key`. Keys come from `RETAILFC_API_KEYS` (comma-separated; at least 16 characters, weaker keys abort startup; two live keys allow rotation). Comparison is constant-time against every key; the credential is never logged, and failures are logged as `auth_failure` with the request id only. Auth is fail-closed; `RETAILFC_API_AUTH=disabled` exists for local development only and logs a warning. Compose refuses to start the `api` service when `RETAILFC_API_KEYS` is empty (`.env.example` leaves it blank on purpose). Not covered: TLS (terminate at an ingress), per-client quotas/rate limiting, key storage in a secret manager, and `/metrics`, `/health/*` are unauthenticated by design (restrict them at the network layer).

## Monitoring
PostgreSQL holds the detailed aggregates (`monitoring_aggregates` per batch × horizon, `monitoring_snapshots`, `forecast_errors`). Prometheus gets only bounded gauges
(`model_version` + `granularity` labels for the most recently approved versions; `signal`/`check` from fixed sets) — never item IDs, forecast IDs or timestamps (test: `test_ready_and_metrics`).
Three separate families of alerts (`monitoring/prometheus/alerts.yml`):

* **Performance** (matured forecasts only): |bias| > 10% (1h), interval coverage < 70% (1h), fallback share > 10% (1h).
* **Data quality:** exporter can't reach DB, missing-actuals fraction > 5% (30m), conflicting actuals, actuals older than 7 days, failed pipeline runs.
* **Drift** (severity `info`): demand or price PSI > 0.25 for 3h — "investigate"; drift alone is not evidence of lower accuracy.
Thresholds are starting points chosen to match the promotion limits and retraining policy; they have not been tuned on real operating history.

## Command reference (real data)
```bash
retailfc ingest --src <folder with calendar.csv sell_prices.csv sales_daily.csv>   # immutable version + manifest
retailfc db migrate && retailfc db load                                                           # raw schema (long format)
(cd dbt && dbt build --vars '{dataset_version: retail-…}' --profiles-dir .)                   # staging + tests (dbt venv)
retailfc eda
retailfc backtest validation        # baselines + SARIMA subset + LightGBM A..D, validation origins only
retailfc backtest select <experiment_id>
retailfc backtest uncertainty && retailfc backtest interpret
retailfc protocol-lock              # freeze config hash + origins BEFORE the holdout
retailfc backtest holdout           # one-shot
retailfc report build <experiment_id>
retailfc train --feature-set <selected>                       # challenger → evidence → promotion rules → MLflow registry
retailfc forecast generate --model-version <N> [--origin D --mode historical_replay]
retailfc actuals replay <from_day> <to_day>                   # HISTORICAL REPLAY of observed days
retailfc monitor refresh --as-of YYYY-MM-DD --train-origin D
retailfc retrain evaluate --as-of YYYY-MM-DD [--dry-run]
retailfc api serve
retailfc registry rollback --to-version <N> --reason "<why>"       # audited re-approval of an earlier version
retailfc registry revoke <registry-name> <N> --reason "<why>"      # withdraw an approval (audit kept; API → 503 model_unavailable)
# weekly line (granularity=week), frozen protocol configs/weekly_v2.yaml:
retailfc weekly backtest --version <V> --config configs/weekly_v2.yaml       # baselines + LightGBM, mechanical selection, manifest
python scripts/build_weekly_report.py <exp_dir> docs/results_weekly_v2.md    # report generated from artifacts
retailfc weekly register <exp_dir>                                          # REGISTRATION of the selected model (approval is separate)
retailfc weekly approve --model-version <N> <exp_dir>                       # APPROVAL by rule (refused unless selection.json names the model)
retailfc weekly forecast --model-version <N> --origin-block 101 --mode historical_replay --version <V>
bash scripts/run_weekly_operations.sh                                        # register → approve → forecast → replay → reconcile → monitor (real dataset)
bash scripts/run_retail_operations.sh   # (daily line, historical) train → promotion gate → forecast batch → replay actuals → monitoring → retraining decision → inference timing (real dataset)
python scripts/zero_forecast_reference.py <experiment_dir> validation|holdout       # all-zero reference metrics from stored actuals
retailfc benchmark inference <N> ; RETAILFC_API_KEY=<key> retailfc benchmark api --requests 500 --concurrency 1   # writes artifacts/benchmarks/*.json with environment
bash scripts/demo_fixture.sh                               # full chain on the SYNTHETIC fixture (needs a PostgreSQL + dbt env)
```

## Benchmark procedure (`retailfc.benchmark`)
* Model inference: `GlobalLGBM.predict` (feature construction + booster) for a fixed chunk of series × 28 horizons; 2 warm-up calls discarded, 15 timed repetitions; report p50/p95 per call and per series.
* API: 500 POST `/forecast` (horizon 28, random known series, fixed seed) after 20 warm-up requests, concurrency 1 and 8; report p50/p95 and throughput. Because the API only reads storage, these numbers are storage-retrieval latency and are never mixed with model inference.
* Always record hardware, software versions (`environment_info()`), dataset size and subset. Monetary cost is **not** reported (no priced resources). Hardware/timing from the build machine in this repository are in `benchmarks/` only if they were actually run.
