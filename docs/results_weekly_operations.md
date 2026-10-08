# Weekly line — operational replay on the real dataset (UCI Online Retail II)

> **Provenance.** Produced by `bash scripts/run_weekly_operations.sh` (2026-10-06, dataset `retail-369d316281ea`, model `retailfc_weekly` **v3** = `mean4`, origin block 101 = 2011-11-07, targets blocks 102–105) and the SQL/CLI checks below against the development PostgreSQL (`retailfc_retail`).
> Versions 1–2 were registered on evidence computed on the **int16-corrupted panel** ([`audit_int16_overflow.md`](audit_int16_overflow.md)); they are tagged `superseded` in MLflow and the development database was rebuilt from scratch, so only version 3 (corrected evidence) has a database approval record. The earlier daily batches and the revocation record of the daily v2 approval lived in that database and are documented, not recreated.
> It is a **historical replay** of already-inspected data: it demonstrates the lifecycle, not independent forecasting performance. Registration, approval and deployment are separate steps.

## What the guards do (all tested in `tests/test_weekly_lifecycle.py`, one shown below on real data)
* **Approval** is refused unless the selection record names the registered model, the manifest's dataset version / protocol version / config hash equal those registered, and `selection.json` / `manifest.json` hash to the values recorded at registration (the hashes are stored in the approval evidence below).
* **Generation** is refused if the configuration file does not hash to the one the version was registered with, or if the dataset version differs (unless explicitly allowed).
* **Selection checks** for forecast coverage and runtime are *measured* (a missing measurement is not a pass): see `results_weekly_v2.md`.

## Checks that matter
* Database-side weekly WAPE of the replayed batch = **0.6933732439938132**, bias −15.2% — identical (to all printed digits) to the backtest value for `mean4` at origin 101 (`per_origin.csv`): storage, replay and reconciliation preserve the numbers.
* A second reconciliation produced **0 new errors**; 0 duplicate error rows; 0 unscored forecasts after all days were released (periods are scored only when all 7 daily actuals exist; tested with partial releases).
* The daily model v2 approval is **revoked** with a reason (audit trail kept); the daily line therefore answers `503 model_unavailable` while the weekly line is served.
* Intervals are not attached to weekly forecasts (`lower`/`upper` null): they remain experimental.

## Evidence (verbatim)
```text
### approved_models (audit trail; MLflow versions 1-2 were registered on the corrupted panel and are tagged superseded)
 id |   model_name    | model_version |  approved_by   | is_current | revoked 
----+-----------------+---------------+----------------+------------+---------
  1 | retailfc_weekly | 3             | rule:weekly_v2 | t          | f
(1 row)

### forecast batches
        forecast_batch_id        |   model_name    | model_version | granularity |       mode        | forecast_origin | n_rows | protocol_version 
---------------------------------+-----------------+---------------+-------------+-------------------+-----------------+--------+------------------
 fbw-2011-11-07-6623903dc424f813 | retailfc_weekly | 3             | week        | historical_replay | 2011-11-07      |  37400 | weekly-v2
(1 row)

### unit conservation (days 708-739): source CSV = raw.sales_long = app.actuals
raw.sales_long|894740
app.actuals|894740
max single cell (app.actuals)|80995
source CSV | 894740
### second reconciliation (idempotency)
{"new_errors": 0, "unscored_forecasts": 0} {"matured_forecasts": 37400, "with_actual": 37400, "coverage": 1.0, "missing_actuals_fraction": 0.0}
### weekly error rows by horizon and duplicate check
 model_version | horizon |  n   | wape_by_h 
---------------+---------+------+-----------
 3             |       1 | 9350 |    0.6740
 3             |       2 | 9350 |    0.6538
 3             |       3 | 9350 |    0.7258
 3             |       4 | 9350 |    0.7242
(4 rows)

 duplicate_error_rows 
----------------------
                    0
(1 row)

### approval evidence recorded for the current weekly version
{"manifest_sha256": "a7e7e3cbeb93fdf1ffa0617a97096b41adcdfe4d3fef200b4bc122d902346337", "selection_sha256": "97953b331c8cad53820e7dabb4ae1f3c1c2f8b32fc391c314864912f303ac6d1"}|82550af575f407ae
### guard: generation with a different configuration is refused
IncompatibleConfig the configuration file does not match the one this model version was registered with (SHA-256 c241dbd41daf… != 82550af575f4…); refusing to generate
```

## API behaviour on the rebuilt Docker Compose stack (manual checks)
`/health/ready` → ready with the current weekly version; `POST /forecast` without key → 401; with key, `horizon: 4` → 200 with `granularity: "week"`, `horizon_unit: "week"`, explicit `target_period_start/end`, `mode: "historical_replay"`;
`horizon: 5` → 422 `horizon_out_of_range`; unknown series → 404 `unknown_series`. Prometheus targets (prometheus, api, exporter) up, alert rules healthy and none firing, Grafana datasources and both dashboards provisioned. Dashboard panels were not inspected visually.
Staleness alerts compare to the wall clock and the data is from 2011: they are not meaningful in this demonstration. The Compose database holds a copy of the `app` schema from the development database (a data copy, not a recomputation).
