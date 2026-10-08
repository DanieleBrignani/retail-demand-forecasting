> **HISTORICAL — verbatim operational logs, superseded where they concern the daily models.** Model v1 (rejected) and v2 (approved after a post-hoc change of the bias gate: per-window → pooled) both used feature set D `fv1`, which leaked future prices of the origin's block.
> The v2 approval is therefore **not valid evidence and has been revoked** (`retailfc registry revoke`, see [`audit_price_leakage.md`](audit_price_leakage.md)). The v2 promotion is labelled post-hoc/exploratory and was never an independent validation.
> The current lifecycle evidence is the weekly line: [`results_weekly_v2.md`](results_weekly_v2.md) and [`results_weekly_operations.md`](results_weekly_operations.md).
> **Also affected by the unit-overflow defect** (`int16`, 131,072 units lost; [`audit_int16_overflow.md`](audit_int16_overflow.md)): all numbers below were computed on the corrupted panel.

# Generated results — operational cycle on the real dataset (UCI Online Retail II)

> **Provenance.** Verbatim output of `scripts/run_retail_operations.sh` (run 2026-10-04, dataset `retail-369d316281ea`, model trained with data up to
> day 711 = 2011-11-11, forecasts for days 712–739, which were then replayed as actuals). Only log noise (Python/MLflow warnings, Prefect server messages) was removed.
> **Caveats.** (1) The replay window coincides with the already-consumed holdout window: this is an operational demonstration, not a new evaluation.
> (2) The `demand_psi` of 0.0 in the first monitoring block came from a defective PSI (it was blind on zero-inflated data); the code was fixed and monitoring re-run — see the second block.
> (3) Timings come from one run on a laptop with other applications open; they are not a benchmark.

## Run log
```text
=== 22:55:30 train
{
  "status": "registered",
  "run_id": "1b364a5c30ed4fffa9af76106aefe8ed",
  "model_name": "retailfc_lgbm",
  "model_version": "1",
  "decision": {
    "promote": false,
    "checks": [
      {
        "check": "evidence_defined",
        "passed": true,
        "detail": ""
      },
      {
        "check": "beats_seasonal_naive",
        "passed": true,
        "detail": "wape/snaive=0.9055 (max 0.97)"
      },
      {
        "check": "improves_on_incumbent",
        "passed": true,
        "detail": "no incumbent: bootstrap promotion allowed on baseline check"
      },
      {
        "check": "bias_limit",
        "passed": false,
        "detail": "bias -45.35% (max +-10.0%)"
      },
      {
        "check": "interval_coverage",
        "passed": true,
        "detail": "coverage 0.946 (min 0.7)"
      },
      {
        "check": "train_runtime_limit",
        "passed": true,
        "detail": "27s (max 14400)"
      },
      {
        "check": "batch_runtime_limit",
        "passed": true,
        "detail": "0s (max 1800)"
      }
    ],
    "failed": [
      "bias_limit"
    ]
  }
}
=== 22:56:49 forecast
{
  "forecast_batch_id": "fb-2011-11-11-ee98b9514d731ab3",
  "status": "written",
  "n_rows": 263592,
  "origin": 711,
  "origin_date": "2011-11-11",
  "inference_seconds": 0.4662583000026643,
  "fallback_share": 0.11737837263649883,
  "n_series": 9414
}
=== 22:57:16 replay
{
  "received": 263592,
  "inserted": 263592,
  "already_present": 0,
  "conflicts": 0
}
=== 22:57:21 monitor
{
  "reconcile": {
    "new_errors": 263592,
    "unscored_forecasts": 0
  },
  "actuals": {
    "matured_forecasts": 263592,
    "with_actual": 263592,
    "coverage": 1.0,
    "missing_actuals_fraction": 0.0
  },
  "drift": {
    "demand_psi": 0.0,
    "price_psi": 0.010968160003629454,
    "price_missing_share_current": 0.6845549182069258,
    "price_missing_share_reference": 0.7540515731287404
  },
  "performance": [
    {
      "forecast_batch_id": "fb-2011-11-11-ee98b9514d731ab3",
      "model_version": "1",
      "n": 263592.0,
      "wape": 1.0425795301031175,
      "bias_pct": -0.42836667019019264,
      "coverage": 0.9368756769767722
    }
  ]
}
=== 22:57:28 retrain
{
  "decision_id": 1,
  "decision": "blocked_data_quality",
  "reasons": [
    "feature missingness 0.296 > 0.2"
  ],
  "triggers": []
}
=== 22:57:57 inference
{
  "p50_ms": 48.70519996620715,
  "p95_ms": 50.195200013695285,
  "mean_ms": 48.98340666356186,
  "n": 15,
  "series_per_call": 500,
  "rows_per_call": 14000,
  "ms_per_series": 0.09796681332712372
}
=== 22:58:09 done
```

## Monitoring re-run after the PSI fix (same database, same batch; reconciliation is idempotent)
```text
{
  "reconcile": {
    "new_errors": 0,
    "unscored_forecasts": 0
  },
  "actuals": {
    "matured_forecasts": 263592,
    "with_actual": 263592,
    "coverage": 1.0,
    "missing_actuals_fraction": 0.0
  },
  "drift": {
    "demand_psi": 0.03033047623478309,
    "price_psi": 0.010968160003629454,
    "price_missing_share_current": 0.6845549182069258,
    "price_missing_share_reference": 0.7540515731287404
  },
  "performance": [
    {
      "forecast_batch_id": "fb-2011-11-11-ee98b9514d731ab3",
      "model_version": "1",
      "n": 263592.0,
      "wape": 1.0425795301031175,
      "bias_pct": -0.42836667019019264,
      "coverage": 0.9368756769767722
    }
  ]
}
```

## Promotion decision recorded for model version 1 and API behaviour
```text
approved models: 0 | batches: (1, 263592)
series: RETAIL_1_10135 UK_1 -> 503 model_unavailable
ready: 503 model_unavailable
```

## Re-run with model revision v2 (all series, level offset, bias pooled over 3 windows) — post-hoc, see experiment_plan §14.1–14.2
```text
=== 23:38:56 train
{
  "status": "promoted",
  "run_id": "36e9c9da70584a52bccfd1aaff16d6e5",
  "model_name": "retailfc_lgbm",
  "model_version": "2",
  "decision": {
    "promote": true,
    "checks": [
      {
        "check": "evidence_defined",
        "passed": true,
        "detail": ""
      },
      {
        "check": "beats_seasonal_naive",
        "passed": true,
        "detail": "wape/snaive=0.8849 (max 0.97)"
      },
      {
        "check": "improves_on_incumbent",
        "passed": true,
        "detail": "no incumbent: bootstrap promotion allowed on baseline check"
      },
      {
        "check": "bias_limit",
        "passed": true,
        "detail": "pooled bias -8.17% over 3 window(s) (max +-10.0%); per-window (informational): -6.8%, -24.1%, +5.5%"
      },
      {
        "check": "interval_coverage",
        "passed": true,
        "detail": "coverage 0.955 (min 0.7)"
      },
      {
        "check": "train_runtime_limit",
        "passed": true,
        "detail": "67s (max 14400)"
      },
      {
        "check": "batch_runtime_limit",
        "passed": true,
        "detail": "0s (max 1800)"
      }
    ],
    "failed": []
  }
}

=== 23:44:02 forecast
{
  "forecast_batch_id": "fb-2011-11-11-a567c15f592d7d21",
  "status": "written",
  "n_rows": 263592,
  "origin": 711,
  "origin_date": "2011-11-11",
  "inference_seconds": 0.5832985999877565,
  "fallback_share": 0.11737837263649883,
  "n_series": 9414
}

=== 23:44:28 replay
{
  "received": 263592,
  "inserted": 0,
  "already_present": 263592,
  "conflicts": 0
}

=== 23:44:34 monitor
{
  "reconcile": {
    "new_errors": 263592,
    "unscored_forecasts": 0
  },
  "actuals": {
    "matured_forecasts": 527184,
    "with_actual": 527184,
    "coverage": 1.0,
    "missing_actuals_fraction": 0.0
  },
  "drift": {
    "demand_psi": 0.03033047623478309,
    "price_psi": 0.010968160003629454,
    "price_missing_share_current": 0.6845549182069258,
    "price_missing_share_reference": 0.7540515731287404
  },
  "performance": [
    {
      "forecast_batch_id": "fb-2011-11-11-a567c15f592d7d21",
      "model_version": "2",
      "n": 263592.0,
      "wape": 1.1267576526202976,
      "bias_pct": 0.05292504008384573,
      "coverage": 0.9545716348881591
    },
    {
      "forecast_batch_id": "fb-2011-11-11-ee98b9514d731ab3",
      "model_version": "1",
      "n": 263592.0,
      "wape": 1.0425795301031175,
      "bias_pct": -0.42836667019019264,
      "coverage": 0.9368756769767722
    }
  ]
}

=== 23:44:43 retrain
{
  "decision_id": 2,
  "decision": "blocked_data_quality",
  "reasons": [
    "feature missingness 0.296 > 0.2"
  ],
  "triggers": []
}

=== 23:45:09 inference
{
  "p50_ms": 54.19310001889244,
  "p95_ms": 55.780319980112836,
  "mean_ms": 54.144706670194864,
  "n": 15,
  "series_per_call": 500,
  "rows_per_call": 14000,
  "ms_per_series": 0.10828941334038973
}

=== 23:45:20 done
```
> Replay was already present (v1 inserted the same actuals); the DB now holds both batches. v2: WAPE 1.1268, bias +5.3%, coverage 0.955; v1: WAPE 1.0426, bias −42.8%.

## Docker Compose stack with the real forecasts (2026-10-04)
Stack started with `docker compose up -d --wait` (postgres, migrate, mlflow, prefect-server/worker, api, exporter, prometheus, grafana: all healthy; migrate exited 0).
The `app` schema (forecasts, actuals, errors, snapshots, approval record) was copied from the development database `retailfc_retail` with `pg_dump --data-only -n app` — a data copy, not a re-computation (527,184 forecast rows = 2 batches).
Verified by hand: `/health/ready` → `{"status":"ready","model_version":"2","auth":"enabled"}`; `POST /forecast` without key → 401, wrong key → 401, valid key → 200 with stored forecasts of model v2 (mode `historical_replay`);
Prometheus targets (prometheus, api, exporter) up and 12+ alert rules loaded; Grafana has the Prometheus and PostgreSQL datasources and both dashboards; the exporter reports `retailfc_forecast_bias_pct{model_version="2"} = 0.0529`.
Not verified: dashboard panels were not inspected visually; `ActualsStale`-type alerts compare with the wall clock and the data is from 2011 (`retailfc_actuals_age_days` ≈ 5,413), so staleness alerts are not meaningful in this demonstration. The MLflow server in compose has an empty registry (the model files live in the local MLflow used for training); the API serves stored forecasts only, so it does not need them.
