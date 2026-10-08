> **HISTORICAL README of the daily study (kept verbatim, not edited).** It describes the work before the price-look-ahead fix, the universe policy and the weekly line. Where it conflicts with the current [README](../README.md) or [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md), the current documents win. Daily numbers in it are superseded (see [`results_daily_fv2.md`](results_daily_fv2.md) and [`audit_price_leakage.md`](audit_price_leakage.md)).

# Retail Demand Forecasting Platform — UCI Online Retail II

An executable, reproducible forecasting platform: validated immutable ingestion → PostgreSQL/dbt staging → leakage-safe feature pipeline →
rolling-origin backtesting → LightGBM / SARIMA / baselines → MLflow Registry with objective promotion rules → batch forecast storage → actuals reconciliation →
monitoring (PostgreSQL + Prometheus + Grafana) → Prefect-scheduled retraining *decisions*.

> **Honest status.** The platform is implemented and verified on a **synthetic fixture** (mechanics only: `pytest`, `scripts/demo_fixture.sh`).
> The real dataset is **UCI Online Retail II** (public, CC BY 4.0). **Forecast accuracy on it is reported only in the generated report**
> (`retailfc report build`, see §15) — if that report does not exist yet, **no accuracy result is claimed anywhere in this repository**.
> No benchmark number comes from synthetic data. This is not a production-readiness claim: see [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).

## 1. Business Question
*Can historical sales, calendar events and prices be turned into reliable future demand forecasts that outperform simple baselines, and can the system be retrained reproducibly?*
Treated as six hypotheses (H1–H6) in [`docs/experiment_plan.md`](docs/experiment_plan.md): LightGBM vs naive/seasonal naive; value of calendar, event and price features; which segments benefit;
horizon behaviour; cost vs benefit; failure regimes.

## 2. Problem
Forecast units per product × sales channel × day, 28 days ahead (API: 1–28), without using information that would not have been available at the forecast origin — on very intermittent demand.

## 3. Objective
Measured, reproducible answers (not assumptions) plus an operational loop: retrain only when evidence says so, promote only when objective rules pass.

## 4. Dataset
**UCI Online Retail II** — https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0): 1,067,371 transaction lines of one UK-based online gift retailer, 2009-12-01 to 2011-12-09
(43 countries; UK ≈ 92% of lines). Downloaded with the user's explicit permission: zip 45,622,418 bytes; `online_retail_II.xlsx` SHA-256
`bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.

`retailfc adapt-online-retail` converts the transactions into three canonical CSVs (`calendar.csv`, `sales_daily.csv`, `sell_prices.csv`); every choice is explicit and written to `adapter_report.json`
(reproduce: `retailfc adapt-online-retail --xlsx <file> --out <dir>`):
* **cleaning:** cancellation invoices (19,494 lines), non-positive quantity (22,950), non-positive price (6,207) and non-product codes (`POST`, `ADJUST`, `BANK CHARGES`, …) are excluded, **not netted** → 1,036,877 lines kept;
* **panel:** 4,707 product codes × 2 sales **channels** (`UK_1` = United Kingdom, `INT_1` = all other countries — channels, not physical stores) = **9,414 series over 739 calendar days**; one department/category (the source has none);
  **91.7% of series-days are zero**; 1,039 series never sell;
* days without any transaction (e.g. Saturdays, closures) are structural zeros; **prices** are weekly medians of the week's sale lines and are absent in weeks without sales (never 0);
* **events** = England & Wales public holidays from the `holidays` package (**my assumption**, not source data).

`retailfc ingest` stores an immutable, read-only, checksummed version (`data/raw/retail-<sha12>/manifest.json`: source, file names, SHA-256, ingestion timestamp, schema version `retail-raw-v1`, observed date range, dataset version).
Identical bytes → verified no-op; changed bytes → new version; tampered files are detected, never overwritten.

Because there are no real stores, regions or categories, hypotheses about those cannot be tested; segments are volume, intermittency and channel (`docs/limitations.md`).

## 5. Architecture
```
Online Retail II xlsx ─adapt─▶ canonical CSVs ─ingest/validate─▶ data/raw/<version> (immutable) ─▶ PostgreSQL raw.* ─dbt─▶ staging.* (+tests)
                                                        │                                                      │
                                                        └───────────▶ Panel (dense, as_of(origin) boundary) ◀──┘  (equality tested)
Panel ─▶ features (origin-anchored, direct multi-horizon) ─▶ baselines · SARIMA subset · LightGBM A..D
      ─▶ rolling-origin backtests ─▶ metrics/predictions parquet ─▶ report
Locked protocol ─▶ single holdout ─▶ MLflow registry (explicit versions) ─▶ promotion rules ─▶ app.approved_models
Batch forecast ─▶ app.forecasts (idempotent) ─▶ API (stored-forecast retrieval, API key) ; actuals ─▶ errors ─▶ monitoring ─▶ Prefect decision flow
```
One shared library `src/retailfc/` serves the CLI, Prefect flows, tests, API and notebook. Layout: `adapters/ ingestion/ warehouse/ features/ models/ backtesting/ registry/ forecasting/ monitoring/ orchestration/ api/`,
`dbt/`, `monitoring/` (Prometheus, Grafana), `docs/`, `tests/`, `notebooks/`.

## 6. Data Pipeline
Validation (Python) + dbt staging with tests: duplicate keys, missing dates, invalid identifiers, negative/invalid sales and prices, join cardinality, calendar coverage, referential integrity, unexpected source schema changes.
Zero sales are observations; absent prices stay absent. Long-format loading is chunked; the dense `Panel` stores sales as int16 (9,414 × 739 × 2 bytes ≈ 14 MB here); training windows/series are configurable (`training.*`).

## 7. EDA Findings
`retailfc eda` (training period only) writes tables, charts and computed interpretations to `<artifacts>/eda/eda_summary.md`. Mandatory caveats: zero ≠ zero demand; no inventory; first sale ≠ launch; association ≠ causation.
The dataset-level facts above (91.7% zeros, 1,039 never-selling series, 604 trading days out of 739) come from `adapter_report.json`.

## 8. Feature Engineering
Definitions, time reference, availability, missing policy and meaning per feature: [`docs/feature_dictionary.md`](docs/feature_dictionary.md) (generated from `features/spec.py`).

## 9. Leakage Prevention
All data access goes through `Panel.as_of(origin)`; lags are `y_{o−k}`; rolling windows end at the origin; the target-aligned seasonal lag is `y_{t−7⌈h/7⌉}` (≤ origin); encoders are fitted on training data only; prices are never back-filled;
a planned-price scenario is separate. Proven by **mutation tests** (random future sales/prices must not change earlier-origin features, baselines or a retrained model's forecast) — `tests/test_features_leakage.py`, `tests/test_experiment_e2e.py`.
Replays refuse models trained after the forecast origin. **Caveat specific to this dataset:** prices are derived from the same transactions, so price features can carry same-week sales information (`docs/data_availability.md`).

## 10. Hypotheses and Experimental Design
[`docs/experiment_plan.md`](docs/experiment_plan.md): primary metric pooled WAPE (limitation stated), origins, final holdout, selection rules, ablation, compute budget, scope of SARIMA, fallback policy, locked-protocol mechanism.
With N = 739 days: validation origins 627 / 655 / 683, holdout origin 711.

## 11. Baselines
Naive, seasonal naive (last 7 days, target-weekday aligned), trailing mean (28 d), seasonal mean (4 weeks). Insufficient history → flagged `short_history`; no positive sale → 0 + `all_zero_history`. Always reported next to any model.

## 12. Models
Global direct LightGBM (Tweedie), quantile LightGBM + conformalised calibration, SARIMA/ARIMA on a restricted, training-selected subset with inner-validation order choice and flagged fallbacks. No deep learning (nothing measured justifies it).

## 13. Backtesting
Rolling origin; 3 non-overlapping validation windows + holdout; never a random split. Metric records contain all required fields (`backtest_id, experiment_id, dataset_version, feature_version, forecast_origin, horizon, model_name, model_version, evaluation_scope, segment_type, segment_value, metric, value, observation_count`) plus `aggregation` (pooled / per-series mean / aggregated demand). Row-level predictions are retained so metrics can be recomputed (tested).

## 14. Feature Ablation
A (history+ids+horizon) → B (+calendar) → C (+holiday events) → D (+price). Non-improving groups are reported; adoption needs ≥ `selection.min_relative_gain` (0.5%, in `experiment.yaml`) relative validation gain.

## 15. Results
Generated by `bash scripts/run_retail_case_study.sh` (+ `retailfc report build`); a verbatim copy of the report is in [`docs/results_online_retail.md`](docs/results_online_retail.md) (dataset `retail-369d316281ea`, config hash `d071e536f2b72358`).
Pooled over 9,414 series and horizons 1–28:

| | Validation WAPE (3 origins) | Holdout WAPE (1 origin) |
|---|---|---|
| LightGBM (set D) | **1.115** | **1.043** |
| mean baseline (28 d) | 1.259 | 1.147 |
| seasonal mean | 1.271 | 1.152 |
| naive | 1.309 | 1.428 |
| seasonal naive | 1.360 | 1.302 |
| *all-zero forecast (reference)* | *1.000* | *1.000* |

* LightGBM has the lowest WAPE and MAE of all non-trivial models in every validation origin and in the locked holdout.
* **No model beats the all-zero forecast on WAPE or MAE** (WAPE ≥ 1 everywhere; ~90% of series-days are zero), and the **mean baseline has the lowest RMSE** in both periods. The all-zero reference was not in the pre-registered protocol; it is computed from stored actuals (`scripts/zero_forecast_reference.py`).
* LightGBM's advantage sits in high-volume, UK-channel series; low-volume, inactive and "intermittent" series are served better by simple baselines. Calendar, event and price features show **no demonstrated value** (details and caveats: [`docs/model_card.md`](docs/model_card.md), [`docs/error_analysis.md`](docs/error_analysis.md)).
* Holdout is a single 28-day window; there is no significance claim.

**Update (model revision v2, post-hoc):** v1 below was rejected. A v2 (all series + level offset, [`docs/experiment_plan.md`](docs/experiment_plan.md) §14) did **not** meet its pre-registered criteria on untouched windows (per-window bias +19%/−26%/+18%; WAPE/MAE worse than v1 and above the all-zero forecast). After seeing that, I changed the gate to judge bias pooled over 3 windows (post-hoc, disclosed); v2 then passed (pooled −8.2%) and is **registered/approved as version 2** by the rules — with holdout WAPE 1.1268, worse than v1's 1.0426. Bias improved, accuracy by WAPE did not.

**Weekly study (pre-registered, [`docs/weekly_protocol.md`](docs/weekly_protocol.md); results [`docs/results_weekly.md`](docs/results_weekly.md)):** at weekly granularity the best models reach WAPE ≈ 0.83–0.85 (grand-total level 16%). The global LightGBM is only ~2% better than a 4-week mean on WAPE and **fails 2 of the 4 pre-registered criteria** (MASE worse than the best baseline in 0/10 origins; interval coverage 0.915 > 0.90), so it is **not** declared a serving candidate and was not substituted for the approved daily v2.

**Operational cycle on the same real data** ([`docs/results_online_retail_operations.md`](docs/results_online_retail_operations.md), `bash scripts/run_retail_operations.sh`):
* The challenger was trained (data ≤ day 711), registered in MLflow as version 1 and **rejected by the promotion rules**: it passes the seasonal-naive ratio (0.906 ≤ 0.97), interval coverage (0.946) and runtime checks but fails the **bias limit (−45.35% vs ±10%)**. No model is therefore approved.
* Forecasts were still generated for that explicit version: 263,592 rows (9,414 series × 28 days), 11.7% flagged fallbacks (series with no earlier sale); the next 28 days were replayed as actuals (263,592 inserted, 0 conflicts) and reconciled (0 unscored).
  The database aggregate (WAPE 1.0426, bias −0.428) equals the backtest's holdout figure, and a second reconciliation produced 0 new errors (idempotent).
* The API answers `503 model_unavailable` for a series that has stored forecasts, because nothing is approved — the intended fail-closed behaviour.
* The retraining flow returned `blocked_data_quality` (feature missingness 0.296 > the 0.2 default: 29.6% of series have no price yet). Demand drift PSI 0.030, price PSI 0.011 (below the 0.25 alert level) after fixing a PSI defect found on this data (below).
* Model inference: p50 48.7 ms / p95 50.2 ms for 500 series × 28 horizons (14,000 rows; 15 repetitions after warm-up; one laptop run — not a rigorous benchmark).

## 16. Error Analysis
[`docs/error_analysis.md`](docs/error_analysis.md) — code and tables implemented; findings are written only from the generated report.

## 17. Prediction Intervals
Quantile LightGBM (0.1/0.5/0.9), conformalised calibration from a strictly earlier window, pinball/coverage/width/crossing by horizon and segment (`retailfc backtest uncertainty`). No coverage guarantee claimed.

## 18. Interpretability
Permutation importance (feature and group level) and LightGBM contribution values on a bounded sample at the last validation origin (`retailfc backtest interpret`); predictive, not causal; correlated lags share importance.

## 19. Forecast Storage and Serving
Idempotent transactional batch writes, deterministic batch ids, conflict-safe reruns, exact model/dataset/feature versions on every row. The API **retrieves** precomputed forecasts of the approved version, behind a fail-closed API-key check (`X-API-Key`, `RETAILFC_API_KEYS`) — [`docs/operations.md`](docs/operations.md).

## 20. Forecast Monitoring
Performance (matured forecasts only), drift, data quality kept separate; PostgreSQL detail + bounded Prometheus gauges; Grafana dashboards and alert rules provisioned from files.

## 21. Retraining Strategy
Decision flow with data-quality gate, cooldown, minimum new data, comparable matured-forecast degradation, governance age, drift → *investigate* only, advisory-lock protection and a full audit trail — [`docs/retraining_policy.md`](docs/retraining_policy.md).

## 22. Reproducibility and Quickstart
```bash
# 1. environment (Python 3.13)
python -m venv .venv && .venv/Scripts/activate        # or source .venv/bin/activate
pip install -r requirements.lock && pip install --no-deps -e ".[dev]"
python -m venv .venv-dbt && .venv-dbt/Scripts/pip install "dbt-core>=1.9" "dbt-postgres>=1.9"   # dbt isolated

# 2. synthetic smoke run (proves the plumbing, never a benchmark)
cp .env.example .env   # set the passwords and RETAILFC_API_KEYS
docker compose up -d postgres
bash scripts/demo_fixture.sh

# 3. real data: download online_retail_II.xlsx from UCI yourself, then (keep WORK outside synced folders)
XLSX=/path/online_retail_II.xlsx WORK=/path/work DBT_BIN=/path/to/dbt bash scripts/run_retail_case_study.sh
```
Full stack: `docker compose up -d --build`. Tests: `pytest -m "not integration and not slow"` (no DB), `RETAILFC_TEST_DSN=… DBT_BIN=… pytest` (everything).
Dependencies are locked in `requirements.lock` (`uv pip compile --universal`). CI (`.github/workflows/ci.yml`): ruff, unit tests, integration + dbt with a Postgres service, docker build/compose-config check; no data or credentials.

## 23. Trade-offs
Direct multi-horizon (no error accumulation, bigger training frame) vs recursive; stored-forecast API (cheap, safe, stale) vs on-demand inference; one global model (shares strength across series) vs per-series fits; WAPE (interpretable, volume-weighted) vs scale-free metrics; Prefect-only orchestration (one system) vs ecosystem breadth; training on a 3,000-series subset (compute) vs all 9,414.

## 24. Failure Cases
Matrix of explicit failure/fallback behaviours with the test that covers each: [`docs/limitations.md`](docs/limitations.md).

## 25. What Failed and What I Changed
* The first Docker Compose start failed: the Postgres health check passed against the image's temporary init server, so `migrate` got "connection refused" → health check now uses TCP.
* A replay design hole: a model trained on later data could forecast an earlier origin and be "scored" → `ModelAfterOriginError`, and the lifecycle test trains at the replay origin.
* A second challenger trained on all data would have been compared in-sample with the incumbent → the pipeline picks an evaluation origin that is out-of-sample for both or refuses (`insufficient_evaluation_data`).
* The first end-to-end demo failed on a Windows folder lock while publishing a raw version (sync client) → retry with backoff + cleanup of read-only temp folders.
* Conformalised intervals gave identical raw/calibrated intervals on the synthetic fixture: ≥20% of conformity scores sat on the clipped zero boundary, so the offset was 0 — documented, offsets are logged.
* The adapter's report showed a real product code (`47503J `, trailing blank in the source) being dropped as "non-product" → codes are stripped before matching, with a test.
* The drift indicator (PSI) returned exactly 0.0 on this data: with ~90% zeros, quantile bins collapse into one bin and any shift is invisible (reproduced with a synthetic shift). Fixed by treating zero as its own class; regression tests added.
* The originally planned dataset could not be obtained, so the project was rebuilt around a different public dataset: dataset-specific column names and validations were removed rather than faked.

## 26. Known Limitations
See [`docs/limitations.md`](docs/limitations.md). Biggest: 91.7% zeros and wholesale-style lumpy demand; channels are constructed (UK vs other); events are my assumption; prices come from the same transactions; very few independent time blocks;
the API serves batch forecasts only; WRMSSE and cost/inventory scenario analysis are not implemented.

## 27. Final Answer
*Were baselines beaten? Which segments/horizons benefited? What failed? Cost? Availability assumptions? Reproducibility? Limits?*
* **Baselines beaten on Online Retail II:** yes for naive, seasonal naive and the mean baseline on pooled WAPE/MAE (validation 3/3 origins and the locked holdout: −20% vs seasonal naive, −27% vs naive, −9% vs the mean baseline on the holdout). **No** against the trivial all-zero forecast on WAPE/MAE, and **no** against the mean baseline on RMSE.
* **Segments/horizons:** benefit in high-volume and UK-channel series; harm in low-volume and inactive series. Horizon effects cannot be separated from weekday effects (all origins are Fridays).
* **What failed:** §25 plus the findings above — bulk single-day orders (2,000–4,800 units) dominate errors; the feature groups added no demonstrable value; holiday effects cannot be tested (one holiday day in all evaluated windows).
* **Computational cost:** on a free machine, the holdout stage fit LightGBM in 4.8 s (840,000 rows) and predicted 263,592 forecasts in 0.4 s; SARIMA took 39 s for 120 series. One laptop run, not a benchmark; no monetary cost is estimated.
* **Availability assumptions:** §9 and `docs/data_availability.md`.
* **Reproducibility status:** code, config, locks, migrations, dbt, CI and Compose are in place; the synthetic pipeline and tests ran end to end; the real-data run is reproducible with the script in §22.
* **Limits of the evidence:** one retailer, intermittent B2B-like demand, derived channels/events/prices, few time blocks; this is **not** a production-readiness claim.

## Verification and readiness
Evidence-based review of every category (PASS / PARTIAL / FAIL), critical requirements, README-claim traceability and exact remaining steps: [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md). Two-minute overview: [`RECRUITER_SUMMARY.md`](RECRUITER_SUMMARY.md).
