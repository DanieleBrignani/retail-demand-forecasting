# Limitations and failure behaviour

## Current state in one paragraph
Revised historical evaluation on already-inspected data (no independent holdout); one retailer, about two years (two full Decembers and the run-up of a third); weekly line served with a **baseline** (4-week mean) because LightGBM failed the MASE and segment guards; intervals experimental and not served;
cold-start products quantified but not evaluated; gross sales (returns not netted); zeros are zero *recorded* sales; CI green on fixtures only (not the real-data pipeline); no load, failure-injection or production-traffic evidence. The daily LightGBM line is invalid as evidence (price look-ahead `fv1`) and its approval is revoked.
Sections below mix current rules and historical daily observations; items marked *(historical)* are superseded by [`audit_price_leakage.md`](audit_price_leakage.md) and [`results_daily_fv2.md`](results_daily_fv2.md).

## Failure-mode matrix (each case has defined behaviour; nothing is silently turned into a normal-looking forecast)
| Case | Behaviour | Evidence |
|---|---|---|
| Missing prices | Left missing (never 0, never back-filled): `price_origin = NaN`, `price_missing = 1`; LightGBM uses native missing handling | `test_prices_are_never_zero_filled_or_backfilled` |
| Stale prices | Last completed-block price carried forward; `price_age_weeks` (completed blocks) records staleness | same test (stale-gap series) |
| Price block not yet complete | Origin inside a block: that block's price is **unavailable** (NaN, `price_missing=1`), never partial or future | `tests/test_price_availability.py` |
| Product not yet observed at the origin | Not part of the universe: no training rows, not scored, forecast absent; if forced, unseen identifier `-1` | `tests/test_price_availability.py`, `tests/test_weekly.py` |
| Weekly period with < 7 daily actuals | Not scored (never scored on partial data); surfaces as missing actuals coverage | `tests/test_weekly_lifecycle.py` |
| Approval found invalid | `registry revoke` (reason required, audit kept); API answers `503 model_unavailable` for that line | `tests/test_weekly_lifecycle.py` |
| Weekly horizon > 4 / daily horizon > 28 | API `422 horizon_out_of_range` | `tests/test_weekly_lifecycle.py` |
| Too little history to train the weekly model | `ValueError("not enough history …")` | `tests/test_weekly.py` |
| Insufficient history | Lags/rolling features NaN (no error); baselines flag `short_history`; seasonal naive falls back to naive with < 7 days | `test_short_history_gives_nan_not_error`, `test_baseline_fallback_flags` |
| All-zero history | Forecast 0 with `fallback_used='all_zero_history'` (model, baselines) / `'sarima_all_zero_history'` | `test_all_zero_history_series_forecasts_zero_with_flag`, SARIMA fallback test |
| Unseen identifiers | Encoder maps to −1 → seasonal-naive fallback `unseen_series` | `test_unseen_series_falls_back_to_seasonal_naive_and_is_flagged` |
| Missing calendar coverage | `CalendarCoverageError` — no forecast | `test_missing_calendar_coverage_is_refused` |
| Source schema change | Ingestion refuses missing required columns (warns on extras); dbt `assert_source_schema_unchanged` fails on added/removed raw columns | `test_source_schema_change_detected…`; dbt test (positive run only — not negative-controlled) |
| Duplicate rows | Ingestion refuses; dbt key tests fail (negative control for the price duplicate) | `test_duplicate_rows_detected`, `test_dbt_tests_fail_on_injected_problems` |
| Missing dates / day columns | Ingestion refuses; dbt calendar integrity fails (negative control) | same tests |
| Negative / invalid sales, invalid prices | Ingestion refuses; dbt fails (negative control) | `test_negative_and_missing_sales_detected`, `test_invalid_prices_and_orphans_detected` |
| SARIMA fit failure | Seasonal-naive forecast, `fallback_used='sarima_fit_failed'`, recorded in `sarima_fits.csv`; series never dropped | `test_sarima_fit_exception_falls_back…` |
| Non-finite model output | Seasonal-naive fallback `nonfinite_model_output` | `test_nonfinite_model_output_uses_flagged_fallback` |
| Negative raw forecast | Clipped at 0, counted (`negative_raw`); storage rejects negatives | `test_negative_raw_model_output_is_clipped_and_counted`, `test_negative_forecasts_rejected` |
| Model artifact unavailable / non-explicit version | `ModelUnavailable`; only integer versions load | `test_model_must_be_selected_by_explicit_version` |
| Model trained after the forecast origin | `ModelAfterOriginError` (prevents leaky replay) | `test_replay_with_model_trained_after_origin_is_refused` |
| Database unavailable | API: 503 `database_unavailable`, readiness fails, liveness stays up | `test_database_unavailable_is_explicit` |
| No approved model / no batch | API 503 `model_unavailable` / `no_forecast_batch` | `test_model_unavailable_when_nothing_approved`, `…without_forecast_batch…` |
| Unknown series | API 404 `unknown_series` | `test_unknown_series_is_404…` |
| Missing actuals | Forecasts without an actual are never scored; large unscored share blocks retraining | `test_reconcile_scores_only_matched_actuals…`, retraining tests |
| Conflicting actuals | Recorded in `actuals_conflicts`, never overwritten; blocks retraining | `test_actuals_idempotent_conflict_detected…` |
| Rerun of a forecast batch | Same content → no-op; different content → `ForecastConflict`; new `run_label` coexists | `test_write_batch_is_idempotent_and_conflict_safe` |
| Concurrent training/promotion | PostgreSQL advisory lock → second run `skipped_concurrent` | `test_concurrent_training_is_refused` |

## Dataset limitations (UCI Online Retail II)
* **No store, region or category dimension:** the source has none, so those hypotheses cannot be tested; results are not comparable with other retail benchmarks and WRMSSE is not computed.
* **Extreme intermittency:** 91.7% of series-days are zero (1,039 of 9,414 series never sell), so WAPE is dominated by a minority of fast-moving codes and zero-heavy metrics (sMAPE with both-zero → 0) are optimistic.
* **Demand ≠ sales ≠ orders:** wholesale-style bulk orders create lumpy demand; cancellations/returns were excluded rather than netted; unobserved stock-outs remain unobserved.
* **Structural zeros:** non-trading days (Saturdays, closures) are zero for all series; holiday events are my assumption and only partly explain closures.
* **Prices come from the same transactions** (median of the 7-day block) and are absent in blocks with no sales. Since `fv2` a block price is visible only after the block has ended; they are never back-filled and must not be read as a planned/listed price.
* **Zero forecast is unbeaten on WAPE/MAE:** with ~90% zero series-days every model had pooled WAPE above 1. WAPE/MAE reward predicting nothing, sMAPE (both-zero → 0) rewards exact zeros, so the discriminating metrics here are the *ranking among non-trivial models* and RMSE.
* **Bulk orders:** single-day orders of 2,000–4,800 units dominate the largest errors and cannot be predicted from history features.
* *(historical)* The daily model v1 failed its promotion rules (bias −45%); v2 later passed only after a post-hoc change of the bias gate and rested on leaking price features, so its approval is revoked. The weekly line uses a pre-registered mechanical rule instead.
* **Structural feature missingness:** 29.6% of series have no price at the origin (never sold), above the 0.2 default of the data-quality gate, so the retraining flow reports `blocked_data_quality` on this sparse panel; the threshold is a policy choice not tuned for sparse panels.
* **Drift on zero-inflated data:** the original PSI was blind (always 0); the corrected version treats zero as a class. PSI alert thresholds (0.25) have not been calibrated on real operating history.
* **Weekday confounding:** all evaluation origins are Fridays, so horizon effects cannot be separated from weekday effects; WAPE is undefined for horizons that land on non-trading Saturdays.
* **Events cannot be assessed:** only one holiday day falls inside the four evaluated windows.
* **One holdout window** in the Nov–Dec run-up to Christmas (higher demand level than the validation windows); there is no estimate of its variability.
* The two "channels" are a construction (UK vs everything else), so channel-level conclusions are about those groupings only.
* *(historical)* v1 trained on a seeded 3,000-series subset; the current daily config trains on all eligible series.

## Known limitations of the design
* **Demand ≠ sales.** Zero sales may be stock-outs or inactivity; the data contain no inventory. First positive sale is not a launch date. Price and event effects are associations, not causal effects.
* **Advance availability of holiday events** is an assumption: they are a fixed public calendar that I added; the source data contain no event information.
* **Price scenario.** The primary scenario uses prices of completed blocks only (`fv2`); the earlier assumption that the current block's price was known was wrong. The "planned prices" scenario is optimistic by construction.
* **Stored-forecast API.** `/forecast` serves precomputed batch forecasts; it cannot answer for a series/horizon/origin that was not batch-generated and it is only as fresh as the last batch. Latency numbers therefore describe storage retrieval, not model inference.
* **Sample size of origins.** Three validation windows plus one holdout give very few independent time blocks; origin-to-origin variation is reported, but no significance is claimed (series, horizons and overlapping targets are dependent).
* **WAPE is volume-weighted;** low-volume intermittent series barely move it. Per-series-mean and aggregated metrics are provided; sMAPE with the both-zero→0 convention is optimistic on intermittent data. **WRMSSE is not implemented.**
* **Prediction intervals** are CQR-calibrated on one earlier window; no coverage guarantee under temporal dependence is claimed. On zero-inflated demand the clipped lower bound and boundary scores can leave the offsets at 0 (observed in the synthetic-fixture run).
* **Historical replay is a simulation.** Actuals arrive on a replay clock; real production latencies, late revisions and operational incidents are not represented. Monitoring/alert thresholds are starting points, not tuned.
* **Statistical-model scope.** SARIMA covers a restricted subset with a 3-candidate order search; conclusions about "ARIMA vs LightGBM" apply only to that subset.
* **Not done:** inventory/asymmetric-cost scenario analysis, monetary cost estimates, deep-learning comparison, hierarchical reconciliation.
* **Environment caveats found while building:** Python 3.13 (3.14 lacks reliable wheels for parts of the stack); dbt must live in a separate environment (dependency conflicts); the repository is inside a OneDrive folder, which makes file-heavy operations (venvs, MLflow artifacts) slow and can interfere with sync — keep `data/` and `artifacts/` outside synced folders for real runs.
