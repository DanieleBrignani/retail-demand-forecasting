# Retail Demand Forecasting Platform — UCI Online Retail II

An executable, reproducible forecasting workflow with a deliberately conservative evaluation: immutable ingestion → PostgreSQL/dbt → leakage-safe features → rolling-origin backtests → mechanical model selection → MLflow registry with rule-based approval/revocation → append-only forecast storage → actuals replay and reconciliation → monitoring (Prometheus/Grafana) → FastAPI → Docker Compose.

> **Correction after an external review (2026-10-06).** Daily quantities were read as `int16`, which silently wrapped two real series-days (74,215 → 8,679 and 80,995 → 15,459; 131,072 units lost). The bug is fixed, covered by regression tests and **every real-data number below was regenerated** on the corrected data ([`docs/audit_int16_overflow.md`](docs/audit_int16_overflow.md)). The selection outcome did not change.
>
> **Status (read first).** Portfolio demonstration with a **modest, honestly reported result**. Not production-ready; no claim of model superiority. Every block of the dataset has been inspected, so all real-data results are
> **revised historical evaluations**, not independent holdouts. See [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) (PASS/PARTIAL/FAIL per requirement) and [`RECRUITER_SUMMARY.md`](RECRUITER_SUMMARY.md) (2-minute read).

## 1. Business question
*Can product-level weekly recorded sales be forecast more accurately and reliably than simple and intermittent-demand baselines, with reproducible model selection and a functioning forecast lifecycle?*
The target is **gross positive recorded sales** (units) per product × sales channel × 7-day block — not unconstrained demand: the data has no stock, assortment or lost-sales information, and zero means "zero recorded sales" under an assumed complete export ([`docs/data_contract.md`](docs/data_contract.md)).

## 2. Data
UCI *Online Retail II* (CC BY 4.0): 1,067,371 invoice lines of one UK online retailer, 2009-12-01 → 2011-12-09. Cancellations are excluded, not netted; non-product codes and non-positive quantity/price are excluded and counted; identical-looking lines are kept (they can be legitimate).
4,707 products × 2 channels (`UK_1` = United Kingdom, `INT_1` = all other countries; channels, not stores; no category hierarchy) = 9,414 series; 91.7% of series-days are zero (73% of the scored weekly observations).
`online_retail_II.xlsx` SHA-256 `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`. Full rules and counts: [`docs/data_contract.md`](docs/data_contract.md), `adapter_report.json`.

## 3. What was verified (current state, 2026-10-05)
| Area | Evidence |
|---|---|
| Information boundary from **source transactions** | `tests/test_price_availability.py` (adapter → ingestion → panel → features; future price/quantity/transaction changes; universe changes) |
| Defect found and fixed | price look-ahead: a future day-4 price of 100 changed an origin-day-1 feature 10 → 55; fixed with completed-block prices (`fv2`) — [`docs/audit_price_leakage.md`](docs/audit_price_leakage.md) |
| Origin-appropriate series universe | products observed by the origin only; cold start quantified, not scored — [`docs/data_contract.md`](docs/data_contract.md) §4 |
| Weekly study (main candidate experiment) | frozen protocol [`docs/weekly_protocol_v2.md`](docs/weekly_protocol_v2.md) + [`configs/weekly_v2.yaml`](configs/weekly_v2.yaml); generated report [`docs/results_weekly_v2.md`](docs/results_weekly_v2.md) |
| Lifecycle on real data | register → approve (rule) → forecast (weekly contract) → replay actuals → reconcile (idempotent) → monitor → API — [`docs/results_weekly_operations.md`](docs/results_weekly_operations.md) |
| Warehouse | `dbt build` on the corrected real data: PASS=35 in 61 s (the former 3-way join-count test took 61 min; replaced by an exhaustive key-uniqueness proof) |
| Unit conservation | total units source = panel = `raw.sales_long` = replayed actuals (checked: 11,402,383 and 894,740 for days 708–739); regression tests fail on the old int16 code |
| Tests | full suite with PostgreSQL + dbt (run `pytest -q` with `RETAILFC_TEST_DSN` and `DBT_BIN`); lint `ruff check src tests scripts monitoring` |
| Docker Compose | image rebuilt, stack recreated on a fresh volume, all services healthy, Prometheus targets up, Grafana provisioned, API probed (401/200/422/404/503) |
| CI | GitHub Actions: run #1 failed (pytest installed with `--no-deps`), fixed; run #2 (`f03c53b`) passed all four jobs — `lint`, `unit`, `integration` (PostgreSQL + dbt), `docker`. Fixtures only; the real-data pipeline is not run in CI |

## 4. Main results (all numbers generated from artifacts)
**Weekly, 10 rolling origins, eligible universe** ([`docs/results_weekly_v2.md`](docs/results_weekly_v2.md)): all-zero reference WAPE 1.000; 4-week mean **0.8465**; 13-week mean 0.8606; naive 0.9806; TSB 0.9806 (one extreme order inflates its forecasts); Croston/SBA 1.3303; seasonal naive (52 weeks) 1.3425;
global LightGBM **0.8251** (best WAPE and RMSE, and best at channel/total level) but **MASE 3.08 vs 2.34** and worse WAPE on low- and mid-volume series. **Selection (mechanical, thresholds not changed after results): the 4-week mean.** LightGBM was rejected by the MASE guard and the segment-degradation check.
Origin-to-origin WAPE varies (std ≈ 0.07–0.10). That spread is reported, but it does **not** by itself show that the LightGBM margin is noise (the two models are scored on the same origins) nor that it is real: no significance test was done and none is claimed.

**Daily comparison under the corrected pipeline** ([`docs/results_daily_fv2.md`](docs/results_daily_fv2.md)): no model beats the all-zero forecast on WAPE/MAE at daily granularity (best WAPE 1.08 vs 1.00); variants that remove bias raise WAPE. This is why the weekly line is the main experiment.
Daily and weekly metrics are **not comparable**.

**Intervals:** experimental, not served (coverage 0.915 all rows / 0.871 active-only vs 0.80 nominal; raw crossing 2.1%; the conformal step is degenerate on zero-heavy data).

## 5. Superseded and exploratory results (kept, not hidden)
* Daily LightGBM v1 (rejected, bias −45%) and v2 (approved after a **post-hoc** change of the bias gate from per-window to pooled): both used feature set D with `fv1` leaking prices. The v2 approval is **revoked**; it was never independent validation. Reports remain with a "historical/superseded" banner: [`docs/results_online_retail.md`](docs/results_online_retail.md), [`docs/results_online_retail_operations.md`](docs/results_online_retail_operations.md), full old README: [`docs/README_daily_study_historical.md`](docs/README_daily_study_historical.md).
* Weekly v1 (not a serving candidate; no universe policy): [`docs/results_weekly.md`](docs/results_weekly.md), its code is not part of the published history (v2 supersedes it).

## 6. Architecture
```
xlsx ─adapt─▶ canonical CSVs ─ingest/validate─▶ immutable versions ─▶ PostgreSQL raw.* ─dbt─▶ staging.* (+35 checks)
                                       └──▶ Panel.as_of(origin)  (completed-block prices, universe = products observed by origin)
Panel ─▶ daily features (fv2) / weekly features (wfv1) ─▶ baselines · Croston/SBA · TSB · LightGBM ─▶ rolling-origin backtests ─▶ manifest + parquet + CSV
selection.json (mechanical) ─▶ MLflow registration ─▶ rule approval / revocation ─▶ app.approved_models
batch forecast (granularity day|week, explicit horizon_unit and target period) ─▶ app.forecasts (append-only) ─▶ API (stored-forecast retrieval, API key)
actuals replay ─▶ app.actuals ─▶ reconcile (daily, or weekly = sum of 7 daily actuals) ─▶ app.forecast_errors ─▶ monitoring ─▶ Prometheus/Grafana
```
Shared library `src/retailfc/`; weekly study in `weekly.py`, weekly lifecycle in `weekly_lifecycle.py`; migrations `0001`–`0004` (0004 adds granularity, explicit periods, revocation).

## 7. Run it (three separate entry points)
```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.lock && pip install --no-deps -e .[dev]
cp .env.example .env                                  # then replace the placeholder secrets (git-ignored)
```
* **A. Synthetic smoke / integration (minutes; mechanics only, no claim about forecasting quality):** `bash scripts/demo_fixture.sh` (needs a PostgreSQL, see [`docs/operations.md`](docs/operations.md)).
* **B. Real-data scientific experiment:** download `online_retail_II.xlsx` from UCI; `retailfc adapt-online-retail …`, `retailfc ingest …` (see `scripts/run_retail_case_study.sh` for the daily study), then
  `retailfc weekly backtest --version <V> --config configs/weekly_v2.yaml` and `python scripts/build_weekly_report.py <exp_dir> docs/results_weekly_v2.md`. Keep the working folder outside synced folders.
* **C. Real-data operational replay:** `WORK=… V=… REVOKE_DAILY=1 bash scripts/run_weekly_operations.sh` (register → approve → forecast → replay → reconcile → monitor), then `docker compose up -d --wait`.
Five-minute demonstration: [`docs/demo_script.md`](docs/demo_script.md). Reproducibility: the manifest records dataset version, feature version, config SHA-256, seed, hardware/software and runtime; the weekly run was executed twice (see "Repetition" in the report). LightGBM is multi-threaded, so bitwise equality is not promised.

## 8. Limitations
No independent holdout; one dataset (one retailer, about two years: two full Decembers and the 2011 run-up); cold-start products not evaluated; gross sales (returns not netted); zeros are not proven zero demand; no decision-cost evaluation; intervals not served; replayed 2011 data (freshness alerts meaningless);
CI runs on fixtures only (not the real-data pipeline); no load testing or production traffic. More: [`docs/limitations.md`](docs/limitations.md), [`docs/model_card.md`](docs/model_card.md), [`docs/error_analysis.md`](docs/error_analysis.md), [`docs/operations.md`](docs/operations.md), [`docs/data_availability.md`](docs/data_availability.md).
