# Data availability and information boundaries

## Time concepts
| Term | Meaning | Where stored |
|---|---|---|
| **Forecast origin** `o` | Last date with sales information available to the forecasting process | `forecast_batches.forecast_origin(_day)` |
| **Target date** | Date being predicted, `o + h`, `h ∈ 1..28` | `forecasts.target_date`, `horizon` |
| **Generation timestamp** | Wall-clock time the batch was produced | `forecasts.generation_time` |
| **Data availability timestamp** | When the origin's data became available. The dataset is historical, so this is not observed; for replay it is the replay ingestion time | `forecast_batches.data_available_at` |

The dataset is historical. Operational behaviour (arrival of actuals, monitoring) is demonstrated by a **historical replay**: observed days are released
chronologically (`retailfc actuals replay`, `source='historical_replay'`, batches marked `mode='historical_replay'`). Replay never produces days beyond the last observed
sales day, and a forecast at origin `o` may only use a model trained with data ≤ `o` (enforced: `ModelAfterOriginError`).

## What is known at origin `o`
| Source | Rule | Test |
|---|---|---|
| Sales | `sales[:, :o]` only; reads go through `Panel.as_of(o)` | `test_future_sales_and_prices_mutation_invariance`, baselines mutation tests |
| Rolling windows / lags | end at `o`; `lag_k = y_{o−k}` | `test_origin_relative_lags_and_rolling_boundaries` |
| Target-aligned seasonal lag | `y_{t−7·⌈h/7⌉}` — same weekday as the target but ≤ `o` | `test_target_aligned_seasonal_lag_never_reads_after_origin` |
| Calendar (weekday, month, day) | known in advance | missing coverage → `CalendarCoverageError` |
| Holiday events | England & Wales public holidays (`holidays` package), a fixed public calendar assumed known ≥ 28 days ahead — **not source data** | documented |
| Prices (primary, "observed", `fv2`) | **completed 7-day blocks only** (a block price is the median of the whole block, so it is known only after the block ends); last completed price carried forward; **never back-filled**; age in completed blocks (`price_age_weeks`) and missingness (`price_missing`) recorded; transaction-derived, not a list price | `tests/test_price_availability.py`, `test_prices_are_never_zero_filled_or_backfilled` |
| Prices ("planned" scenario) | target-week price used; assumes the price plan is known; **separate result set** | `test_planned_price_scenario_is_the_only_one_reading_future_prices` |

## Weekly forecasts
Weekly batches (`granularity='week'`) use the dataset's 7-day blocks (Tue–Mon). The forecast origin is the last day of the last complete block; `horizon` counts blocks (`horizon_unit='week'`, 1–4) and every row stores `target_period_start/end`. A weekly actual is the sum of the seven daily actuals and a period is scored only when all seven exist. Weekly and daily batches are never mixed in one batch (checked) and have separate registry lines.

## Learned preprocessing
Identifier and event vocabularies are fitted on data up to the training cut-off only, and identifiers only for products observed by the cut-off (`fit_encoders(panel, cutoff_day)`); unseen identifiers map to −1 and trigger the
`unseen_series` fallback at prediction time. Volume-segment edges are fitted at the first validation origin; segments are recomputed at every origin from pre-origin data.

## Dataset provenance (UCI Online Retail II)
Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii, CC BY 4.0, downloaded with the user's explicit permission. `online_retail_II.xlsx` SHA-256
`bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` (zip: 45,622,418 bytes). `retailfc adapt-online-retail` writes `adapter_report.json` next to the canonical CSVs.

| Item | Rule / assumption |
|---|---|
| Sales | sum of `Quantity` per product code, channel and calendar day over non-cancelled lines with quantity > 0 and price > 0; a sale on day d is available at the end of day d (timestamps exist but there is no reporting lag to model) |
| Product codes | `^\d{5}[A-Za-z]{0,2}$` after stripping blanks; others (`POST`, `ADJUST`, `BANK CHARGES`, …) are excluded and listed in the report |
| Channels | `UK_1` = Country "United Kingdom", `INT_1` = every other country; **not physical stores**; one department/category because the source has none |
| Panel | complete product × channel grid; products never sold in a channel are all-zero series. **Universe at an origin = products with a recorded sale on or before it** (`Panel.eligible_at`); later products are cold start (quantified per origin in `universe_coverage.csv`, not scored) |
| Zero days | days without any transaction are **zero recorded sales** under the assumption that the export is complete — not proven closures and not proven zero demand; missing source coverage and unknown assortment status are not observable here |
| Prices | median of the **7-day block's** sale lines, rounded to cents; blocks without sales have **no** price row. Known only after the block has ended (see `data_contract.md` §3). Prices come from the same transactions as the sales — weaker than a listed price; price features must not be read as an operational price signal |
| Week key | `week_id` = 1 + the index of the 7-day block starting at the first date (plain blocks, not a retail fiscal week) |

## Raw versioning
`retailfc ingest` records source, file names, SHA-256 checksums, ingestion timestamp, schema version (`retail-raw-v1`), observed date range, and a content-derived dataset version
(`retail-<sha256[:12]>`) in `manifest.json`. Raw directories are written once (atomic rename, retried on transient Windows locks) and made read-only; identical bytes → verified no-op; different bytes → a new version;
a stored version whose bytes no longer match its manifest is reported as corrupt and never overwritten. The source file itself is obtained by the user from UCI; nothing in the pipeline downloads data.
