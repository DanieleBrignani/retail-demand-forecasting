# Data contract — Online Retail II adaptation, information boundaries, universe policy

Facts about the source come from `adapter_report.json` (written by `retailfc.adapters.online_retail`) for dataset version `retail-369d316281ea`; rules are enforced by code and tests named below.

## 1. Target and what it is not
* **Target = gross positive recorded sales**: sum of `Quantity` per product code, channel and day (weekly: per 7-day block) over invoice lines that are not cancellations, with `Quantity > 0` and `Price > 0`.
* It is **not unconstrained demand**: the export has no stock, assortment, availability or lost-sales information, so a zero can be zero demand, a stock-out, a product not on offer, or a day the shop did not record.
* Zero values are described as **zero recorded sales under the assumption that the export is complete**. They are not proven closures and not proven zero demand. Missing source coverage and unknown assortment status are unobservable here;
  insufficient history is observable and reported (history-length segments, `no_channel_history` fallback flag).

## 2. Cleaning rules (counts in `adapter_report.json`, tests in `tests/test_online_retail_adapter.py`)
| rule | treatment |
|---|---|
| Invoice starting with `C` (cancellation) | excluded, **not netted** against the original sale (the original invoice is not linkable in the export). Returns therefore do not reduce recorded sales; this makes the target a *gross* quantity. |
| `Quantity <= 0` | excluded |
| `Price` missing or `<= 0` | excluded (a zero price is never used as "unavailable") |
| non-product stock codes (postage, fees, adjustments, bank charges, vouchers, …) | excluded by the pattern `^\d{5}[A-Za-z]{0,2}$` (after trimming blanks); the list seen is in the report |
| missing identifiers | a row with a missing/non-positive `Quantity` or `Price`, a missing `InvoiceDate` or `Country`, or a `StockCode` that is not a product code is excluded and counted (`excluded_*` fields); a missing `Customer ID` is **not** an exclusion reason (guest orders are legitimate). `Customer ID` never leaves the adapter: no customer identifier is stored or published. |
| duplicates | **no de-duplication.** Two identical lines (same invoice, product, quantity, time, price) can be legitimate repeated scans/lines; a rule that removes them would delete real sales. The count of exact-duplicate lines is not used by any rule. |
| channels | `UK_1` (Country = United Kingdom) and `INT_1` (all other countries): **sales channels, not stores**. No physical stores, regions or categories exist (`RETAIL_1` / `RETAIL` are placeholders). |

## 3. Price availability (information boundary) — policy `completed-block prices`, feature version `fv2`
* The price of a 7-day block is the **median `Price` over all the block's transaction lines**; it is a transaction-derived price, **not** an externally observed list price.
* A block price therefore becomes known only **once the block has ended**. `Panel.as_of(origin)` exposes the prices of *completed* blocks only (a block is complete at origin *o* iff *o* is its last day or later); the origin's own block is excluded otherwise.
  Features carry the latest completed-block price forward, **never backfill**, keep missing as `NaN` (`price_missing = 1`), and report `price_age_weeks` in completed blocks. A zero price never stands for "unavailable".
* The rule applies identically in historical training (`build_training_frame` uses `as_of` for every training origin) and at serving/forecast time.
* Warehouse: `stg_prices.sell_price` is a block summary; `fct_sales_feature_ready` exposes it as `block_price` with `block_price_known_from_day_num` so no consumer can mistake it for a day-level price.
* Before this fix (feature version `fv1`) the origin's own block price, including future days of the block, was exposed: see [`audit_price_leakage.md`](audit_price_leakage.md) for the reproduction and the experiments affected.
* Tests: `tests/test_price_availability.py` builds hand-made transaction tables, runs the real adapter → ingestion → panel → features, changes only future information and compares earlier features by business key.

## 4. Series universe at an origin (cold-start policy)
* No external product catalogue exists and none is invented. A product belongs to the universe at origin *o* iff it has at least one recorded sale (any channel) on or before *o* (`Panel.eligible_at`).
  Both channels of a known product are eligible; a channel without a sale yet is scored with an explicit `no_channel_history` fallback flag.
* Encoder vocabularies (`fit_encoders`) are fitted on the universe at the training cutoff; products launched later map to the unseen code `-1` (documented cold start).
* Training rows at origin *o′* exist only for products observed by *o′*; evaluation and forecast generation use the same universe (identical for every compared model).
* The adapter still writes a panel containing all products of the whole file (it must, to build matrices); the **universe is applied at model time**, so ingestion needs no knowledge of the future and the stored dataset version is unchanged.
* Evaluation of genuinely unseen products is **not** performed; `universe_coverage.csv` reports, per origin, how many series and what share of the realised target-window demand belong to not-yet-launched products.

## 5. Calendar
Daily: the dataset's calendar (first day 2009-12-01, Tuesday; 739 observed days). Weekly: 7-day blocks starting on the first observed day (Tue–Mon), not ISO weeks; 105 complete blocks, the 106th (4 days) is dropped. Weekly and daily metrics are not comparable.
Holiday events are England & Wales public holidays from the `holidays` package (an assumption, not source data).
