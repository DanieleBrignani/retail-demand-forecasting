# Audit — integer overflow of daily units (int16), found by an external review after publication

## The defect
`load_panel_from_raw()` read `sales_daily.csv` with `dtype="int16"` (and the warehouse loader and its staging read-back did the same). Two series-days of the real data exceed 32,767 units and **wrapped around modulo 65,536**:

| series | date | true units | loaded as |
|---|---|---|---|
| `RETAIL_1_23166` UK_1 | 2011-01-18 (day 414) | 74,215 | 8,679 |
| `RETAIL_1_23843` UK_1 | 2011-12-09 (day 739, last observed day) | 80,995 | 15,459 |

Total: **131,072 units (2 × 65,536) of 11,402,383 (1.15%) were removed** from every panel, from the PostgreSQL `raw.sales_long` table, from the replayed actuals (the day-739 cell) and therefore from every real-data number published up to commit `5eb637c`.
The source data were intact; the code altered them. I reproduced it before changing anything (panel total 11,271,311 vs CSV total 11,402,383).

## Why it was not caught
No test used a daily quantity above 32,767 (the synthetic data are Poisson with small means, the adapter tests use quantities ≤ 10), the dbt checks verify structure, not totals, and the staging read-back used the same narrow dtype as the loader, so "panel from staging equals panel from raw files" was true but wrong.
Lesson: a conservation check (total units source = panel = warehouse) and tests at the extremes of the data, not at its typical values.

## Fix
`panel.read_sales_csv` parses as int64, checks the range, and casts to int32 only if nothing changes; `build_panel` asserts the cast conserves the sum; the warehouse loader and the staging read-back use the same function / `SALES_DTYPE`.
Regression tests (they fail on the old code with the same 15,459 value): `tests/test_price_availability.py::test_large_daily_quantities_survive_…`, `…refuses_values_it_cannot_hold`, `…weekly_aggregation_keeps_large_quantities`, and `tests/test_dbt_warehouse.py::test_warehouse_load_and_staging_panel_keep_large_quantities`.

## What was affected and what was done
| item | status |
|---|---|
| every real-data result before this fix (daily validation/holdout, the daily comparison, weekly v1, weekly v2, the operational replay numbers, the DB-side metrics) | **superseded**; archived outputs kept under `*-int16bug-*` folders and the historical reports carry a banner |
| PostgreSQL development database | dropped and rebuilt from scratch with the corrected loader (`raw.sales_long`: max 80,995, total 11,402,383 = source); the earlier daily batches/approvals were derived state and are not recreated — the daily line is documented as historical |
| weekly v2 backtest, daily comparison, selection, registration, approval, replay, reconciliation | **re-executed** on the corrected panel; new generated reports replace the numbers in the documents |
| selection outcome | see `results_weekly_v2.md` (generated); the external review's independent recomputation had the 4-week mean selected as well |
| the earlier approval of `retailfc_weekly` v1/v2 | their evidence came from the corrupted panel; they are revoked with a reason, and a new version is registered and approved from the corrected evidence |
