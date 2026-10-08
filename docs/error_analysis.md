> **Two parts.** Part A (current): weekly line, generated tables in [`results_weekly_v2.md`](results_weekly_v2.md). Part B (historical, below the rule): daily LightGBM set D `fv1` — superseded, kept verbatim.

# Part A — weekly error analysis (descriptive; no significance claimed)
* **Where the global model helps and hurts** (WAPE by past-volume segment, `results_weekly_v2.md`): LightGBM beats the 4-week mean on high-volume series (0.742 vs 0.800), on smooth and intermittent series and in the UK channel, but loses on low-volume (1.504 vs 1.162) and mid-volume (1.175 vs 1.096) series.
  That split is why pooled WAPE and series-level MASE disagree and why the segment guard rejected it.
* **Inactive series** (no sale in the previous 26 blocks) are forecast as exactly 0 by `mean4` (WAPE 1.000 by definition); LightGBM scores 2.22 there. Zero forecasts are only "right" under absolute-error metrics; they say nothing about stock decisions.
* **Aggregation hides error:** at channel/total level WAPE is 0.22/0.17 for `mean4` and 0.20/0.16 for LightGBM, versus 0.85 at product × channel level — cancellation across series, not accuracy of individual products.
* **Horizon:** WAPE by horizon (blocks 1–4) is in the report; the differences between horizons are small relative to origin-to-origin variation.
* **Origins:** `mean4` WAPE ranges from 0.693 (origin 101) to 0.959 (origin 81); its bias per origin ranges from −31% (origin 93) to +13% (origin 77), i.e. the pooled −7% hides opposing errors of both signs — the reason the selection rule also checks bias per origin.
* **Cold start:** 0.6%–7.2% of realised target-window units belong to products first sold after the origin; they are neither forecast nor scored (table in the report).
* **Intervals:** see the report; aggregate coverage is inflated by inactive series covered by zero-width intervals at 0.

---
# Part B — historical (daily LightGBM, superseded)
# Error analysis — UCI Online Retail II (validation windows, selected LightGBM set D unless noted)

Sources: [`docs/results_online_retail.md`](results_online_retail.md) (verbatim report) and the CSV tables written by `retailfc report build` to `<WORK>/report/error_analysis/`
(`analysis.error_tables`). Everything is **descriptive**: observations are correlated across series, horizons and windows, so no significance is claimed.

## Findings
1. **Largest errors are single-day bulk orders.** The eight largest absolute errors are actuals of 2,000–4,850 units on one day for one product-channel while the forecast is below 45 units
   (e.g. `RETAIL_1_84077`, UK channel, actual 4,848 vs forecast 9.4). A handful of such events dominates RMSE and the high-volume WAPE; they are not predictable from the history features used here.
2. **Pooled WAPE is above 1 for every model**, i.e. none beats an all-zero forecast (`scripts/zero_forecast_reference.py`: WAPE 1.000, MAE 2.048 on validation, 2.650 on the holdout). Persistent bias:
   4,086 of 9,414 series (43%) are over-forecast at *every* origin and 698 under-forecast at every origin for set D (seasonal naive: 60 and 848), while total bias is negative (−0.48 of demand) — the model under-predicts the few
   very large days and over-predicts many small ones.
3. **Segments.** High-volume and UK-channel series are where LightGBM beats the baselines; low-volume, inactive and "intermittent" series are where it loses to the simple mean (see `docs/model_card.md`).
   For series with no sales in the trailing window ("inactive"/"no_sales") the baselines output exactly 0 and score WAPE 1.00; LightGBM outputs small positive values and scores 1.2–1.4.
4. **Horizon.** WAPE by horizon is *not* a clean degradation curve. For `lgbm_D` (validation): h=7 1.151, h=14 1.136, h=21 1.056, h=28 1.078. The curve is dominated by the **day of the week of the target**:
   every origin (627, 655, 683, 711) is a Friday, so h=1, 8, 15, 22 are Saturdays when the retailer sells nothing — pooled WAPE is **undefined** there (Σ|actual| = 0; reported as undefined, not 0) and the other horizons differ mainly through weekday and spikes.
   With three origins on the same weekday, horizon effects cannot be separated from weekday effects.
5. **Holiday period.** The only event day inside any evaluated window is 2011-08-29 (Late Summer Bank Holiday, validation window 1): 9,414 observations, all actuals 0 (no sales — presumably closed), WAPE undefined,
   MAE 0.135 (the model still forecasts a small positive value). No conclusion about holiday effects is possible (see `docs/limitations.md`).
6. **Price-change periods (descriptive; grouping uses the target week's price).** "Unknown" (no price listed in the target week — which here means no sales that week) covers 79% of observations and has WAPE 2.05;
   weeks with a changed price: WAPE 1.12; unchanged: 0.97. Because prices exist only in weeks with sales, this split mirrors sales activity and must not be read as a price effect.
7. **History length and fallbacks.** 15% of the evaluated rows (120,624) are series with no positive sale before the origin; the model returns 0 with `fallback_used='all_zero_history'` (flagged, WAPE 1.000 by construction).
   Series with > 90 days of history: WAPE 1.123; 29–90 days: 1.095; ≤ 28 days: 1.055.
8. **Model-fit failures (SARIMA).** `sarima_fits.csv` (validation) records every fit and fallback; the subset comparison is in `docs/results_online_retail.md`. SARIMA was no better than the naive forecasts in validation (WAPE 1.073 vs 1.044) and tied with LightGBM on the holdout subset (1.015 vs 1.012).

## Questions and the table that answers each
| Question | Output (per model: `seasonal_naive`, `lgbm_*`) | Code |
|---|---|---|
| Largest absolute errors | `largest_errors_<model>.csv` | `analysis.error_tables` |
| Persistent over-/under-forecasting | counts in the report | same |
| Horizon degradation | `horizon_degradation_<model>.csv`, `*_wape_by_horizon.csv/png` | same, `report.py` |
| Low-volume / intermittent series | `by_volume_segment_<model>.csv`, `by_intermittency_segment_<model>.csv` | same |
| Event periods | `event_periods_<model>.csv` | same |
| Price-change periods | `price_change_periods_<model>.csv` | same |
| Insufficient-history behaviour | `history_length_<model>.csv` | same |
| Model-fit failures & fallback performance | `fallback_performance_<model>.csv`, `sarima_fits.csv` | same, `models/stats.py` |
| Matched model comparison / origin variability | paired table in the report | `analysis.paired_vs_reference` |

## Statistical caution
Comparisons use identical origins, series and actuals. Reported uncertainty is the **spread across origins** (3 validation origins + 1 holdout) and descriptive win-rates; no p-values or confidence intervals are produced
because rows are dependent. A resampling interval would have to resample time blocks (e.g. weeks), not rows or series.

## What I would investigate next (not done)
Handling of bulk orders (separate B2B/wholesale lines or capping), a hurdle model (probability of any sale × size), metrics suited to intermittent demand (scaled/pinball losses; an all-zero baseline in the protocol),
more origins on different weekdays, and a real holiday/promotion calendar.
