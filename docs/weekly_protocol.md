> **SUPERSEDED by [`weekly_protocol_v2.md`](weekly_protocol_v2.md).** Kept as the record of what was pre-registered for the first weekly run.

# Weekly-level forecasting — protocol, written BEFORE any weekly model was run

**Status of this document:** pre-registration. The daily study (docs/experiment_plan.md) is finished; its validation windows and holdout are consumed.
Nothing in this weekly study has a pristine holdout: I already know the daily results on the same data. The mitigation is that the recipe below is **fixed now and not tuned afterwards**
(no hyper-parameter search, no feature search, no threshold changes). Every number is reported whether or not it is favourable.

## 1. Why
Daily product×channel demand is ~92% zeros; WAPE/MAE are minimised by predicting zero, so the daily problem cannot show a useful model. Operational decisions (reorder, safety stock)
are taken at weekly granularity, where demand is much denser. This study asks: **at weekly granularity, does a global model beat simple and intermittent-demand baselines, and is it calibrated?**

## 2. Data and levels
* Source: the same ingested Online Retail II version (`retail-369d316281ea`), daily panel → **weeks** = the 7-day blocks of the panel calendar (Tue–Mon). The 106th block has 4 days and is dropped → **105 complete weeks**.
* L1 = product × channel (9,414 series); L2 = channel totals (2 series: UK_1, INT_1); L3 = grand total (1 series, sum of the two channels). L2/L3 forecasts are **sums of the L1 forecasts** (bottom-up), not separate models.
* Horizons h = 1..4 weeks (direct strategy). Units: sum of daily units per week.

## 3. Origins
Origin t = last observed week. Ten origins spaced 4 weeks apart so the 4-week targets do not overlap: t ∈ {65, 69, …, 101} (targets weeks 66..105). Each model sees only weeks ≤ t.

## 4. Models (all fixed)
Point forecasts (reported for h=1..4):
* `all_zero` — reference. `naive` (last week), `mean4`, `mean13`, `seasonal_naive52` (week t+h−52; defined for all origins ≥ 65 since t+h−52 ≥ 13),
  `croston_sba` (α=0.1, Syntetos-Boylan bias correction) and `tsb` (α_demand=0.1, α_prob=0.1), computed per series on history only.
* `lgbm_w` — global direct LightGBM, objective Tweedie (power 1.2), level offset `log(mean over the last 13 weeks + 0.1)`, 400 rounds max, learning rate 0.05, num_leaves 63, min_data_in_leaf 100,
  feature_fraction 0.8, early stopping 30 on the **last 4 training origins** (time-ordered, never random). Training rows: all origins t' with 26 ≤ t' ≤ t−h (target ≤ t), all series, one model with `h` as a feature.
  Features (all computed from weeks ≤ t'): lags 1–8, rolling mean over 4/8/13/26 weeks, share of non-zero weeks over 4/13/26, weeks since last sale, weeks since first sale (capped 104), rolling max over 13 weeks,
  mean of the same ISO-week-of-year ± 0 in the previous year **only if available (else NaN)**, target week-of-year (sin/cos), channel, h. No price, no event features (shown without value in the daily study).
* Quantiles (0.1, 0.5, 0.9) for pinball loss and an 80% interval: `lgbm_w` quantile boosters trained on `y / level` and rescaled by the level (as in the daily model; no conformal correction here) versus the baseline
  `empirical_q` = the series' own empirical quantiles of the last 26 weeks. `all_zero` quantiles are 0.

## 5. Metrics (L1 unless stated)
* Pooled WAPE, MAE, RMSE, bias% (positive = over-forecast); all-zero WAPE is 1 by definition.
* **MASE**: for each series, mean abs error over (origin, h) divided by that series' in-sample one-step naive MAE (weeks ≤ origin); series whose scale is 0 are excluded (counted). Reported as the mean over series.
* **RMSSE** analogously with squared errors. **Pinball loss** averaged over q ∈ {0.1, 0.5, 0.9} (pooled, in units); coverage of the 80% interval.
* L2/L3: WAPE and bias% on bottom-up sums per (origin, target week).
* Per-origin values and the number of origins where the model beats each baseline are always reported.

## 6. Success criteria for calling `lgbm_w` a serving candidate (declared in advance)
All of the following, on the ten origins:
1. Pooled L1 **MASE** of `lgbm_w` is at least **5% lower than the best baseline's MASE** (best among naive, mean4, mean13, seasonal_naive52, croston_sba, tsb) and lower in **≥ 7 of 10 origins**.
2. Pooled L1 **WAPE** is **≤ 0.90** (i.e. at least 10% better than the all-zero forecast).
3. **|pooled L1 bias|** ≤ 10% **and** |bias| at L3 (grand total) ≤ 10% in at least 6 of 10 origins.
4. Pooled pinball loss below that of `empirical_q`; 80% interval coverage between 0.70 and 0.90.
If any criterion fails the weekly model is **not** called a candidate; results are still published. No other model is substituted after the fact.

## 7. What would NOT be claimed even if all criteria hold
No statistical-significance claim (ten overlapping-in-information origins in one dataset, one store, two years). No claim about seasonality beyond one prior year. No external validity. The study is not a holdout evaluation (see header).
