# Model card — weekly line (`retailfc_weekly`, selected model: 4-week trailing mean)

Numbers are copied from the generated [`results_weekly_v2.md`](results_weekly_v2.md) (protocol `weekly-v2`, config SHA-256 in its manifest). **Revised historical evaluation on already-inspected data — not an independent holdout.**

## Model
* **Selected and served:** `mean4` — the forecast for blocks t+1…t+4 is the mean of the last four completed 7-day blocks of the series (a baseline; nothing is trained). It was selected by the mechanical rule of [`weekly_protocol_v2.md`](weekly_protocol_v2.md) §7.
* **Rejected candidate:** global direct LightGBM (`lgbm_w`; Tweedie, level offset, no price/event features, zero tuning trials). Reasons from the selection record: MASE 3.079 vs 2.336 (guard 1.10×) and WAPE 1.504 vs 1.162 on low-volume and 1.175 vs 1.096 on mid-volume series (guard +5%).
* Registry: `retailfc_weekly` v1 holds a specification (family, name, parameters, protocol/config hashes); inference runs through `retailfc.weekly.baseline_forecasts`. Approved by rule `weekly_v2`.

## Intended use and non-use
Weekly units per product × sales channel for the products already observed at the origin, four blocks ahead, as a monitored reference forecast.
**Not validated for:** new products (cold start is quantified, not evaluated), inventory decisions (sales ≠ demand; no cost objective evaluated), promotion planning, other retailers, or any claim of causal effects.

## Evaluation (10 rolling origins 65…101 step 4, horizons 1–4 blocks; identical rows for every model)
| model | WAPE | MAE | RMSE | bias | MASE |
|---|---|---|---|---|---|
| all-zero (reference only) | 1.0000 | 12.11 | 63.17 | −100% | 2.357 |
| **mean4 (selected)** | **0.8465** | 10.25 | 49.53 | −7.0% | **2.336** |
| mean13 | 0.8606 | 10.42 | 56.14 | −12.4% | 2.363 |
| TSB | 0.9806 | 11.88 | 125.03 | +1.8% | 2.412 |
| naive | 0.9806 | 11.88 | 61.10 | −1.4% | 2.532 |
| seasonal naive (52 wk) | 1.3425 | 16.26 | 106.19 | +9.8% | 2.820 |
| Croston/SBA | 1.3303 | 16.11 | 71.15 | +38.7% | 4.435 |
| LightGBM (rejected) | 0.8251 | 9.99 | 47.95 | −7.6% | 3.079 |

Channel/total aggregates (bottom-up): `mean4` WAPE 0.2228 / 0.1717, LightGBM 0.1996 / 0.1583 — errors cancel across series and do **not** imply accurate individual products.
Origin-level WAPE: `mean4` mean 0.868 (std 0.082); LightGBM 0.837 (std 0.063), better than `mean4` in 8 of 10 origins (WAPE) — no significance claim (correlated series, overlapping information).

## Known risks and limitations
Bias per origin is unstable (`mean4`: −31% to +13% across origins; pooled −7%), so the pooled figure alone would be misleading; WAPE is volume-weighted; the all-zero forecast has WAPE 1 by definition and absolute-error metrics reward predicting zero on sparse series; a moving average cannot anticipate seasonality or promotions;
12.7% of scored rows have a constant pre-origin history and are excluded from MASE/RMSSE for every model; prediction intervals are experimental and are **not** served; the data are replayed 2011 transactions.
Replay lifecycle evidence: [`results_weekly_operations.md`](results_weekly_operations.md).

---
> **HISTORICAL — daily LightGBM card (superseded).** The text below describes the daily model `retailfc_lgbm` v1/v2 evaluated with feature set D `fv1` (leaking price features) and without the origin-appropriate universe. Kept verbatim; its approval is revoked. See [`audit_price_leakage.md`](audit_price_leakage.md) and [`results_daily_fv2.md`](results_daily_fv2.md).

# Model card — global direct LightGBM demand forecaster (`retailfc_lgbm`)

All numbers below are copied from [`docs/results_online_retail.md`](results_online_retail.md) (generated, verbatim) unless a command is named. Dataset: UCI Online Retail II, adapted
(9,414 product × channel series, 739 days, 91.7% zero series-days). One locked holdout window and three validation windows — **few independent time blocks**.

## Intended use
Daily unit-sales forecasts (product × channel, 1–28 days ahead) for sparse retail panels of this shape, served from precomputed batches. **Not validated for:** new products without history,
promotion planning (prices here are derived from the same transactions), inventory or replenishment decisions (sales ≠ demand), other retailers, or low-volume/inactive series (see below).

## Model
* LightGBM, **direct global multi-horizon**: one model; rows = (series, origin, horizon); `h` is a feature; objective Tweedie (variance power 1.1; bounded random search over leaves/min-leaf/feature-fraction/learning-rate/L2, 4 trials).
  Selected parameters: `num_leaves=31, min_data_in_leaf=300, feature_fraction=0.8, learning_rate=0.05, lambda_l2=1.0`.
* Features: `docs/feature_dictionary.md` (origin-anchored history, identifiers, target-date calendar, holiday events, price features). Selected feature set: **D** (all groups).
* Training: seeded random subset of 3,000 series (evaluation on all 9,414), 252-day window of training origins, early stopping at origin−28 then refit.
* Intervals: quantile boosters (0.1, 0.9) with conformalised offsets; point and interval forecasts clipped at 0. Fallbacks (all-zero history, unseen identifier, non-finite output) are always flagged.
* Provenance: dataset version, feature version, config, hyper-parameters, origins, environment and evidence are logged to MLflow for registered versions (no model has been registered in the real-data run yet).

## Evaluation protocol
Rolling origin: validation origins d_627 / d_655 / d_683 (targets 2011-08-20 … 2011-11-11), final holdout origin d_711 (targets 2011-11-12 … 2011-12-09), protocol locked (config hash) before the holdout, holdout evaluated once.
Primary metric pooled WAPE; secondary MAE, RMSE, sMAPE, bias (forecast − actual).

## Results (pooled, horizons 1–28, all 9,414 series)
| | Validation (3 origins) WAPE / MAE / RMSE | Holdout (1 origin) WAPE / MAE / RMSE |
|---|---|---|
| **LightGBM (set D)** | 1.1146 / 2.282 / 19.11 | **1.0426** / 2.763 / 35.79 |
| mean (28 d) | 1.2588 / 2.578 / **18.83** | 1.1470 / 3.040 / **35.07** |
| seasonal mean | 1.2708 / 2.602 / 20.08 | 1.1518 / 3.052 / 36.15 |
| naive | 1.3085 / 2.679 / 20.56 | 1.4279 / 3.784 / 36.86 |
| seasonal naive | 1.3603 / 2.785 / 22.83 | 1.3016 / 3.450 / 38.17 |
| *all-zero forecast (reference)* | *1.0000 / 2.048 / 19.71* | *1.0000 / 2.650 / 36.37* |

What this does and does not show:
* **LightGBM has lower pooled WAPE and MAE than every baseline** (holdout: −20% vs seasonal naive, −27% vs naive, −9% vs the mean baseline; validation: −18%, −15%, −11%), and it was better in each of the 3 validation origins (3/3 vs naive and seasonal naive).
* **But every model has WAPE above 1, so none beats the trivial all-zero forecast on WAPE or MAE.** On data that are ~90% zeros these metrics are minimised by predicting nothing. The ranking among the non-zero models is therefore the informative part, not the absolute level. No all-zero baseline was in the pre-registered protocol; it is reported as a post-hoc reference computed from the stored actuals.
* **On RMSE the mean baseline is better than LightGBM** in both periods; LightGBM is better than naive, seasonal naive and the all-zero forecast on RMSE. sMAPE is not usable here (it gives LightGBM ≈160 vs 24–26 for naive because of the both-zero convention).
* **Where it helps:** high-volume series (holdout WAPE 0.979 vs 1.13–1.39 for baselines) and the UK channel (0.965 vs 1.05–1.31); the lumpy class (holdout 1.02 vs 1.16–1.44) and, in validation, the erratic class — but in the holdout the erratic class is better served by the mean baselines (0.87 vs 0.78–0.83). **Where it hurts:** low-volume series (1.74 vs 1.29 for the mean baseline), inactive series (1.22 vs 1.00), and the "intermittent" class (1.63 vs 1.50–1.53 for the mean baselines). In validation, LightGBM beat seasonal naive on only ~24% of individual series (descriptive): it wins on aggregate because high-volume series dominate WAPE.
* **Feature groups:** adding calendar (set B) made validation WAPE *worse* (−0.54% relative); events (C) +0.22% (below the 0.5% rule); price (D) +0.67% over A, which the rule adopted — a gain smaller than the origin-to-origin swings (e.g. 1.07–1.20 for the same model). Holiday events cannot be assessed: only **one** holiday day falls inside all four evaluated windows. Permutation importance (one origin): history 0.109, calendar 0.056, price −0.007 (negative), ids −0.004, events 0.000. Prices are derived from the same transactions.
* **Restricted 120-series subset (same series for all models), holdout:** LightGBM 1.012, SARIMA 1.015 (practically tied), mean 1.054, seasonal mean 1.097, naive 1.246, seasonal naive 1.857; SARIMA has the better RMSE of the two (21.7 vs 25.9). In validation seasonal naive (0.995) was marginally ahead of LightGBM set A (0.998).
* **Intervals:** empirical coverage 0.94 for a nominal 0.80, with identical raw and calibrated intervals (offsets 0): with ~90% zero outcomes the clipped intervals trivially cover zeros, so this coverage is uninformative; 4.6% quantile crossing; mean width 2.54. No coverage guarantee is claimed.
* **Bias:** pooled bias is negative (−0.43 to −0.48 of total demand: under-forecasting on the high-volume series), yet 43–44% of series are persistently over-forecast at every origin — consistent with over-forecasting on many low-volume series being behind WAPE above 1 (not isolated by a dedicated experiment).

## Operational run on the real data
The model (data ≤ day 711) was registered as version 1 and **not promoted** by the objective rules (bias −45.35% vs the ±10% limit; the other checks passed). Its 263,592 stored forecasts reproduce the holdout WAPE (1.0426) when scored against the replayed actuals. Details: `docs/results_online_retail_operations.md`.

## Cost (indicative, one run on a laptop with other applications open — not a benchmark)
Model inference benchmark (`retailfc benchmark inference`): p50 48.7 ms / p95 50.2 ms for 500 series × 28 horizons (≈0.1 ms per series; 15 repetitions after 2 warm-ups). Holdout stage on a free machine: LightGBM fit 4.8 s on 840,000 rows (73 rounds), prediction 0.4 s for 263,592 forecasts; SARIMA 39 s for 120 series (28 days each); baselines < 0.1 s. The validation-stage runtimes in the generated report are contaminated by CPU/memory starvation and must not be used.
No monetary cost is estimated (no priced resources, no assumptions).

## Known risks
Volume-weighted primary metric; the zero forecast is unbeaten on WAPE/MAE; a single holdout window in the Nov–Dec run-up to Christmas (the all-zero forecast's MAE, i.e. mean demand, is 2.65 there vs 2.05 in validation) with only two earlier December periods in the data; derived channels, events and prices; unobserved stock-outs; the model is not causal.
