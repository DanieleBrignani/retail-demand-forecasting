# Weekly study v2 — results (protocol `weekly-v2`)

> **Generated** by `python scripts/build_weekly_report.py` from `weekly-v2-retail-369d316281ea` (manifest below). Nothing in this file is typed by hand.
> **Status: revised historical evaluation on already-inspected data — not an independent holdout.** A new configuration hash does not create independent test data.
> The protocol is [`weekly_protocol_v2.md`](weekly_protocol_v2.md); thresholds were partly written after weekly v1 results (disclosed there).
> Weekly and daily metrics are **not comparable** (different units, aggregation and eligible universe).

## Manifest
* dataset `retail-369d316281ea`, weekly feature version `wfv1-b62b142b`, config SHA-256 `82550af575f407ae…`, seed 0, tuning trials 0.
* 105 complete 7-day blocks (dataset blocks, Tue–Mon), origins [65, 69, 73, 77, 81, 85, 89, 93, 97, 101], horizons [1, 2, 3, 4] blocks, runtime 482s.
* environment: {"python": "3.13.3", "platform": "Windows-11-10.0.26200-SP0", "cpu_count": 16, "numpy": "2.5.3", "pandas": "3.0.6", "lightgbm": "4.7.0"}
* command: `retailfc weekly backtest --config configs/weekly_v2.yaml`

## Selection (mechanical, `retailfc.weekly.select_model`)
**Selected: `mean4`** (strongest qualified baseline: `mean4`; qualified: ['mean4']). Every check of every candidate:

| candidate | check | result | detail |
|---|---|---|---|
| naive | forecast_coverage | pass | 1.0000 of eligible series-rows have a finite forecast (min 1.0) |
| naive | beats_all_zero | **FAIL** | wape 0.9806 vs all-zero 1.0000 (max ratio 0.95) |
| naive | mase_guard | pass | mase 2.5315 vs best baseline 2.3357 (max ratio 1.1) |
| naive | bias_guard | **FAIL** | pooled bias -1.4% (max ±10.0); 70% of origins within ±30.0% (min 80%); worst origin 39.8% |
| naive | runtime | pass | max train 0.00s (limit 600), max batch 0.07s (limit 120); baselines have no training (0 s) and their batch time is the time of all baselines together |
| naive | segment_degradation | **FAIL** | channel:INT_1 1.431 vs 1.260; channel:UK_1 0.889 vs 0.763; volume:high 0.942 vs 0.800; volume:mid 1.210 vs 1.096; intermittency:intermittent 1.095 vs 0.984; intermittency:smooth 0.869 vs 0.727; intermittency:sparse 1.280 vs 1.117; history:13-51w 0.917 vs 0.829; history:<13w 0.988 vs 0.805; history:>=52w 0.997 vs 0.856 |
| mean4 | forecast_coverage | pass | 1.0000 of eligible series-rows have a finite forecast (min 1.0) |
| mean4 | beats_all_zero | pass | wape 0.8465 vs all-zero 1.0000 (max ratio 0.95) |
| mean4 | mase_guard | pass | mase 2.3357 vs best baseline 2.3357 (max ratio 1.1) |
| mean4 | bias_guard | pass | pooled bias -7.0% (max ±10.0); 90% of origins within ±30.0% (min 80%); worst origin 31.1% |
| mean4 | runtime | pass | max train 0.00s (limit 600), max batch 0.07s (limit 120); baselines have no training (0 s) and their batch time is the time of all baselines together |
| mean4 | segment_degradation | pass | no segment >5% worse than the strongest baseline |
| mean13 | forecast_coverage | pass | 1.0000 of eligible series-rows have a finite forecast (min 1.0) |
| mean13 | beats_all_zero | pass | wape 0.8606 vs all-zero 1.0000 (max ratio 0.95) |
| mean13 | mase_guard | pass | mase 2.3634 vs best baseline 2.3357 (max ratio 1.1) |
| mean13 | bias_guard | **FAIL** | pooled bias -12.4% (max ±10.0); 90% of origins within ±30.0% (min 80%); worst origin 33.3% |
| mean13 | runtime | pass | max train 0.00s (limit 600), max batch 0.07s (limit 120); baselines have no training (0 s) and their batch time is the time of all baselines together |
| mean13 | segment_degradation | **FAIL** | intermittency:sparse 1.198 vs 1.117; history:<13w 0.896 vs 0.805 |
| seasonal_naive52 | forecast_coverage | pass | 1.0000 of eligible series-rows have a finite forecast (min 1.0) |
| seasonal_naive52 | beats_all_zero | **FAIL** | wape 1.3425 vs all-zero 1.0000 (max ratio 0.95) |
| seasonal_naive52 | mase_guard | **FAIL** | mase 2.8200 vs best baseline 2.3357 (max ratio 1.1) |
| seasonal_naive52 | bias_guard | pass | pooled bias +9.8% (max ±10.0); 90% of origins within ±30.0% (min 80%); worst origin 56.3% |
| seasonal_naive52 | runtime | pass | max train 0.00s (limit 600), max batch 0.07s (limit 120); baselines have no training (0 s) and their batch time is the time of all baselines together |
| seasonal_naive52 | segment_degradation | **FAIL** | channel:INT_1 1.727 vs 1.260; channel:UK_1 1.265 vs 0.763; volume:high 1.151 vs 0.800; volume:low 2.480 vs 1.162; volume:mid 1.666 vs 1.096; intermittency:intermittent 1.386 vs 0.984; intermittency:smooth 1.108 vs 0.727; intermittency:sparse 1.647 vs 1.117; history:13-51w 1.112 vs 0.829; history:<13w 1.000 vs 0.805; history:>=52w 1.453 vs 0.856 |
| croston_sba | forecast_coverage | pass | 1.0000 of eligible series-rows have a finite forecast (min 1.0) |
| croston_sba | beats_all_zero | **FAIL** | wape 1.3303 vs all-zero 1.0000 (max ratio 0.95) |
| croston_sba | mase_guard | **FAIL** | mase 4.4352 vs best baseline 2.3357 (max ratio 1.1) |
| croston_sba | bias_guard | **FAIL** | pooled bias +38.7% (max ±10.0); 30% of origins within ±30.0% (min 80%); worst origin 87.8% |
| croston_sba | runtime | pass | max train 0.00s (limit 600), max batch 0.07s (limit 120); baselines have no training (0 s) and their batch time is the time of all baselines together |
| croston_sba | segment_degradation | **FAIL** | channel:INT_1 2.626 vs 1.260; channel:UK_1 1.068 vs 0.763; volume:high 0.901 vs 0.800; volume:low 2.233 vs 1.162; volume:mid 1.423 vs 1.096; intermittency:intermittent 1.236 vs 0.984; intermittency:sparse 1.813 vs 1.117; history:13-51w 1.276 vs 0.829; history:<13w 0.975 vs 0.805; history:>=52w 1.396 vs 0.856 |
| tsb | forecast_coverage | pass | 1.0000 of eligible series-rows have a finite forecast (min 1.0) |
| tsb | beats_all_zero | **FAIL** | wape 0.9806 vs all-zero 1.0000 (max ratio 0.95) |
| tsb | mase_guard | pass | mase 2.4119 vs best baseline 2.3357 (max ratio 1.1) |
| tsb | bias_guard | pass | pooled bias +1.8% (max ±10.0); 80% of origins within ±30.0% (min 80%); worst origin 38.2% |
| tsb | runtime | pass | max train 0.00s (limit 600), max batch 0.07s (limit 120); baselines have no training (0 s) and their batch time is the time of all baselines together |
| tsb | segment_degradation | **FAIL** | channel:UK_1 0.919 vs 0.763; volume:high 0.927 vs 0.800; volume:low 1.560 vs 1.162; volume:mid 1.203 vs 1.096; intermittency:intermittent 1.263 vs 0.984; intermittency:sparse 1.435 vs 1.117; history:13-51w 1.335 vs 0.829; history:<13w 0.902 vs 0.805 |
| lgbm_w | forecast_coverage | pass | 1.0000 of eligible series-rows have a finite forecast (min 1.0) |
| lgbm_w | beats_all_zero | pass | wape 0.8251 vs all-zero 1.0000 (max ratio 0.95) |
| lgbm_w | mase_guard | **FAIL** | mase 3.0787 vs best baseline 2.3357 (max ratio 1.1) |
| lgbm_w | bias_guard | pass | pooled bias -7.6% (max ±10.0); 100% of origins within ±30.0% (min 80%); worst origin 26.8% |
| lgbm_w | runtime | pass | max train 70.04s (limit 600), max batch 0.85s (limit 120); baselines have no training (0 s) and their batch time is the time of all baselines together |
| lgbm_w | segment_degradation | **FAIL** | volume:low 1.504 vs 1.162; volume:mid 1.175 vs 1.096 |
| lgbm_w | improves_on_strongest_baseline | pass | wape gain +2.53% vs mean4 (min 1.0%) |

## L1 (product × channel) pooled metrics, eligible universe, 10 origins × 4 horizons
`all_zero` is a reference only (never selectable). MASE/RMSSE exclude, for every model, rows whose pre-origin history is constant (column `n_mase_excluded`).
| model | wape | mae | rmse | bias_pct | mase | rmsse | n | n_mase_excluded |
|---|---|---|---|---|---|---|---|---|
| all_zero | 1.0000 | 12.1124 | 63.1656 | -1.0000 | 2.3574 | 5.9893 | 349480.0000 | 44336.0000 |
| croston_sba | 1.3303 | 16.1128 | 71.1518 | 0.3867 | 4.4352 | 6.0251 | 349480.0000 | 44336.0000 |
| lgbm_w | 0.8251 | 9.9939 | 47.9534 | -0.0757 | 3.0787 | 5.8634 | 349480.0000 | 44336.0000 |
| mean13 | 0.8606 | 10.4242 | 56.1379 | -0.1239 | 2.3634 | 5.9254 | 349480.0000 | 44336.0000 |
| mean4 | 0.8465 | 10.2529 | 49.5305 | -0.0702 | 2.3357 | 5.8801 | 349480.0000 | 44336.0000 |
| naive | 0.9806 | 11.8778 | 61.0951 | -0.0136 | 2.5315 | 5.8518 | 349480.0000 | 44336.0000 |
| seasonal_naive52 | 1.3425 | 16.2611 | 106.1857 | 0.0980 | 2.8200 | 6.0259 | 349480.0000 | 44336.0000 |
| tsb | 0.9806 | 11.8778 | 125.0258 | 0.0183 | 2.4119 | 5.9260 | 349480.0000 | 44336.0000 |

Aggregated levels (bottom-up sums of eligible L1 forecasts; errors can cancel — this does not imply accurate individual products):
| model | wape L2_channel | wape L3_total | bias_pct L2_channel | bias_pct L3_total |
|---|---|---|---|---|
| all_zero | 1.0000 | 1.0000 | -1.0000 | -1.0000 |
| croston_sba | 0.4843 | 0.4124 | 0.3867 | 0.3867 |
| lgbm_w | 0.1996 | 0.1583 | -0.0757 | -0.0757 |
| mean13 | 0.2513 | 0.2239 | -0.1239 | -0.1239 |
| mean4 | 0.2228 | 0.1717 | -0.0702 | -0.0702 |
| naive | 0.2755 | 0.2149 | -0.0136 | -0.0136 |
| seasonal_naive52 | 0.2923 | 0.2408 | 0.0980 | 0.0980 |
| tsb | 0.2635 | 0.2218 | 0.0183 | 0.0183 |

WAPE by horizon (blocks):
| h | all_zero | croston_sba | lgbm_w | mean13 | mean4 | naive | seasonal_naive52 | tsb |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.3459 | 0.8249 | 0.8580 | 0.8396 | 0.9604 | 1.2541 | 0.9865 |
| 2 | 1.0000 | 1.3640 | 0.8408 | 0.8790 | 0.8564 | 0.9952 | 1.3703 | 1.0034 |
| 3 | 1.0000 | 1.2902 | 0.8061 | 0.8380 | 0.8316 | 0.9783 | 1.3551 | 0.9488 |
| 4 | 1.0000 | 1.3241 | 0.8298 | 0.8687 | 0.8587 | 0.9886 | 1.3881 | 0.9859 |

## Origin-level variability (WAPE per origin) — no significance claim
| model | mean | std | min | max | origins_better_than_mean4 |
|---|---|---|---|---|---|
| all_zero | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| croston_sba | 1.3992 | 0.2477 | 1.0003 | 1.6935 | 0.0000 |
| lgbm_w | 0.8366 | 0.0635 | 0.6779 | 0.9239 | 8.0000 |
| mean13 | 0.8880 | 0.1195 | 0.7138 | 1.1397 | 5.0000 |
| mean4 | 0.8681 | 0.0825 | 0.6934 | 0.9587 | 0.0000 |
| naive | 1.0102 | 0.1549 | 0.7808 | 1.2577 | 0.0000 |
| seasonal_naive52 | 1.3686 | 0.1518 | 1.1895 | 1.7527 | 0.0000 |
| tsb | 1.0229 | 0.1596 | 0.7520 | 1.2519 | 0.0000 |

## Segments (derived from past data only) — WAPE
|  | all_zero | croston_sba | lgbm_w | mean13 | mean4 | naive | seasonal_naive52 | tsb |
|---|---|---|---|---|---|---|---|---|
| channel INT_1 | 1.0000 | 2.6258 | 1.2436 | 1.2051 | 1.2596 | 1.4313 | 1.7273 | 1.2838 |
| channel UK_1 | 1.0000 | 1.0681 | 0.7404 | 0.7909 | 0.7629 | 0.8894 | 1.2646 | 0.9193 |
| history 13-51w | 1.0000 | 1.2765 | 0.7986 | 0.8455 | 0.8292 | 0.9167 | 1.1124 | 1.3349 |
| history <13w | 1.0000 | 0.9749 | 0.7709 | 0.8964 | 0.8050 | 0.9876 | 1.0000 | 0.9016 |
| history >=52w | 1.0000 | 1.3963 | 0.8326 | 0.8590 | 0.8563 | 0.9965 | 1.4535 | 0.8978 |
| history no_channel_history | 1.0000 | 1.0000 | 2.5350 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| intermittency inactive | 1.0000 | 45.1071 | 2.2154 | 1.0000 | 1.0000 | 1.0000 | 13.7073 | 1.4001 |
| intermittency intermittent | 1.0000 | 1.2361 | 0.9403 | 1.0042 | 0.9839 | 1.0948 | 1.3864 | 1.2629 |
| intermittency smooth | 1.0000 | 0.7328 | 0.6916 | 0.7255 | 0.7267 | 0.8687 | 1.1078 | 0.7524 |
| intermittency sparse | 1.0000 | 1.8127 | 1.1441 | 1.1982 | 1.1169 | 1.2795 | 1.6474 | 1.4346 |
| volume high | 1.0000 | 0.9010 | 0.7420 | 0.8151 | 0.8002 | 0.9420 | 1.1511 | 0.9270 |
| volume inactive | 1.0000 | 45.1071 | 2.2154 | 1.0000 | 1.0000 | 1.0000 | 13.7073 | 1.4001 |
| volume low | 1.0000 | 2.2334 | 1.5044 | 1.1612 | 1.1620 | 1.1845 | 2.4797 | 1.5602 |
| volume mid | 1.0000 | 1.4227 | 1.1754 | 1.1097 | 1.0963 | 1.2101 | 1.6658 | 1.2033 |

## Universe, cold start and fallback
Products first sold after the origin are **not scored** (cold start); their realised demand share is below. Fallback rate = share of scored rows whose channel had no earlier sale (`no_channel_history`): 0.1269; minimum forecast coverage over models 100% (measured: share of eligible series-rows with a finite forecast).
| origin | n_eligible | n_cold_start | target_units_eligible | target_units_cold_start | share_units_cold_start |
|---|---|---|---|---|---|
| 65 | 8204.0000 | 1210.0000 | 301869.0000 | 4847.0000 | 0.0158 |
| 69 | 8282.0000 | 1132.0000 | 325296.0000 | 5847.0000 | 0.0177 |
| 73 | 8372.0000 | 1042.0000 | 339316.0000 | 26383.0000 | 0.0721 |
| 77 | 8538.0000 | 876.0000 | 324842.0000 | 17142.0000 | 0.0501 |
| 81 | 8726.0000 | 688.0000 | 319002.0000 | 5377.0000 | 0.0166 |
| 85 | 8800.0000 | 614.0000 | 406348.0000 | 6445.0000 | 0.0156 |
| 89 | 8874.0000 | 540.0000 | 387007.0000 | 7201.0000 | 0.0183 |
| 93 | 9016.0000 | 398.0000 | 572252.0000 | 19346.0000 | 0.0327 |
| 97 | 9208.0000 | 206.0000 | 567918.0000 | 16410.0000 | 0.0281 |
| 101 | 9350.0000 | 64.0000 | 689206.0000 | 4207.0000 | 0.0061 |

## Prediction intervals (80% nominal) — experimental
Overall and active-series-only coverage, width, pinball and the raw quantile-crossing rate **before** the monotone correction:
|  | empirical_coverage | mean_width | pinball | raw_crossing_rate |
|---|---|---|---|---|
| lgbm_w all | 0.9147 | 23.4545 | 3.2164 | 0.0211 |
| lgbm_w active_only | 0.8708 | 37.3747 | 5.0532 | 0.0215 |
| lgbm_w_cal all | 0.9147 | 23.4545 | 3.2164 | 0.0211 |
| lgbm_w_cal active_only | 0.8708 | 37.3747 | 5.0532 | 0.0215 |
| empirical_q all | 0.9080 | 23.3438 | 3.6428 | 0.0000 |
| empirical_q active_only | 0.8601 | 37.1983 | 5.7325 | 0.0000 |
| all_zero all | 0.7301 | 0.0000 | 6.0562 | 0.0000 |
| all_zero active_only | 0.5766 | 0.0000 | 9.5783 | 0.0000 |

Coverage by past-volume segment (all rows). Aggregate coverage hides the split: the `inactive` segment is almost always covered by a zero-width interval at 0, while active segments are covered less:
| key | empirical_q | lgbm_w |
|---|---|---|
| high | 0.8182 | 0.8263 |
| inactive | 0.9887 | 0.9887 |
| low | 0.8969 | 0.9116 |
| mid | 0.8648 | 0.8742 |
The conformal correction from the preceding origin is exactly 0 on this data (more than 80% of the conformity scores are exactly 0 because most actuals and quantiles are 0), so `lgbm_w_cal` equals `lgbm_w`: the calibration step is uninformative on zero-heavy data.
Intervals are therefore **not attached to serving**.

## Repetition (same configuration, same data, second execution: `weekly-v2-repeat-retail-369d316281ea`)
Maximum absolute difference of each pooled metric between the two executions (baselines are deterministic; LightGBM is multi-threaded, so bitwise equality is **not promised** — here the two runs happened to coincide on the same machine with the same seed and thread count):
| model | wape | mae | rmse | bias_pct | mase |
|---|---|---|---|---|---|
| all_zero | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| croston_sba | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| lgbm_w | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| mean13 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| mean4 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| naive | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| seasonal_naive52 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| tsb | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

Selected model in the repetition: `mean4` (first run: `mean4`). Documented tolerance if a rerun differs: 1e-9 for baselines, 5e-3 absolute on pooled metrics for `lgbm_w`.
