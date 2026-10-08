# Daily comparison under the corrected pipeline (`fv2` prices + origin-appropriate universe)

> **Generated** by `python scripts/build_daily_report.py` from `online_retail_case_study-retail-369d316281ea-d797ee67b35e0c5e/daily-fv2-universe`. Status: **revised historical evaluation on already-inspected data** (origins 627, 655, 683, 711 are the original validation windows and the consumed holdout) — not an independent holdout.
> Command: `retailfc backtest variants --origins 627,655,683,711 --version retail-369d316281ea --config configs/online_retail.yaml --tag daily-fv2-universe`.
> Daily metrics are **not comparable** with the weekly ones (different units, aggregation and eligible universe). LightGBM variants use feature set D under `fv2`; `v2_all_offset` = all series + level offset (the former "v2" recipe, now without the price look-ahead).
> `all_zero_reference` = forecasting zero everywhere (WAPE is exactly 1 by definition).

## Pooled over the four origins (eligible universe at each origin; identical rows for every model)
| model | wape | mae | rmse | bias_pct | n |
|---|---|---|---|---|---|
| all_zero_reference | 1.0000 | 2.1975 | 20.0112 | -1.0000 | 1022168.0000 |
| v1_subset | 1.0798 | 2.3728 | 19.2156 | -0.4405 | 1022168.0000 |
| offset_subset | 1.0901 | 2.3956 | 18.7206 | -0.2880 | 1022168.0000 |
| all_series | 1.1426 | 2.5109 | 18.3990 | -0.1241 | 1022168.0000 |
| v2_all_offset | 1.1673 | 2.5651 | 18.3475 | -0.0561 | 1022168.0000 |
| mean | 1.2323 | 2.7080 | 18.7181 | -0.0934 | 1022168.0000 |
| seasonal_mean | 1.2424 | 2.7302 | 20.1938 | -0.0934 | 1022168.0000 |
| seasonal_naive | 1.3536 | 2.9744 | 23.1952 | -0.0561 | 1022168.0000 |
| naive | 1.3554 | 2.9786 | 20.8891 | -0.0919 | 1022168.0000 |

## WAPE per origin
| model | 627 | 655 | 683 | 711 |
|---|---|---|---|---|
| all_series | 1.2497 | 1.1158 | 1.1666 | 1.0805 |
| all_zero_reference | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mean | 1.4185 | 1.1784 | 1.2505 | 1.1515 |
| naive | 1.3839 | 1.2706 | 1.3197 | 1.4411 |
| offset_subset | 1.1526 | 1.0677 | 1.1486 | 1.0191 |
| seasonal_mean | 1.4506 | 1.1797 | 1.2600 | 1.1565 |
| seasonal_naive | 1.5186 | 1.4193 | 1.2343 | 1.3110 |
| v1_subset | 1.1515 | 1.0694 | 1.1023 | 1.0260 |
| v2_all_offset | 1.2477 | 1.1146 | 1.2318 | 1.1052 |

## Universe at each origin (cold-start products are not scored)
| origin | n_eligible | n_not_yet_launched | share_demand_not_yet_launched |
|---|---|---|---|
| 627 | 8874.0000 | 540.0000 | 0.0288 |
| 655 | 9030.0000 | 384.0000 | 0.0362 |
| 683 | 9232.0000 | 182.0000 | 0.0283 |
| 711 | 9370.0000 | 44.0000 | 0.1131 |

## Reading (interpretation, not part of any protocol)
* No model beats the all-zero forecast on WAPE or MAE at daily granularity; this is expected for ~92% zeros (the median is 0). The lowest RMSE is reached by the LightGBM variants.
* Variants that reduce bias (all series, level offset) raise WAPE and MAE: bias and absolute error trade off on this data.
* This is why the main candidate experiment is weekly ([`results_weekly_v2.md`](results_weekly_v2.md)); the daily line is not served and its former approval is revoked.
* Seeds/params: "see config"; `training.max_series`/`level_offset` per variant are in `retailfc.experiments.DEFAULT_VARIANTS`.
