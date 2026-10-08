# Retraining policy

Implemented in `retailfc.orchestration.retraining.decide` (pure, unit-tested) and run by the Prefect flow `retraining-evaluation` (scheduled daily). **Time passing is not a reason
to retrain**; the scheduled check usually concludes `no_retrain`. Every outcome is written to `app.retraining_decisions` with inputs, policy and reasons (audit trail).

| Order | Gate | Outcome |
|---|---|---|
| 1 | Another training/promotion run holds the PostgreSQL advisory lock | `skipped_locked` |
| 2 | Data quality: missing-actuals fraction > 5% on matured forecasts, any conflicting actuals, or feature missingness > 20% | `blocked_data_quality` (never retrains on questionable data) |
| 3 | Cooldown: trained < 14 days ago | `no_retrain` |
| 4 | Triggers — (a) **performance**: WAPE on matured forecasts of the current model ≥ 10% worse than its validation reference, with ≥ 5000 matured observations; (b) **governance**: model older than 180 days | candidate `retrain` |
| 5 | Minimum new data: ≥ 14 new actual days since the model's last forecast origin | otherwise `no_retrain` |
| – | **Drift** (demand PSI ≥ 0.25 for 3 consecutive checks) alone | `investigate` — drift is not accuracy degradation |

Missing actuals: forecasts without actuals are not scored; if the unscored share of *matured* forecasts exceeds the limit the decision is blocked, not "degraded".

On `retrain`: the challenger is trained, scored on a matured out-of-sample window against the incumbent, and promoted only if all promotion rules pass
(`docs/experiment_plan.md` §9). Training, challenger evaluation and promotion share one advisory lock. Thresholds are defaults in `config/experiment.yaml` (`retraining:`);
they are policy choices, not statistically derived, and should be revisited once real monitoring history exists.
