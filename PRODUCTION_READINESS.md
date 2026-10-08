# Production-readiness review — state after the evidence-based remediation (2026-10-05)

**Bottom line.** This is a **portfolio demonstration** with a credible negative/modest scientific result. It is **not production-ready** and no claim of model superiority is made. Three readings are kept apart:
(1) portfolio-demonstration readiness, (2) scientific-evidence quality, (3) readiness for a specified production environment (none was specified, so none is claimed).

**Unresolved critical items (prominent):**
* No independent holdout exists: every block of the dataset has been inspected, so all results are *revised historical evaluations*.
* CI: run #1 on GitHub failed (pytest installed with `--no-deps`), run #2 on `f03c53b` passed all four jobs; the repository is private, so no public badge. The CI uses a throwaway API key and fixtures only.
* The daily LightGBM line is invalid as evidence (price look-ahead `fv1`); its approval is revoked. A corrected daily comparison is reported (`docs/results_daily_fv2.md`) but the daily line is not served.
* Genuinely unseen (cold-start) products are not evaluated, only quantified.

Legend: PASS = demonstrated by an executed check whose outcome is recorded; PARTIAL = implemented/tested but with a stated gap; FAIL = not met. "Evidence" is a repository path; "executed check" is what was actually run and its outcome.

## 1. Portfolio-demonstration readiness
| # | Requirement | Status | Repository evidence | Executed check and outcome | Limitation | Required improvement |
|---|---|---|---|---|---|---|
| 1 | Source integrity (immutable versions, checksums, validation) | PASS | `ingestion/raw.py`, `ingestion/validate.py`, `tests/test_ingestion.py` | `pytest` ingestion tests pass; `verify_version` run in every pipeline start | Only the one UCI file; no schema registry | Add provenance for a second source |
| 2 | Correct target and eligible scope documented | PASS | `docs/data_contract.md`, `adapters/online_retail.py`, `tests/test_online_retail_adapter.py` | adapter tests pass (8); counts in `adapter_report.json` | Gross recorded sales (returns not netted); zeros ≠ proven zero demand | Net returns if invoices become linkable |
| 3 | Temporal safety **from original transactions onward** | PASS | `tests/test_price_availability.py`, `Panel.as_of`, `tests/test_features_leakage.py`, `tests/test_weekly.py` | defect reproduced first (10→55), then fixed; full suite 186 passed (incl. 11 price-availability, 19 weekly, 14 weekly-lifecycle tests) | Tests are on hand-made tables; real-data leakage cannot be disproved, only tested | Property-based tests on random transaction tables |
| 4 | Series-universe policy (no future catalogue) | PASS | `Panel.eligible_at`, `weekly.eligible_mask`, `docs/data_contract.md` §4, tests | encoder/training/evaluation tests pass; `universe_coverage.csv` per origin | Cold-start products are quantified, not forecast-evaluated | Evaluate a documented cold-start rule |
| 5 | Valid temporal splits, no label in evaluation windows | PASS | `docs/weekly_protocol_v2.md` §4, `weekly.training_set`, `backtesting/runner.py` | tests on training targets/boundaries pass; weekly v2 run executed | Early-stopping window is the last 4 target blocks of the training period | — |
| 6 | Reproducible metrics (config hash, manifest, regenerated reports) | PARTIAL | `weekly.run` manifest, `scripts/build_weekly_report.py`, `docs/results_weekly_v2.md` | full run executed twice: identical pooled metrics (see report "Repetition"); not promised bitwise in general | LightGBM multi-thread not bit-reproducible; tolerance documented | Pin threads for a bitwise mode |
| 7 | Honest promotion decisions | PARTIAL | `weekly.select_model`, `docs/weekly_protocol_v2.md` §7, `registry/promotion.py` | selection executed mechanically; LightGBM rejected, `mean4` selected; unit tests of the rule pass | Two thresholds written after weekly v1 results; the daily v2 pass was post-hoc and is revoked | Freeze rules before any future dataset |
| 8 | Model / data / feature lineage (generation refuses a different config/dataset; approval verifies dataset, protocol, config and evidence-file hashes) | PASS | `forecast_batches` (dataset, feature, protocol version), MLflow tags, `manifest.json` | weekly batch row shows `weekly-v2`, `wfv1-…`, dataset version | Registry for the baseline stores a spec, not weights | — |
| 9 | Idempotent, append-only persistence | PASS | `forecasting/store.py`, `tests/test_weekly_lifecycle.py`, `tests/test_db_lifecycle.py` | rerun → `noop_identical`; new vintage keeps old rows; second reconciliation → 0 new errors (also on real data) | Single-node PostgreSQL | Concurrency stress test |
| 10 | Actuals reconciliation (daily and weekly) | PASS | `store.reconcile`, tests | partial-release test: block scored only when all 7 days exist; DB WAPE equals backtest WAPE on real data | Replay only; no late-arriving corrections | Late-actuals policy |
| 11 | Secret exclusion | PASS | `.gitignore`, `.env.example`, `tests` (no secrets), zip excludes `.env` | `git ls-files` contains no `.env`; keys must be ≥ 16 chars and are fail-closed | No secret scanner in CI | Add a secret scan job |
| 12 | Functioning demonstrated lifecycle (end to end) | PASS | `scripts/run_weekly_operations.sh`, `docs/results_weekly_operations.md` | registration → approval → batch → replay → reconcile → monitor → API executed on real data; Compose stack rebuilt and probed | Historical replay on 2011 data; wall-clock alerts meaningless | Run against a live feed |
| 13 | API: success and error paths, auth | PASS | `api/app.py`, `tests/test_api.py`, `tests/test_weekly_lifecycle.py` | 401/200/422/404/503 paths exercised in tests and by curl on the rebuilt stack | No load test, no TLS, no rate limiting | Behind a gateway; load test |
| 14 | Docker Compose rebuilt and healthy | PASS | `docker-compose.yml` | `docker compose build` + `up -d --wait`: all services healthy; targets up; 13 alert rules healthy | Dashboards not inspected visually | Screenshot-based check |
| 15 | dbt checks (integrity, negative controls) | PASS | `dbt/`, `tests/test_dbt_warehouse.py` | 3 integration tests incl. negative controls pass; `dbt build` on the real data: PASS=35 in 119 s | Full-panel join-count test replaced by an exhaustive key-uniqueness proof (not a sample) | — |
| 16 | CI workflow | PASS | `.github/workflows/ci.yml` | pushed 2026-10-08: run #1 failed (pytest `--no-deps` install), run #2 (`f03c53b`) green on `lint`, `unit`, `integration` (PostgreSQL + dbt), `docker`; evidence is the Actions page (screenshot reviewed by the author) | Private repository (no public badge); runs fixtures only, not the real-data pipeline; Node 20 deprecation warnings | Pin newer action versions; Push to an authorised remote and read the run |
| 17 | Documentation consistent with the verified state | PASS | README, this file, `docs/` | contradictions listed in the task were resolved (see "Historical vs current" below) | Hand-edited docs can drift | Generate more docs from artifacts |

## 2. Scientific-evidence quality
| # | Requirement | Status | Evidence / outcome | Limitation |
|---|---|---|---|---|
| S1 | Independent holdout | **FAIL** | none exists; study is labelled a revised historical evaluation | all blocks inspected; a new config hash is not new data |
| S2 | Fair comparison on identical origins, universe and actuals | PASS | `weekly.long_frame` scores all models on the same rows; MASE exclusions identical for all | — |
| S3 | Beats the strongest baseline convincingly | **FAIL** | LightGBM +2.5% WAPE but fails MASE and segment guards; `mean4` selected | origin-to-origin WAPE spread is reported (std ≈ 0.07–0.08) but is not a significance test: the gain cannot be called real or noise |
| S4 | Metrics appropriate to sparse demand | PARTIAL | WAPE, MAE, RMSE, bias, MASE, RMSSE, all-zero reference, segments | no decision-cost (inventory) objective |
| S5 | Aggregation effects explained | PASS | L1 vs L2 vs L3 table; text states cancellation | — |
| S6 | Probabilistic forecasts calibrated | **FAIL (reported as experimental)** | coverage 0.915 all rows / 0.871 active (nominal 0.8), raw crossing 2.1% > 2% limit; conformal step degenerate | intervals not attached to serving |
| S7 | Statistical significance | not claimed | origin variability reported only | correlated series/overlapping information |
| S8 | Daily vs weekly contrast | PARTIAL | `docs/results_daily_fv2.md` (corrected daily comparison) | different units/universe; not comparable |

## 3. Readiness for a specified production environment
No environment, SLO, data feed, security model or on-call process was specified, so this section cannot be rated. Required before any production claim: a real data feed and late-actuals policy, authentication beyond a shared key (OIDC/mTLS behind a gateway), TLS, load and failure-injection tests, an independent prospective evaluation, runbooks and alert routing, backup/restore tests, and a cost-based evaluation of the decision the forecast supports.

## Historical vs current (contradictions resolved)
| Earlier statement | Status now |
|---|---|
| "No model approved" (daily v1 rejected) vs "v2 approved" | Daily v1 was rejected; daily v2 passed a post-hoc pooled-bias gate, rested on `fv1` price features (leaky) and is **revoked**. Currently approved: `retailfc_weekly` v1 = `mean4` (rule `weekly_v2`). |
| "Real-data results pending" vs real results reported | Real-data results exist for weekly v2 (generated) and the corrected daily comparison; historical daily reports are kept and bannered as superseded. |
| Historical v1 configuration vs current v2 configuration | `configs/online_retail.yaml` is the daily-study config (v2 settings); weekly uses `configs/weekly_v2.yaml`; each report states its config. |
| Previous service verification vs current Compose state | The stack was torn down, rebuilt from the current code and re-probed on 2026-10-05; earlier verification is superseded. |
