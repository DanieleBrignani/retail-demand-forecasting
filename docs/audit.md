# Audit ledger

Superseded by [`PRODUCTION_READINESS.md`](../PRODUCTION_READINESS.md), which holds the per-category review and the evidence.

> **2026-10-06 correction:** an external review found a unit-overflow defect (`int16`); see [`audit_int16_overflow.md`](audit_int16_overflow.md). Everything below that concerns real-data numbers was re-executed after the fix; the test count is from the final run.

Executed checks (2026-10-05/06, after the remediation; details in `PRODUCTION_READINESS.md`):
* `ruff check src tests scripts monitoring` and `ruff format --check`: clean.
* `pytest -q` with real PostgreSQL 16 (`RETAILFC_TEST_DSN`) and dbt-postgres (`DBT_BIN`): **186 passed, 0 failed, 0 skipped** (2 min 14 s).
* Real dataset: weekly v2 backtest executed twice (identical pooled metrics); `dbt build` on the corrected real data PASS=35 in 61 s; weekly lifecycle replay (register → approve → forecast → replay → reconcile twice → monitor) with database WAPE equal to the backtest WAPE; corrected daily comparison (`docs/results_daily_fv2.md`).
* Synthetic end-to-end demo (`scripts/demo_fixture.sh`): the daily chain ran to completion in a fresh location; its first weekly step failed on too-early origins in the smoke config (a real defect: now a guarded error with a test), the weekly steps were then re-run successfully (selected `mean13` on the fixture, registered, approved, forecast, monitored). The whole script was not re-executed in one go after that fix.
* Docker Compose: rebuilt from the current code on a fresh volume and probed on 2026-10-05; the host Docker daemon was restarted afterwards and the stack was brought back up (see the final report for its state).
* GitHub Actions: never run (no remote). No load benchmark of the weekly API was run.

Historical (superseded): the 133-test count, the daily real-data run and the "dbt 37/37" correction below refer to the earlier daily study.

Correction kept from the earlier ledger: "dbt 37/37" cited from the demo script was invalid because dbt pointed at a different database; dbt is pointed at the pipeline's database via `scripts/dbt_env.py`.
