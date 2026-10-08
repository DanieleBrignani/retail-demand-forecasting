# Five-minute demonstration

Prerequisites (one-off): Python 3.13 venv with `pip install -r requirements.lock && pip install --no-deps -e .[dev]`, Docker Desktop, a `.env` (copy `.env.example`, replace the placeholder secrets — the file is git-ignored).
Real-data commands need `online_retail_II.xlsx` (UCI, CC BY 4.0) and a working folder **outside synced folders** (`WORK`).

## Minute 0–1 — the problem and the honest framing
Show `RECRUITER_SUMMARY.md`. Say: weekly product-level recorded sales; zero-inflated data; a revised historical evaluation (nothing is an independent holdout); a baseline was selected because the rules said so.

## Minute 1–2 — a defect found and fixed (the strongest story)
```bash
python -m pytest tests/test_price_availability.py -q
```
Open `docs/audit_price_leakage.md`: the future day-4 price changed an origin-day-1 feature (10 → 55); the old tests mutated an already-aggregated matrix; the new ones start from transaction rows.

## Minute 2–3 — the evidence and the selection
Open `docs/results_weekly_v2.md`: the all-zero reference, the baselines, LightGBM (better WAPE, worse MASE), the table of guardrail checks that rejected it, and why `mean4` was selected. Mention what is *not* shown (no independent holdout, intervals not attached).

## Minute 3–4 — the lifecycle and the API (stack must be running)
```bash
docker compose up -d --wait
curl -s localhost:8010/health/ready
KEY=$(grep ^RETAILFC_API_KEYS .env | cut -d= -f2 | cut -d, -f1)
curl -s -X POST localhost:8010/forecast -H "content-type: application/json" -H "x-api-key: $KEY" \
     -d '{"item_id":"RETAIL_1_10002","store_id":"INT_1","horizon":4}'            # weekly: horizon_unit=week, explicit periods
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8010/forecast -H "content-type: application/json" \
     -d '{"item_id":"RETAIL_1_10002","store_id":"INT_1","horizon":4}'            # 401 without a key
curl -s -X POST localhost:8010/forecast -H "content-type: application/json" -H "x-api-key: $KEY" \
     -d '{"item_id":"RETAIL_1_10002","store_id":"INT_1","horizon":5}'            # 422 horizon_out_of_range
```
Say: the API retrieves stored forecasts (it does not run inference); batches are append-only; the daily model's approval was revoked, so the daily line answers `503 model_unavailable`.

## Minute 4–5 — monitoring and the limits
Open Grafana (`http://localhost:3005`, admin password from `.env`) → "Forecast quality" dashboard; Prometheus targets at `http://localhost:9095/targets`. State the limits: replayed historical data (so "freshness" alerts are meaningless), no CI run, no production traffic.

---
## The three separate entry points
| | command | what it proves | outputs |
|---|---|---|---|
| **A. Synthetic smoke/integration** (minutes) | `bash scripts/demo_fixture.sh` (needs the dev PostgreSQL of `docs/operations.md`; set `RETAILFC_DSN`) | mechanics only — **no claim about forecasting quality** | `artifacts_demo/`, `data_demo/` (git-ignored) |
| **B. Real-data scientific experiment** | `bash scripts/run_retail_case_study.sh` (daily study) and `retailfc weekly backtest --version <V> --config configs/weekly_v2.yaml` (weekly, frozen protocol) then `python scripts/build_weekly_report.py <exp_dir> docs/results_weekly_v2.md` | metrics, selection, manifests | `<ARTIFACTS>/experiments/weekly-v2-<V>/` |
| **C. Real-data operational replay** | `WORK=… V=… REVOKE_DAILY=1 bash scripts/run_weekly_operations.sh` | registration → approval → batch → replay → reconcile → monitor | PostgreSQL `app` schema, MLflow registry |
