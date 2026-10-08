# Recruiter summary (≈ 2 minutes)

**1. Business question.** Can product-level *weekly recorded sales* be forecast more accurately and reliably than simple and intermittent-demand baselines, with reproducible model selection and a working forecast lifecycle?

**2. Dataset and target.** UCI Online Retail II (one UK online retailer, Dec 2009–Dec 2011, 1.07M invoice lines). Target = gross positive recorded sales per product × channel (UK / international) per 7-day block — *not* unconstrained demand (no stock or assortment data). 9,414 series, 73% of weekly observations are zero.

**3. Architecture.** Ingestion with immutable dataset versions → PostgreSQL + dbt (35 checks, with negative controls) → leakage-safe features → rolling-origin backtests → MLflow registry with rule-based approval/revocation → append-only forecast storage → actuals replay and reconciliation → Prometheus/Grafana → FastAPI (API-key auth) → Docker Compose. A small synthetic fixture exercises the whole path in minutes.

**4. Main engineering challenge.** Making the information boundary real from the *source transactions* onward (not just on the last matrix), and keeping daily and weekly forecasts, versions and registry lines from being confused in storage, reconciliation and the API.

**5. Main scientific challenge.** Zero-inflated demand: an all-zero forecast has WAPE exactly 1 and wins absolute-error metrics on sparse series, so every result is shown next to that reference, with MASE/RMSSE, segments, origin variability and guardrails instead of a single score.

**6. Important design decision.** Selection is mechanical and pre-registered (`docs/weekly_protocol_v2.md`): a complex model must pass every guardrail *and* beat the strongest baseline; otherwise the baseline is selected. Thresholds are not changed after seeing results.

**7. Verified result.** On 10 rolling origins the global LightGBM improves pooled WAPE by ~2.5% over the 4-week mean (0.8251 vs 0.8465) but fails the series-level MASE guard and degrades low/mid-volume segments, so the **4-week mean was selected** and run through the full lifecycle (registered, approved by rule, forecast, replayed, reconciled to the same WAPE as the backtest, served by the API). Source: `docs/results_weekly_v2.md`.

**8. Failures discovered and corrected.** (a) An `int16` read silently wrapped two huge real orders (74,215 → 8,679), losing 131,072 units from every published number; found by an external review, fixed, tested, all results regenerated (`docs/audit_int16_overflow.md`). (b) A price feature leaked future transactions of its own block (a future price change of 10→100 moved an earlier feature 10→55); the original tests missed it because they mutated an already-aggregated matrix. Fixed with completed-block prices, regression tests built from transactions, a revoked approval and superseded reports (`docs/audit_price_leakage.md`).

**9. Trade-off.** Weekly granularity makes the problem learnable but is coarser than daily; simplicity (a moving average) beat complexity under the declared rules; weekly and daily numbers are not comparable.

**10. Current limitations.** Revised historical evaluation on already-inspected data — no independent holdout; one dataset covering about two years (two full Decembers and the 2011 run-up); cold-start products not evaluated; prediction intervals experimental and not served; CI workflow never executed (no remote); no production traffic or load testing. **Not claimed: production readiness or model superiority.**
