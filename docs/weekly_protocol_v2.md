# Weekly protocol v2 (`weekly-v2`) — frozen before the v2 run

Configuration: [`configs/weekly_v2.yaml`](../configs/weekly_v2.yaml) (its SHA-256 is recorded in every manifest). Supersedes [`weekly_protocol.md`](weekly_protocol.md) (v1), whose results stay published in
[`results_weekly.md`](results_weekly.md) with the status "exploratory, superseded" (reasons in §9).

## 0. Status of the evidence — read first
* **This is a revised historical evaluation.** Every block of the dataset has already been inspected (daily validation windows, the consumed daily holdout, and the weekly v1 run on the same ten origins).
  Nothing here is an untouched holdout, and a new configuration hash does not create independent data. No real observations were added.
* The selection thresholds below were written after seeing the weekly v1 results. Two of them are **new** in v2 (MASE ratio guard, per-origin bias guard); the others are copied from the daily promotion rules. This weakens the
  independence of the outcome and is disclosed here, in the manifest and in the results.
* No claim of statistical significance is made (correlated series, overlapping information across origins).

## 1. Business question
Can product-level **weekly recorded sales** be forecast more accurately and reliably than simple and intermittent-demand baselines, with reproducible model selection and a functioning forecast lifecycle?
The target is *gross positive recorded sales* (units); it is **not** unconstrained demand (no inventory, assortment or lost-sales information exists).

## 2. Data, target and calendar
* Source: Online Retail II, adapted by `retailfc.adapters.online_retail` (dataset version `retail-369d316281ea`). Cancellations/returns are excluded, not netted; non-product codes, non-positive quantity or price are excluded (counts in `adapter_report.json`);
  identical-looking lines are **kept** (they can be legitimate); rows without a product code are impossible after the `^\d{5}[A-Za-z]{0,2}$` filter and rows with a missing price are excluded.
* Levels: L1 product×channel (UK_1 = United Kingdom, INT_1 = all other countries; these are channels, not stores; there is no category hierarchy); L2 channel totals; L3 grand total. L2/L3 are bottom-up sums of eligible L1 forecasts.
* **Calendar decision.** Weeks are the dataset's own 7-day blocks starting on the first observed day (Tuesday, 2009-12-01 … Monday), *not* ISO weeks. Reasons: (i) the same blocks define the price blocks and the daily-to-weekly mapping used by the lifecycle;
  (ii) ISO weeks would make the first and last weeks partial; (iii) a consistent calendar for all models. 105 complete blocks; the 106th (4 days) is dropped. Weekly and daily metrics are **not comparable** (different units and aggregation); no table puts them side by side.
* Zeros are **zero recorded sales** under the assumption that the export is complete. They are not proven closures or proven zero demand. Four situations are kept apart: zero recorded sales; missing source coverage (not observable here);
  unknown assortment status (not observable); insufficient history (reported through history-length segments and the fallback rate).

## 3. Eligible series (origin-appropriate universe)
A product is part of the universe at origin *t* iff it has ≥ 1 recorded sale (any channel) in blocks ≤ *t* (no external catalogue exists, none is invented). Both its channels are eligible; a channel with no sale yet is scored with an explicit
`no_channel_history` fallback flag. Products first sold after *t* are **cold start**: not scored in L1 and not in the training rows of any origin < their launch, but their realised demand share is reported for every origin.
Evaluation of genuinely unseen products is therefore *not* part of this study; it is quantified (share of demand) and left open.

## 4. Origins, horizons, splits
Ten origins *t* ∈ {65, 69, …, 101}; horizons 1–4 blocks. Each model uses blocks ≤ *t* only. No random split. LightGBM training rows are all (*t′*, *h*) with 26 ≤ *t′*, *t′*+*h* ≤ *t*; early stopping uses the last 4 target blocks (training targets ≤ *t* − 4),
so no label from the evaluation window enters training. Segment thresholds and MASE denominators use data ≤ *t* only.

## 5. Models
Reference only: `all_zero`. Baselines: `naive`, `mean4`, `mean13`, `seasonal_naive52` (needs 52 earlier blocks; all origins qualify), `croston_sba`, `tsb`. Candidate: global direct `lgbm_w` (Tweedie, level offset `log(mean13+0.1)`,
features listed in `retailfc.weekly.FEATURES`, **no price or event features** — the weekly study never reads prices, so the daily price look-ahead defect (see `docs/audit_price_leakage.md`) does not affect it; this is verified by a test).
Tuning budget: **zero trials** (fixed recipe). The all-zero forecast wins absolute-error metrics on very sparse demand because the optimal point forecast under absolute error is the median, which is 0 when most outcomes are 0;
that says nothing about an inventory policy, which would need an asymmetric cost objective that this study does not evaluate.

## 6. Metrics (all computed from stored predictions; formulas in `retailfc.weekly`)
Pooled WAPE, MAE, RMSE, signed bias (positive = over-forecast); **MASE** (per row |e| / mean |Δ₁| of the series' own history ≤ *t*, averaged over series-origin pairs, h-averaged inside the pair) and **RMSSE** (same with squared errors);
series with a constant/all-zero history have denominator 0 and are **excluded from MASE/RMSSE for every model** (count reported; no epsilon). Forecast coverage = share of eligible series-rows with a forecast; fallback rate = share flagged `no_channel_history`.
Reported at L1, L2, L3, by horizon, channel, volume segment, intermittency segment and history length, per origin (mean, std, min, max, wins over the strongest baseline). Aggregated levels can cancel errors and do not imply accurate product forecasts.
Intervals (80%, quantiles 0.1/0.5/0.9): nominal vs empirical coverage (all rows **and** active series only), mean width, pinball loss, raw quantile-crossing rate **before** the monotone correction (policy: running maximum over quantiles), by horizon and segment;
calibration from the preceding origin only (split conformal); intervals are attached to serving **only** if the thresholds in the config hold, otherwise they stay experimental.

## 7. Selection and promotion rules (mechanical: `retailfc.weekly.select_model`)
1. Qualified candidates are those passing **all** guardrails: coverage = 100%; WAPE ≤ 0.95 × all-zero WAPE; MASE ≤ 1.10 × the best baseline MASE; |pooled bias| ≤ 10% **and** |bias| ≤ 30% in ≥ 80% of origins (pooled bias must not hide opposing errors);
   no segment (carrying ≥ 2% of demand) whose WAPE is > 5% worse than the strongest baseline's in that segment; runtime limits.
2. The *strongest baseline* is the qualified baseline with the lowest pooled L1 WAPE. An ML candidate must additionally improve pooled WAPE by ≥ 1% over it.
3. Selected = lowest pooled WAPE among qualified candidates (ML candidates subject to rule 2). If no model is qualified, serving is **unavailable** and the rejection stands. Rule thresholds are not changed after results.
4. Incumbent comparison: none exists for the weekly line at first selection (bootstrap); later challengers must beat the incumbent on the same origins by the same margin.
Registry **registration** (a versioned artefact), **approval** (rules above), **historical evaluation** (this study) and **operational deployment** (forecast generation + serving) are separate records.

## 8. Reproducibility
`retailfc weekly backtest --config configs/weekly_v2.yaml` writes predictions (parquet), metric tables (CSV), `selection.json` and a manifest (config hash, dataset version, feature version, seeds, hardware/software, runtime). Re-running is expected to reproduce
baselines exactly and LightGBM within a documented tolerance (multi-threaded training is not bit-reproducible); the repetition and its tolerance are in the results.

## 9. Why v1 is superseded
v1 (`weekly.md`/`results_weekly.md`) scored all 9,414 series at every origin, including products that had not launched yet, and its LightGBM training rows included not-yet-launched series; it also reported only a pooled MASE without
segment, fallback, origin-variability, crossing or active-series-coverage diagnostics, and its interval step silently repaired crossing. Its numbers remain published for transparency.

## 10. Implementation corrections after the first v2 run (no threshold or config change)
A review after the first run found three implementation gaps; they were fixed and the whole protocol was **re-executed with the same configuration (same SHA-256)**: (1) weekly generation now refuses a configuration or dataset that differs from the registered one;
(2) approval now verifies dataset, protocol, config hash and the hashes of `selection.json`/`manifest.json`, not only the model name; (3) the `forecast_coverage` check was hard-coded to pass and the `runtime` check passed when a time was missing — both are now measured (a missing measurement fails the check).
The selection outcome and every pooled metric were unchanged (maximum absolute difference 0 against the pre-fix run, kept under `weekly-v2-prefix-run1-*`). Statements in earlier text about "one Christmas" were imprecise: the data cover two full Decembers and the run-up of a third.

## 11. Data defect found after publication (unit overflow) — protocol re-executed on corrected data
An external review found that daily units had been read as `int16` (two series-days wrapped, 131,072 units lost): [`audit_int16_overflow.md`](audit_int16_overflow.md). Configuration, thresholds and rules are unchanged (same SHA-256); the backtest, selection, registration (new registry version), approval, replay and reconciliation were re-executed and all reports regenerated.
Outputs computed on the corrupted panel are archived under `*-int16bug-*` and are superseded. The selected model did not change.
