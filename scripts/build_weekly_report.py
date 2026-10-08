"""Render docs/results_weekly_v2.md from the artifacts written by `retailfc weekly backtest` (no number is typed by hand).

python scripts/build_weekly_report.py <experiment_dir> <out.md> [--repeat <second_experiment_dir>]
"""

import json
import sys
from pathlib import Path

import pandas as pd


def md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    df = df.copy()
    cols = [df.index.name or "", *map(str, df.columns)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        cells = [" ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)]
        for v in r:
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(exp: str, out: str, repeat: str | None = None) -> None:
    e = Path(exp)
    man = json.loads((e / "manifest.json").read_text())
    sel = json.loads((e / "selection.json").read_text())
    L1 = pd.read_csv(e / "L1.csv", index_col=0)
    lv = pd.read_csv(e / "levels.csv", index_col=[0, 1]).unstack()
    lv.columns = [f"{a} {b}" for a, b in lv.columns]
    var = pd.read_csv(e / "origin_variability.csv", index_col=0)
    cov = pd.read_csv(e / "universe_coverage.csv").set_index("origin")
    seg = pd.read_csv(e / "segments.csv")
    segp = seg.pivot(index=["segment_type", "segment"], columns="model", values="wape")
    hor = pd.read_csv(e / "by_horizon.csv").pivot(index="h", columns="model", values="wape")
    itv = pd.read_csv(e / "intervals.csv")
    iv_over = itv[itv.by == "overall"].set_index(["model", "scope"])[
        ["empirical_coverage", "mean_width", "pinball", "raw_crossing_rate"]
    ]
    iv_vol = itv[
        (itv.by == "volume") & (itv.scope == "all") & itv.model.isin(["lgbm_w", "empirical_q"])
    ].pivot(index="key", columns="model", values="empirical_coverage")
    fb = pd.read_csv(e / "fallback_rate.csv")
    checks = []
    for m, c in sel["candidates"].items():
        for x in c["checks"]:
            checks.append(f"| {m} | {x['check']} | {'pass' if x['passed'] else '**FAIL**'} | {x['detail']} |")
    t = f"""# Weekly study v2 — results (protocol `{man["protocol_version"]}`)

> **Generated** by `python scripts/build_weekly_report.py` from `{e.name}` (manifest below). Nothing in this file is typed by hand.
> **Status: revised historical evaluation on already-inspected data — not an independent holdout.** A new configuration hash does not create independent test data.
> The protocol is [`weekly_protocol_v2.md`](weekly_protocol_v2.md); thresholds were partly written after weekly v1 results (disclosed there).
> Weekly and daily metrics are **not comparable** (different units, aggregation and eligible universe).

## Manifest
* dataset `{man["dataset_version"]}`, weekly feature version `{man["weekly_feature_version"]}`, config SHA-256 `{man["config_sha256"][:16]}…`, seed {man["seed"]}, tuning trials {man["tuning_trials"]}.
* {man["n_blocks"]} complete 7-day blocks (dataset blocks, Tue–Mon), origins {man["origins"]}, horizons {man["horizons"]} blocks, runtime {man["runtime_seconds"]:.0f}s.
* environment: {json.dumps(man["environment"])}
* command: `{man["command"]}`

## Selection (mechanical, `retailfc.weekly.select_model`)
**Selected: `{sel["selected"]}`** (strongest qualified baseline: `{sel["strongest_baseline"]}`; qualified: {sel["qualified"]}). Every check of every candidate:

| candidate | check | result | detail |
|---|---|---|---|
{chr(10).join(checks)}

## L1 (product × channel) pooled metrics, eligible universe, 10 origins × 4 horizons
`all_zero` is a reference only (never selectable). MASE/RMSSE exclude, for every model, rows whose pre-origin history is constant (column `n_mase_excluded`).
{md(L1[["wape", "mae", "rmse", "bias_pct", "mase", "rmsse", "n", "n_mase_excluded"]])}

Aggregated levels (bottom-up sums of eligible L1 forecasts; errors can cancel — this does not imply accurate individual products):
{md(lv)}

WAPE by horizon (blocks):
{md(hor)}

## Origin-level variability (WAPE per origin) — no significance claim
{md(var)}

## Segments (derived from past data only) — WAPE
{md(segp)}

## Universe, cold start and fallback
Products first sold after the origin are **not scored** (cold start); their realised demand share is below. Fallback rate = share of scored rows whose channel had no earlier sale (`no_channel_history`): {fb.iloc[0, 0]:.4f}; minimum forecast coverage over models {fb.iloc[0, 1]:.0%} (measured: share of eligible series-rows with a finite forecast).
{md(cov[["n_eligible", "n_cold_start", "target_units_eligible", "target_units_cold_start", "share_units_cold_start"]])}

## Prediction intervals (80% nominal) — experimental
Overall and active-series-only coverage, width, pinball and the raw quantile-crossing rate **before** the monotone correction:
{md(iv_over)}

Coverage by past-volume segment (all rows). Aggregate coverage hides the split: the `inactive` segment is almost always covered by a zero-width interval at 0, while active segments are covered less:
{md(iv_vol)}
The conformal correction from the preceding origin is exactly 0 on this data (more than 80% of the conformity scores are exactly 0 because most actuals and quantiles are 0), so `lgbm_w_cal` equals `lgbm_w`: the calibration step is uninformative on zero-heavy data.
Intervals are therefore **not attached to serving**.
"""
    if repeat:
        r = Path(repeat)
        a, b = pd.read_csv(e / "L1.csv", index_col=0), pd.read_csv(r / "L1.csv", index_col=0)
        d = (
            b[["wape", "mae", "rmse", "bias_pct", "mase"]] - a[["wape", "mae", "rmse", "bias_pct", "mase"]]
        ).abs()
        sel2 = json.loads((r / "selection.json").read_text())
        t += f"""
## Repetition (same configuration, same data, second execution: `{r.name}`)
Maximum absolute difference of each pooled metric between the two executions (baselines are deterministic; LightGBM is multi-threaded, so bitwise equality is **not promised** — here the two runs happened to coincide on the same machine with the same seed and thread count):
{md(d, "{:.2e}")}

Selected model in the repetition: `{sel2["selected"]}` (first run: `{sel["selected"]}`). Documented tolerance if a rerun differs: 1e-9 for baselines, 5e-3 absolute on pooled metrics for `lgbm_w`.
"""
    Path(out).write_text(t, encoding="utf-8")


if __name__ == "__main__":
    args = sys.argv[1:]
    rep = args[args.index("--repeat") + 1] if "--repeat" in args else None
    main(args[0], args[1], rep)
