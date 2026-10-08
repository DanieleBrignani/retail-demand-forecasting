"""Render docs/results_daily_fv2.md from the artifacts of `retailfc backtest variants` (no number typed by hand).

python scripts/build_daily_report.py <variants_dir> <out.md>
"""

import json
import sys
from pathlib import Path

import pandas as pd


def main(d: str, out: str) -> None:
    p = Path(d)
    s = pd.read_csv(p / "summary.csv")
    allr = s[s.origin == "all"].set_index("model").drop(columns="origin").sort_values("wape")
    per = s[s.origin != "all"].pivot(index="model", columns="origin", values="wape")
    cov = pd.read_csv(p / "universe_coverage.csv").set_index("origin")

    def md(df, fmt="{:.4f}"):
        cols = [df.index.name or "", *map(str, df.columns)]
        rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for i, r in df.iterrows():
            rows.append(
                "| "
                + " | ".join([str(i), *[fmt.format(v) if isinstance(v, float) else str(v) for v in r]])
                + " |"
            )
        return "\n".join(rows)

    meta = json.loads((p / "run_meta.json").read_text()) if (p / "run_meta.json").exists() else {}
    t = f"""# Daily comparison under the corrected pipeline (`fv2` prices + origin-appropriate universe)

> **Generated** by `python scripts/build_daily_report.py` from `{p.parent.name}/{p.name}`. Status: **revised historical evaluation on already-inspected data** (origins 627, 655, 683, 711 are the original validation windows and the consumed holdout) — not an independent holdout.
> Command: `retailfc backtest variants --origins 627,655,683,711 --version retail-369d316281ea --config configs/online_retail.yaml --tag daily-fv2-universe`.
> Daily metrics are **not comparable** with the weekly ones (different units, aggregation and eligible universe). LightGBM variants use feature set D under `fv2`; `v2_all_offset` = all series + level offset (the former "v2" recipe, now without the price look-ahead).
> `all_zero_reference` = forecasting zero everywhere (WAPE is exactly 1 by definition).

## Pooled over the four origins (eligible universe at each origin; identical rows for every model)
{md(allr[["wape", "mae", "rmse", "bias_pct", "n"]])}

## WAPE per origin
{md(per)}

## Universe at each origin (cold-start products are not scored)
{md(cov[["n_eligible", "n_not_yet_launched", "share_demand_not_yet_launched"]])}

## Reading (interpretation, not part of any protocol)
* No model beats the all-zero forecast on WAPE or MAE at daily granularity; this is expected for ~92% zeros (the median is 0). The lowest RMSE is reached by the LightGBM variants.
* Variants that reduce bias (all series, level offset) raise WAPE and MAE: bias and absolute error trade off on this data.
* This is why the main candidate experiment is weekly ([`results_weekly_v2.md`](results_weekly_v2.md)); the daily line is not served and its former approval is revoked.
* Seeds/params: {json.dumps(meta.get("seed", "see config"))}; `training.max_series`/`level_offset` per variant are in `retailfc.experiments.DEFAULT_VARIANTS`.
"""
    Path(out).write_text(t, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
