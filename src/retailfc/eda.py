"""Reproducible EDA. Uses ONLY the training period (days <= first validation origin) so that findings that drive
modelling decisions never see validation or holdout data. Interpretations are generated from the computed numbers.

Limitations (repeated in every summary): zero sales do not prove zero demand; sales do not reveal stock-outs or
inventory; the first positive sale is not necessarily a launch; associations with price/events are not causal.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .panel import Panel  # noqa: E402
from .protocol import make_origins  # noqa: E402
from .segments import fit_volume_edges, segment_frame  # noqa: E402

LIMITS = [
    "Zero sales do not prove zero demand.",
    "Sales do not reveal stock-outs or inventory.",
    "First positive sale does not necessarily identify product launch.",
    "Correlation with price or events does not establish causality.",
]


def _save(fig, out: Path, name: str) -> str:
    p = out / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def run_eda(panel: Panel, cfg: dict, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    origins = make_origins(panel.n_days, cfg)
    T = origins.validation[0]  # training period = days 1..T
    S = panel.sales[:, :T].astype(np.float32)
    cal = panel.calendar.iloc[:T].copy()
    ids = panel.ids
    summary: dict = {
        "training_period_days": int(T),
        "training_period": [
            str(pd.Timestamp(panel.dates[0]).date()),
            str(pd.Timestamp(panel.dates[T - 1]).date()),
        ],
        "n_series": panel.n_series,
        "limitations": LIMITS,
        "charts": {},
        "findings": [],
    }
    F = summary["findings"]

    # 1. sales distribution + zero frequency
    zero_share = float((S == 0).mean())
    pos = S[S > 0]
    q = {f"p{p}": float(np.quantile(S, p / 100)) for p in (50, 90, 99, 99.9)}
    dist = pd.DataFrame(
        {
            "value": ["mean", "zero_share", "mean_positive", "max", *q],
            "stat": [
                float(S.mean()),
                zero_share,
                float(pos.mean()) if pos.size else np.nan,
                float(S.max()),
                *q.values(),
            ],
        }
    )
    dist.to_csv(out / "sales_distribution.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(np.clip(S[S > 0], 0, np.quantile(pos, 0.99) if pos.size else 1), bins=40)
    ax[0].set(title="Positive daily sales (clipped at p99)", xlabel="units", ylabel="rows")
    zs = (S == 0).mean(axis=1)
    ax[1].hist(zs, bins=40)
    ax[1].set(title="Zero-sales share per series", xlabel="share of days with zero sales")
    summary["charts"]["distribution"] = _save(fig, out, "sales_distribution")
    F.append(
        f"{zero_share:.1%} of item-store-days have zero sales; median series is zero on {np.median(zs):.1%} of days. "
        f"Zero ≠ zero demand (stock-outs/inactivity are unobserved)."
    )

    # 2. weekly + longer seasonality
    tot = S.sum(axis=0)
    dow = cal["wday"].to_numpy()
    by_dow = pd.Series(tot).groupby(dow).mean()
    by_dow.rename_axis("wday").rename("mean_total_units").to_csv(out / "seasonality_weekday.csv")
    cal["tot"] = tot
    by_month = cal.groupby("month")["tot"].mean()
    by_year = cal.groupby("year")["tot"].mean()
    by_month.to_csv(out / "seasonality_month.csv")
    by_year.to_csv(out / "level_by_year.csv")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    by_dow.plot.bar(ax=ax[0], title="Mean daily total by weekday (wday 1=Sat)")
    by_month.plot.bar(ax=ax[1], title="Mean daily total by month")
    ax[2].plot(pd.to_datetime(cal["date"]), pd.Series(tot).rolling(28, min_periods=1).mean())
    ax[2].set_title("28-day rolling mean of total units")
    summary["charts"]["seasonality"] = _save(fig, out, "seasonality")
    peak, trough = by_dow.idxmax(), by_dow.idxmin()
    F.append(
        f"Weekly pattern: weekday code {peak} has the highest mean daily total, {trough} the lowest "
        f"(peak/trough ratio {by_dow.max() / by_dow.min():.2f})."
    )

    # 3. variation by state / store / category
    d = ids.copy()
    d["units"] = S.sum(axis=1)
    d["zero_share"] = zs
    for col in ("state_id", "store_id", "cat_id"):
        g = d.groupby(col).agg(
            units=("units", "sum"), series=("units", "size"), mean_zero_share=("zero_share", "mean")
        )
        g["units_per_series_day"] = g["units"] / (g["series"] * T)
        g.to_csv(out / f"by_{col}.csv")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, col in zip(ax, ("state_id", "store_id", "cat_id"), strict=True):
        d.groupby(col)["units"].sum().div(1).plot.bar(ax=a, title=f"Total units by {col}")
    summary["charts"]["segments"] = _save(fig, out, "by_segment")

    # 4. intermittency + volume segments (training-period, origin T)
    edges = fit_volume_edges(panel, T, cfg)
    seg = segment_frame(panel, T, cfg, edges)
    cnt = seg["intermittency_segment"].value_counts(normalize=True)
    cnt.to_csv(out / "intermittency_classes.csv", header=["share_of_series"])
    seg["volume_segment"].value_counts(normalize=True).to_csv(
        out / "volume_segments.csv", header=["share_of_series"]
    )
    fig, ax = plt.subplots(figsize=(5, 3.4))
    cnt.plot.bar(ax=ax, title="Intermittency classes (ADI/CV², trailing year)")
    summary["charts"]["intermittency"] = _save(fig, out, "intermittency")
    F.append(
        "Demand classes (Syntetos-Boylan, last 364 days of training): "
        + ", ".join(f"{k} {v:.1%}" for k, v in cnt.items())
        + "."
    )
    summary["volume_edges"] = edges

    # 5. price coverage and variation (listed prices only; absent stays absent)
    w_end = int(panel.day_week_idx[T - 1])
    P = panel.price_weekly[:, : w_end + 1]
    cover = (~np.isnan(P)).mean()
    first_listed = np.where((~np.isnan(P)).any(axis=1), np.argmax(~np.isnan(P), axis=1), -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.nanstd(P, axis=1) / np.nanmean(P, axis=1)
        chg = np.nanmean(np.abs(np.diff(P, axis=1)) > 1e-9, axis=1)
    pd.DataFrame(
        {"cat_id": ids.cat_id, "cv": cv, "weekly_change_rate": chg, "coverage": (~np.isnan(P)).mean(axis=1)}
    ).groupby("cat_id").mean().to_csv(out / "price_variation_by_cat.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(np.nan_to_num(cv, nan=0), bins=40)
    ax[0].set(title="Within-series price CV", xlabel="CV")
    ax[1].hist((~np.isnan(P)).mean(axis=1), bins=40)
    ax[1].set(title="Price coverage per series (listed weeks)")
    summary["charts"]["prices"] = _save(fig, out, "prices")
    F.append(
        f"Price listed for {cover:.1%} of (series, week) cells up to the training end; {(first_listed > 0).mean():.1%} of series "
        f"are first listed after the first week. Missing prices are left missing (never 0)."
    )

    # 6. holiday-event associations (descriptive; within-series normalised)
    scale = np.maximum(S.mean(axis=1, keepdims=True), 1e-6)
    rel = S / scale  # sales relative to each series' own mean
    day_rel = rel.mean(axis=0)  # mean relative demand per day
    ev = cal["event_type_1"].fillna("none").astype(str).to_numpy()
    et = pd.Series(day_rel).groupby(ev).agg(["mean", "size"])
    et.to_csv(out / "event_association.csv")
    base = et.loc["none", "mean"] if "none" in et.index else np.nan
    F.append(
        "Event days vs non-event days (relative demand, descriptive only): "
        + (
            ", ".join(
                f"{k} {v / base:.2f}x" for k, v in et["mean"].items() if k != "none" and np.isfinite(base)
            )
            or "no event days in the training period"
        )
        + ". Not causal."
    )
    fig, ax = plt.subplots(figsize=(5, 3.4))
    if np.isfinite(base) and len(et) > 1:
        (et["mean"] / base).plot.bar(ax=ax, title="Relative demand by event type (none = 1)")
    else:
        ax.axis("off")
    summary["charts"]["events"] = _save(fig, out, "events")

    # 7. first positive sale, inactive periods
    pos_mask = S > 0
    first_pos = np.where(pos_mask.any(axis=1), np.argmax(pos_mask, axis=1) + 1, 0)
    lead_zero = float((first_pos > 28).mean())
    # zero runs >= 28 days after first sale
    long_zero = 0.0
    run_share = []
    for i in range(0, S.shape[0], 5000):
        blk = (S[i : i + 5000] == 0).astype(np.int8)
        run = np.zeros(blk.shape[0])
        cur = np.zeros(blk.shape[0])
        for t in range(blk.shape[1]):
            cur = (cur + 1) * blk[:, t]
            run = np.maximum(run, cur)
        run_share.append(run)
    runs = np.concatenate(run_share)
    long_zero = float((runs >= 28).mean())
    pd.DataFrame({"first_positive_day": first_pos}).describe().to_csv(out / "first_positive_sale.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(first_pos[first_pos > 0], bins=40)
    ax[0].set(title="Day of first positive sale", xlabel="day number")
    ax[1].hist(np.minimum(runs, 365), bins=40)
    ax[1].set(title="Longest zero-sales run per series (days, capped 365)")
    summary["charts"]["inactivity"] = _save(fig, out, "first_sale_inactivity")
    F.append(
        f"{lead_zero:.1%} of series have no sale in their first 28 days (late first positive sale), and {long_zero:.1%} contain a zero run of "
        f">=28 days: apparent inactivity that may reflect availability rather than demand."
    )

    # 8. missing observations / source coverage
    summary["coverage"] = {
        "missing_sales_observations": 0,
        "calendar_days": int(len(panel.calendar)),
        "observed_sales_days": int(panel.n_days),
        "future_calendar_days_without_sales": int(len(panel.calendar) - panel.n_days),
        "price_cell_coverage_training": float(cover),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md = [
        "# EDA summary (auto-generated; training period only)\n",
        f"Training period: {summary['training_period'][0]} .. {summary['training_period'][1]} "
        f"({T} days, {panel.n_series} series). Dataset `{panel.dataset_version}`.\n",
        "## Findings (computed)\n",
        *[f"- {x}" for x in F],
        "\n## Limitations\n",
        *[f"- {x}" for x in LIMITS],
        "\n## Charts\n",
        *[f"![{k}]({v})" for k, v in summary["charts"].items()],
    ]
    (out / "eda_summary.md").write_text("\n".join(md), encoding="utf-8")
    return {"out": str(out), "findings": F, "charts": summary["charts"]}
