"""Post-hoc analysis on stored experiment outputs: paired comparisons, error analysis, uncertainty, interpretability.

All comparisons are on matched (origin, series, horizon) observations. No significance is claimed: series within an
origin are cross-sectionally correlated and consecutive horizons/overlapping windows are serially dependent, so the
effective sample size is closer to the number of origins than to the number of rows. We therefore report per-origin
differences and descriptive win-rates only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import experiments as E
from . import protocol as P
from .backtesting import metrics as M
from .backtesting import runner as R
from .features.build import build_examples
from .features.spec import group_of
from .models.lgbm import GlobalLGBM, feature_groups_for
from .panel import Panel
from .uncertainty import QuantileBundle, interval_metrics


# ------------------------------------------------------------------ paired comparison
def paired_vs_reference(pred: pd.DataFrame, model: str, ref: str) -> dict:
    a = pred[pred.model_name == model].set_index(["origin", "series_idx", "h"])
    b = pred[pred.model_name == ref].set_index(["origin", "series_idx", "h"])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    rows = []
    for o in sorted(set(idx.get_level_values(0))):
        am, bm = a.xs(o, level=0), b.xs(o, level=0)
        wa = (am.forecast - am.actual).abs().sum() / am.actual.abs().sum()
        wb = (bm.forecast - bm.actual).abs().sum() / bm.actual.abs().sum()
        rows.append(
            {
                "origin": int(o),
                "wape_model": wa,
                "wape_ref": wb,
                "delta": wa - wb,
                "rel_delta": (wa - wb) / wb,
            }
        )
    per_origin = pd.DataFrame(rows)
    ea, eb = (a.forecast - a.actual).abs(), (b.forecast - b.actual).abs()
    series_better = ea.groupby(level="series_idx").sum() < eb.groupby(level="series_idx").sum()
    return {
        "model": model,
        "reference": ref,
        "n_matched": int(len(idx)),
        "per_origin": per_origin,
        "mean_delta_wape": float(per_origin["delta"].mean()),
        "min_delta": float(per_origin["delta"].min()),
        "max_delta": float(per_origin["delta"].max()),
        "origins_model_better": int((per_origin["delta"] < 0).sum()),
        "n_origins": int(len(per_origin)),
        "share_series_model_better_descriptive": float(series_better.mean()),
        "note": "descriptive; no p-values (dependent observations)",
    }


def error_tables(pred: pd.DataFrame, panel: Panel, model: str, out: Path, top: int = 25) -> dict:
    d = pred[pred.model_name == model].copy()
    d["err"] = d.forecast - d.actual
    d["abs_err"] = d.err.abs()
    ids = panel.ids.iloc[d.series_idx.to_numpy()].reset_index(drop=True)
    d = d.reset_index(drop=True)
    d["item_id"], d["store_id"] = ids.item_id, ids.store_id
    res = {}
    d.nlargest(top, "abs_err")[
        ["origin", "item_id", "store_id", "h", "target_day", "actual", "forecast", "err"]
    ].to_csv(out / f"largest_errors_{model}.csv", index=False)
    per = d.groupby(["series_idx", "origin"]).agg(bias=("err", "sum"), abs_a=("actual", "sum")).reset_index()
    sign = (
        per.assign(s=np.sign(per.bias))
        .groupby("series_idx")
        .agg(
            over=("s", lambda x: (x > 0).mean()),
            under=("s", lambda x: (x < 0).mean()),
            n_origins=("s", "size"),
        )
    )
    pers = sign[sign.n_origins >= 2]
    pers_over = pers[pers.over == 1.0].index
    pers_under = pers[pers.under == 1.0].index
    res["persistent_over_series"] = int(len(pers_over))
    res["persistent_under_series"] = int(len(pers_under))
    res["series_with_multiple_origins"] = int(len(pers))
    M.pooled(d, ["h"]).to_csv(out / f"horizon_degradation_{model}.csv")
    for col in ("volume_segment", "intermittency_segment"):
        M.pooled(d, [col]).to_csv(out / f"by_{col}_{model}.csv")
    fb = d.groupby(d.fallback_used.replace("", "none")).apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                **{k: float(v) for k, v in M.pooled(g).iloc[0][["wape", "mae", "bias_pct"]].items()},
            }
        ),
        include_groups=False,
    )
    fb.to_csv(out / f"fallback_performance_{model}.csv")
    # event periods and price-change periods (target-day attributes, known in the calendar / listed prices)
    cal = panel.calendar
    ev = cal["event_name_1"].notna().to_numpy()
    d["is_event"] = ev[d.target_day.to_numpy() - 1]
    M.pooled(d, ["is_event"]).to_csv(out / f"event_periods_{model}.csv")
    wk = panel.day_week_idx[d.target_day.to_numpy() - 1]
    P_ = panel.price_weekly
    prev = np.maximum(wk - 1, 0)
    cur_p = P_[d.series_idx.to_numpy(), wk]
    prev_p = P_[d.series_idx.to_numpy(), prev]
    d["price_change_week"] = np.where(
        np.isnan(cur_p) | np.isnan(prev_p),
        "unknown",
        np.where(np.abs(cur_p - prev_p) > 1e-9, "changed", "unchanged"),
    )
    M.pooled(d, ["price_change_week"]).to_csv(out / f"price_change_periods_{model}.csv")
    # history length at origin (insufficient history behaviour)
    first_pos = np.where((panel.sales > 0).any(axis=1), np.argmax(panel.sales > 0, axis=1) + 1, 10**6)
    hist = d.origin.to_numpy() - first_pos[d.series_idx.to_numpy()]
    d["history_bucket"] = pd.cut(
        hist, [-(10**7), 0, 28, 90, 10**7], labels=["no_history", "<=28d", "29-90d", ">90d"]
    )
    M.pooled(d, ["history_bucket"]).to_csv(out / f"history_length_{model}.csv")
    return res


# ------------------------------------------------------------------ uncertainty experiment
def run_uncertainty_experiment(panel: Panel, cfg: dict) -> dict:
    """Quantile LightGBM (0.1/0.5/0.9) raw vs CQR-calibrated. Calibration window strictly precedes the evaluated window.
    Validation windows only; the holdout is evaluated with the SAME pre-holdout calibration after the protocol lock."""
    exp_id = E.make_experiment_id(cfg, panel.dataset_version)
    out = E.experiment_dir(exp_id) / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)
    origins = P.make_origins(panel.n_days, cfg)
    val = list(origins.validation)
    vmeta = json.loads((E.experiment_dir(exp_id) / "validation" / "run_meta.json").read_text())
    sel_json = E.experiment_dir(exp_id) / "selection.json"
    key = (
        json.loads(sel_json.read_text())["selected_feature_set"]
        if sel_json.exists()
        else list(cfg["features"]["feature_sets"])[-1]
    )
    groups = feature_groups_for(cfg, key)
    params = vmeta.get("tuned_params") or None
    sel = E._train_series(panel, cfg)
    allp = np.arange(panel.n_series)
    H = list(range(1, 29))
    nominal = cfg["uncertainty"]["quantiles"][-1] - cfg["uncertainty"]["quantiles"][0]
    rows, qpred, offsets_log = [], [], {}
    # evaluate windows 2..k using calibration from the previous window (its bundle trained with data <= previous origin)
    for prev, cur in zip(val[:-1], val[1:], strict=True):
        point = GlobalLGBM.fit(panel, prev, cfg, groups, params=params, series_idx=sel)
        cal = QuantileBundle.fit(panel, prev, cfg, groups, sel, params, point)
        raw_c = cal.predict_raw(panel, prev, allp, H)
        y_c = panel.sales[raw_c.series_idx.to_numpy(), prev + raw_c.h.to_numpy() - 1].astype(float)
        cal.calibrate(raw_c, y_c, nominal)
        # evaluation uses a bundle trained with data <= cur, offsets from the strictly earlier calibration window
        point2 = GlobalLGBM.fit(panel, cur, cfg, groups, params=params, series_idx=sel)
        ev = QuantileBundle.fit(panel, cur, cfg, groups, sel, params, point2)
        ev.offsets = dict(cal.offsets)
        offsets_log[int(cur)] = dict(cal.offsets)
        raw_e = ev.predict_raw(panel, cur, allp, H)
        seg = R.attach_truth_and_segments(
            raw_e[["series_idx", "h"]].assign(forecast=0.0, fallback_used=""),
            panel,
            cur,
            cfg,
            vmeta["volume_edges"],
        )[["series_idx", "h", "actual", "volume_segment", "intermittency_segment"]]
        for label, applied in (("raw", _apply_zero(ev, raw_e)), ("cqr_calibrated", ev.apply(raw_e))):
            m = applied.merge(seg, on=["series_idx", "h"])
            r = interval_metrics(m, nominal, label)
            r["origin"] = cur
            rows.append(r)
            if label == "cqr_calibrated":
                qpred.append(m.assign(origin=cur))
        rows[-1]["nonneg_clip_rate"] = float(raw_e["neg_raw_lower"].mean())
        rows[-2]["nonneg_clip_rate"] = float(raw_e["neg_raw_lower"].mean())
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "interval_metrics.csv", index=False)
    pd.concat(qpred).to_parquet(out / "interval_predictions.parquet", index=False)
    summary = {
        "experiment_id": exp_id,
        "feature_set": key,
        "nominal": nominal,
        "evaluated_origins": val[1:],
        "calibration": "CQR per horizon bucket from previous validation window; no coverage guarantee under temporal dependence claimed",
        "cqr_offsets_by_eval_origin": offsets_log,
        "note_offsets": "Offsets are 0 when >= (1-nominal) of conformity scores sit on the clipped zero boundary (zero-inflated demand): "
        "CQR then neither widens nor shrinks. Compare raw vs calibrated coverage, do not assume a guarantee.",
        "overall": res[(res.scope == "all")]
        .groupby("variant")[
            ["coverage", "mean_width", "crossing_rate", "pinball_0.1", "pinball_0.5", "pinball_0.9"]
        ]
        .mean()
        .to_dict("index"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _apply_zero(bundle: QuantileBundle, raw: pd.DataFrame) -> pd.DataFrame:
    saved = bundle.offsets
    bundle.offsets = {}
    try:
        return bundle.apply(raw)
    finally:
        bundle.offsets = saved


# ------------------------------------------------------------------ interpretability
def run_interpretability(panel: Panel, cfg: dict) -> dict:
    """Permutation importance (validation, WAPE increase) and LightGBM contribution values on a bounded sample.
    Uses the LAST validation origin only; never the holdout. Importance is predictive, not causal; correlated lags
    share importance."""
    exp_id = E.make_experiment_id(cfg, panel.dataset_version)
    out = E.experiment_dir(exp_id) / "interpretability"
    out.mkdir(parents=True, exist_ok=True)
    origins = P.make_origins(panel.n_days, cfg)
    vmeta = json.loads((E.experiment_dir(exp_id) / "validation" / "run_meta.json").read_text())
    key = list(cfg["features"]["feature_sets"])[-1]
    groups = feature_groups_for(cfg, key)
    o = origins.validation[-1]
    sel = E._train_series(panel, cfg)
    model = GlobalLGBM.fit(panel, o, cfg, groups, params=vmeta.get("tuned_params") or None, series_idx=sel)
    ic = cfg["interpretability"]
    rng = np.random.default_rng(cfg["seed"])
    series = np.sort(
        rng.choice(panel.n_series, min(panel.n_series, max(1, ic["sample_rows"] // 28)), replace=False)
    )
    X = build_examples(
        panel,
        o,
        series,
        list(range(1, 29)),
        groups,
        model.enc,
        with_target=True,
        price_scenario=model.price_scenario,
    )
    X = X[X.y.notna()].reset_index(drop=True)
    cols = model.feature_cols
    y = X.y.to_numpy()
    base_pred = model.predict_matrix(X)
    base = np.abs(base_pred - y).sum() / np.abs(y).sum()
    seg = R.attach_truth_and_segments(
        X[["series_idx", "h"]].assign(forecast=0.0, fallback_used=""), panel, o, cfg, vmeta["volume_edges"]
    )
    X = X.merge(
        seg[["series_idx", "h", "volume_segment", "intermittency_segment"]],
        on=["series_idx", "h"],
        how="left",
    )
    imp = []
    groups_cols: dict[str, list[str]] = {}
    for c in cols:
        groups_cols.setdefault(group_of(c), []).append(c)

    def perm_wape(cs: list[str], mask=None) -> float:
        incr = []
        for _ in range(ic["permutation_repeats"]):
            Xp = X[cols].copy()
            perm = rng.permutation(len(Xp))
            for c in cs:
                Xp[c] = Xp[c].to_numpy()[perm]
            pr = model.predict_matrix(Xp)
            m = np.ones(len(Xp), bool) if mask is None else mask
            incr.append(np.abs(pr[m] - y[m]).sum() / np.abs(y[m]).sum())
        return float(np.mean(incr))

    for c in cols:
        imp.append({"feature": c, "group": group_of(c), "wape_permuted": perm_wape([c])})
    imp = pd.DataFrame(imp)
    imp["wape_increase"] = imp["wape_permuted"] - base
    imp.sort_values("wape_increase", ascending=False).to_csv(out / "permutation_by_feature.csv", index=False)
    gi = [{"group": g, "wape_permuted": perm_wape(cs)} for g, cs in groups_cols.items()]
    gi = pd.DataFrame(gi)
    gi["wape_increase"] = gi["wape_permuted"] - base
    gi.sort_values("wape_increase", ascending=False).to_csv(out / "permutation_by_group.csv", index=False)
    contrib = model.booster.predict(X[cols], pred_contrib=True)[:, :-1]
    abs_c = pd.DataFrame(np.abs(contrib), columns=cols)
    abs_c["volume_segment"] = X["volume_segment"].to_numpy()
    abs_c.drop(columns="volume_segment").mean().sort_values(ascending=False).rename(
        "mean_abs_contribution"
    ).to_csv(out / "contrib_overall.csv")
    seg_tab = abs_c.groupby("volume_segment").mean()
    gt = pd.DataFrame({g: seg_tab[cs].sum(axis=1) for g, cs in groups_cols.items()})
    gt.to_csv(out / "contrib_group_by_volume_segment.csv")
    res = {
        "origin": int(o),
        "baseline_wape": float(base),
        "rows": int(len(X)),
        "feature_set": key,
        "group_importance": gi.set_index("group")["wape_increase"].to_dict(),
        "notes": [
            "predictive importance, not causality",
            "correlated lag/rolling features share and dilute importance",
            "computed on a bounded sample at the last validation origin; not used to select features on the holdout",
        ],
    }
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    return res
