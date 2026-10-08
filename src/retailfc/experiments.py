"""Experiment orchestration shared by CLI, Prefect flows, notebooks and tests.

Stages
------
1. ``run_validation``  -- baselines, restricted SARIMA, LightGBM ablation A..D on the validation origins
                          (tuning only on data before the first validation origin). Never touches the holdout.
2. ``select``          -- deterministic selection rule applied to validation metrics only.
3. ``run_holdout``     -- locked, single evaluation on the final holdout (requires protocol lock).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import protocol as P
from .backtesting import runner as R
from .features.spec import columns_for, feature_version
from .models.baselines import BASELINES, predict_baseline
from .models.lgbm import GlobalLGBM, feature_groups_for, tune
from .models.stats import SarimaForecaster, select_subset
from .panel import Panel


class _Baseline:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg

    def predict(self, panel, origin, sel, horizons):
        return predict_baseline(self.name, panel, origin, sel, horizons, self.cfg)


def code_revision() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, cwd=C.home()
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unavailable"


def environment_info() -> dict:
    import lightgbm
    import statsmodels

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lightgbm.__version__,
        "statsmodels": statsmodels.__version__,
        "code_revision": code_revision(),
    }


def baseline_specs(cfg: dict) -> list[R.ModelSpec]:
    return [R.ModelSpec(n, "v1", (lambda panel, o, n=n: _Baseline(n, cfg))) for n in BASELINES]


def lgbm_spec(
    cfg: dict, key: str, params: dict | None, series_idx: np.ndarray | None, name: str | None = None
) -> R.ModelSpec:
    groups = feature_groups_for(cfg, key)
    cols = columns_for(groups, cfg["features"]["price_scenario"])
    ver = feature_version(cols, cfg["features"]["price_scenario"])
    nm = name or f"lgbm_{key}"
    return R.ModelSpec(
        nm,
        f"{nm}@{ver}",
        lambda panel, o: GlobalLGBM.fit(
            panel, o, cfg, groups, params=params, series_idx=series_idx, name=nm, measure_memory=False
        ),
    )


def experiment_dir(experiment_id: str) -> Path:
    return C.artifacts_dir() / "experiments" / experiment_id


def make_experiment_id(cfg: dict, dataset_version: str) -> str:
    return f"{cfg['experiment_name']}-{dataset_version}-{C.config_hash(cfg)}"


def _train_series(panel: Panel, cfg: dict) -> np.ndarray:
    ms = cfg["training"]["max_series"]
    if ms and ms < panel.n_series:
        return np.sort(np.random.default_rng(cfg["seed"]).choice(panel.n_series, ms, replace=False))
    return np.arange(panel.n_series)


def run_validation(
    panel: Panel,
    cfg: dict,
    *,
    skip_sarima: bool = False,
    feature_keys: list[str] | None = None,
    skip_tuning: bool = False,
) -> dict:
    exp_id = make_experiment_id(cfg, panel.dataset_version)
    out = experiment_dir(exp_id) / "validation"
    out.mkdir(parents=True, exist_ok=True)
    origins = P.make_origins(panel.n_days, cfg)
    P.check_non_overlap(origins, cfg["horizon"])
    val = list(origins.validation)
    sel = _train_series(panel, cfg)
    edges = R.edges_for(panel, val[0], cfg)  # tertile edges from data <= first validation origin
    keys = feature_keys or list(cfg["features"]["feature_sets"])

    t_tune = time.perf_counter()
    params, trials = ({}, [])
    if not skip_tuning and cfg["lightgbm"]["tuning"]["n_trials"] > 0:
        params, trials = tune(panel, val[0], cfg, feature_groups_for(cfg, keys[-1]), sel)
    tune_seconds = time.perf_counter() - t_tune

    specs = baseline_specs(cfg)
    scope = {s.name: "full_panel" for s in specs}
    subset = None
    if not skip_sarima:
        subset = select_subset(panel, val[0], cfg, edges)
        sar = SarimaForecaster(cfg, subset)
        specs.append(R.ModelSpec("sarima", "v1", lambda panel, o, sar=sar: sar, restricted_to=subset))
        scope["sarima"] = "restricted_subset"
    for k in keys:
        s = lgbm_spec(cfg, k, params, sel)
        specs.append(s)
        scope[s.name] = "full_panel"

    R.run_backtest(panel, cfg, specs, val, np.arange(panel.n_series), out, volume_edges=edges)
    base = {
        "backtest_id": f"{exp_id}:validation",
        "experiment_id": exp_id,
        "dataset_version": panel.dataset_version,
        "feature_version": feature_version(
            columns_for(feature_groups_for(cfg, keys[-1]), cfg["features"]["price_scenario"])
        ),
    }
    met = R.compute_all_metrics(out, base, scope_by_model=scope)
    # restricted-subset comparison: every other model re-scored on the same subset
    if subset is not None:
        sub_parts = []
        for f in sorted((out / "predictions").glob("*.parquet")):
            pr = pd.read_parquet(f)
            if pr["model_name"].iloc[0] == "sarima":
                continue
            pr = pr[pr["series_idx"].isin(subset)]
            sub_parts.append(
                R.metric_records(
                    pr,
                    {
                        **base,
                        "model_name": pr["model_name"].iloc[0],
                        "model_version": pr["model_version"].iloc[0],
                        "evaluation_scope": "restricted_subset",
                    },
                )
            )
        if sub_parts:
            met = pd.concat([met, *sub_parts], ignore_index=True)
            met.to_parquet(out / "metrics.parquet", index=False)
    meta = {
        "experiment_id": exp_id,
        "config_hash": C.config_hash(cfg),
        "dataset_version": panel.dataset_version,
        "origins": origins.as_dict(),
        "volume_edges": edges,
        "n_series": panel.n_series,
        "n_train_series": int(len(sel)),
        "sarima_subset_size": None if subset is None else int(len(subset)),
        "tuned_params": params,
        "tuning_trials": trials,
        "tuning_seconds": tune_seconds,
        "environment": environment_info(),
        "stage": "validation",
    }
    R.write_json(out / "run_meta.json", meta)
    if subset is not None:
        R.write_json(
            out / "sarima_subset.json",
            {"series_idx": subset.tolist(), "items": panel.ids.iloc[subset].to_dict("records")},
        )
        pd.DataFrame(sar.records).to_csv(out / "sarima_fits.csv", index=False)
    return meta


def select_feature_set(cfg: dict, exp_id: str) -> dict:
    """Rule (config-independent, fixed before results): walk A->D; adopt a richer set only if mean validation
    pooled WAPE (1-28, over validation origins) improves by >= min_gain relative to the current choice."""
    min_gain = cfg["selection"]["min_relative_gain"]
    met = pd.read_parquet(experiment_dir(exp_id) / "validation" / "metrics.parquet")
    m = met[
        (met.aggregation == "pooled")
        & (met.metric == "wape")
        & (met.segment_type == "all")
        & (met.horizon == "1-28")
        & met.forecast_origin.notna()
        & (met.evaluation_scope == "full_panel")
    ]
    per = m.groupby("model_name")["value"].mean()
    keys = list(cfg["features"]["feature_sets"])
    chosen = keys[0]
    log = []
    for k in keys[1:]:
        a, b = per[f"lgbm_{chosen}"], per[f"lgbm_{k}"]
        gain = (a - b) / a
        log.append(
            {
                "candidate": k,
                "vs": chosen,
                "wape_prev": float(a),
                "wape_cand": float(b),
                "rel_gain": float(gain),
                "adopted": bool(gain >= min_gain),
            }
        )
        if gain >= min_gain:
            chosen = k
    res = {
        "selected_feature_set": chosen,
        "min_gain": min_gain,
        "log": log,
        "mean_validation_wape": per.to_dict(),
    }
    R.write_json(experiment_dir(exp_id) / "selection.json", res)
    return res


def run_holdout(panel: Panel, cfg: dict, *, allow_repeat: bool = False) -> dict:
    P.assert_protocol_locked(cfg, panel.dataset_version)
    exp_id = make_experiment_id(cfg, panel.dataset_version)
    exp = experiment_dir(exp_id)
    sel_path = exp / "selection.json"
    if not sel_path.exists():
        raise RuntimeError("Run validation and selection before the holdout.")
    selection = json.loads(sel_path.read_text())
    vmeta = json.loads((exp / "validation" / "run_meta.json").read_text())
    origins = P.make_origins(panel.n_days, cfg)
    P.record_holdout_evaluation(
        {"experiment_id": exp_id, "selected": selection["selected_feature_set"]}, allow_repeat=allow_repeat
    )
    out = exp / "holdout"
    out.mkdir(parents=True, exist_ok=True)
    sel = _train_series(panel, cfg)
    key = selection["selected_feature_set"]
    specs = baseline_specs(cfg)
    scope = {s.name: "full_panel" for s in specs}
    s = lgbm_spec(cfg, key, vmeta["tuned_params"] or None, sel, name="lgbm_selected")
    specs.append(s)
    scope[s.name] = "full_panel"
    subset = None
    sj = exp / "validation" / "sarima_subset.json"
    if sj.exists():
        subset = np.array(json.loads(sj.read_text())["series_idx"])
        sar = SarimaForecaster(cfg, subset)
        specs.append(R.ModelSpec("sarima", "v1", lambda panel, o, sar=sar: sar, restricted_to=subset))
        scope["sarima"] = "restricted_subset"
    R.run_backtest(
        panel,
        cfg,
        specs,
        [origins.holdout],
        np.arange(panel.n_series),
        out,
        volume_edges=vmeta["volume_edges"],
    )
    base = {
        "backtest_id": f"{exp_id}:holdout",
        "experiment_id": exp_id,
        "dataset_version": panel.dataset_version,
        "feature_version": s.version.split("@")[1],
    }
    met = R.compute_all_metrics(out, base, scope_by_model=scope)
    if subset is not None:
        parts = []
        for f in sorted((out / "predictions").glob("*.parquet")):
            pr = pd.read_parquet(f)
            if pr["model_name"].iloc[0] == "sarima":
                continue
            pr = pr[pr["series_idx"].isin(subset)]
            parts.append(
                R.metric_records(
                    pr,
                    {
                        **base,
                        "model_name": pr["model_name"].iloc[0],
                        "model_version": pr["model_version"].iloc[0],
                        "evaluation_scope": "restricted_subset",
                    },
                )
            )
        met = pd.concat([met, *parts], ignore_index=True)
        met.to_parquet(out / "metrics.parquet", index=False)
    R.write_json(
        out / "run_meta.json",
        {
            "experiment_id": exp_id,
            "origin": origins.holdout,
            "selected": key,
            "stage": "holdout",
            "environment": environment_info(),
        },
    )
    return {"experiment_id": exp_id, "holdout_origin": origins.holdout, "selected": key}


# ----------------------------------------------------------------------------------------- model-variant comparison
DEFAULT_VARIANTS = {
    "v1_subset": {"training.max_series": 3000, "lightgbm.level_offset": False},
    "all_series": {"training.max_series": None, "lightgbm.level_offset": False},
    "offset_subset": {"training.max_series": 3000, "lightgbm.level_offset": True},
    "v2_all_offset": {"training.max_series": None, "lightgbm.level_offset": True},
}


def run_variants(
    panel: Panel,
    cfg: dict,
    origins: list[int],
    variants: dict[str, dict] | None = None,
    *,
    feature_key: str = "D",
    params: dict | None = None,
    tag: str = "variants",
) -> dict:
    """Backtest several LightGBM configurations (plus baselines) on the SAME origins/series/actuals.
    Only information up to each origin is used. Writes predictions + a pooled summary; never touches the holdout unless asked."""

    variants = variants or DEFAULT_VARIANTS
    exp_id = make_experiment_id(cfg, panel.dataset_version)
    out = experiment_dir(exp_id) / tag
    out.mkdir(parents=True, exist_ok=True)
    specs = baseline_specs(cfg)
    for name, ov in variants.items():
        vcfg = C.override(cfg, **ov)
        specs.append(lgbm_spec(vcfg, feature_key, params, _train_series(panel, vcfg), name=name))
    edges = R.edges_for(panel, origins[0], cfg)
    R.run_backtest(panel, cfg, specs, origins, np.arange(panel.n_series), out, volume_edges=edges)
    summary = summarize_predictions(out)
    summary.to_csv(out / "summary.csv", index=False)
    R.write_json(
        out / "run_meta.json",
        {
            "origins": origins,
            "variants": variants,
            "feature_key": feature_key,
            "params": params,
            "dataset_version": panel.dataset_version,
            "environment": environment_info(),
        },
    )
    return {"out": str(out), "summary": summary.round(4).to_dict("records")}


def summarize_predictions(out: Path) -> "pd.DataFrame":
    """Pooled WAPE/MAE/RMSE/bias per model for all origins and for each origin, plus the all-zero reference."""
    import pandas as pd

    from .backtesting import metrics as M

    rows = []
    zero_done = False
    for f in sorted((Path(out) / "predictions").glob("*.parquet")):
        pr = pd.read_parquet(f)
        name = pr["model_name"].iloc[0]
        for label, g in [("all", pr), *[(int(o), d) for o, d in pr.groupby("origin")]]:
            m = M.pooled(g).iloc[0]
            rows.append(
                {
                    "model": name,
                    "origin": label,
                    "wape": m["wape"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "bias_pct": m["bias_pct"],
                    "n": int(m["n"]),
                }
            )
            if not zero_done:
                a = g["actual"].to_numpy(float)
                rows.append(
                    {
                        "model": "all_zero_reference",
                        "origin": label,
                        "wape": 1.0,
                        "mae": a.mean(),
                        "rmse": float(np.sqrt((a**2).mean())),
                        "bias_pct": -1.0,
                        "n": len(a),
                    }
                )
        zero_done = True
    return pd.DataFrame(rows)
