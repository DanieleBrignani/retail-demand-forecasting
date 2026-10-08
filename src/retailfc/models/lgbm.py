"""Global direct LightGBM forecaster (point and quantile) with temporal early stopping and serialization."""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.build import Encoders, build_examples, build_training_frame, fit_encoders
from ..features.spec import categorical_columns, columns_for, feature_version
from ..panel import Panel
from ..protocol import training_origins

_SNAIVE_CFG = {"baselines": {"mean_window_days": 28, "seasonal_mean_weeks": 4}}

LEVEL_COL = (
    "roll_mean_28"  # trailing 28-day mean ending at the origin: the series level known at forecast time
)


def level_of(X: pd.DataFrame, eps: float) -> np.ndarray:
    """Series level used as an offset/scale: trailing 28-day mean (NaN -> 0, never negative) + eps."""
    return np.clip(np.nan_to_num(X[LEVEL_COL].to_numpy(dtype=float), nan=0.0), 0.0, None) + eps


def _link_for(objective: str, offset: bool) -> str | None:
    """How the level offset enters: 'log' (tweedie/poisson: mu = level * exp(f)), 'ratio' (quantile: target y/level)."""
    if not offset:
        return None
    if objective in ("tweedie", "poisson"):
        return "log"
    if objective == "quantile":
        return "ratio"
    raise ValueError(f"level_offset is not defined for objective {objective!r}")


def feature_groups_for(cfg: dict, key: str) -> list[str]:
    return list(cfg["features"]["feature_sets"][key])


@dataclass
class FitInfo:
    train_seconds: float = 0.0
    n_train_rows: int = 0
    best_iteration: int = 0
    n_features: int = 0
    train_origins: list[int] = field(default_factory=list)
    es_origin: int | None = None
    peak_mem_mb: float | None = None
    es_score: float | None = None


class GlobalLGBM:
    def __init__(
        self,
        booster: lgb.Booster,
        enc: Encoders,
        groups: list[str],
        price_scenario: str,
        params: dict,
        info: FitInfo | None = None,
        name: str = "lgbm",
        level_link: str | None = None,
        level_eps: float = 0.1,
    ):
        self.level_link, self.level_eps = level_link, level_eps
        self.booster, self.enc, self.groups = booster, enc, groups
        self.price_scenario, self.params = price_scenario, params
        self.info = info or FitInfo()
        self.feature_cols = columns_for(groups, price_scenario)
        self.feature_version = feature_version(self.feature_cols, price_scenario)
        self.name = name

    # ---------------------------------------------------------------- training
    @classmethod
    def fit(
        cls,
        panel: Panel,
        backtest_origin: int,
        cfg: dict,
        groups: list[str],
        *,
        params: dict | None = None,
        series_idx: np.ndarray | None = None,
        price_scenario: str | None = None,
        name: str = "lgbm",
        measure_memory: bool = False,
        objective_override: dict | None = None,
    ) -> GlobalLGBM:
        """Fit using only information available at ``backtest_origin``.

        Early stopping uses origin-28 (targets end at the origin) with training origins <= origin-56 so that
        no training target window overlaps the early-stopping window; then (optionally) refit on origins
        <= origin-28 with the selected number of rounds.
        """
        t0 = time.perf_counter()
        if measure_memory:
            tracemalloc.start()
        scen = price_scenario or cfg["features"]["price_scenario"]
        lg = cfg["lightgbm"]
        p = dict(lg["params"])
        if params:
            p.update(params)
        if objective_override:
            p.update(objective_override)
        H = cfg["horizon"]
        horizons = list(range(1, H + 1))
        sel = np.arange(panel.n_series) if series_idx is None else np.asarray(series_idx)
        enc = fit_encoders(panel, backtest_origin)
        es_o = backtest_origin - cfg["training"]["early_stopping_origin_gap"]
        tr_o = training_origins(backtest_origin, cfg, gap=cfg["training"]["early_stopping_origin_gap"])
        cols = columns_for(groups, scen)
        cats = categorical_columns(cols)
        eps = float(lg.get("level_offset_eps", 0.1))
        link = _link_for(p.get("objective", ""), bool(lg.get("level_offset", False)))

        def xy(df: pd.DataFrame):
            """(features, label, init_score) with the level offset applied when enabled."""
            if link == "log":
                return df[cols], df["y"], np.log(level_of(df, eps))
            if link == "ratio":
                return df[cols], df["y"].to_numpy() / level_of(df, eps), None
            return df[cols], df["y"], None

        tr = build_training_frame(panel, tr_o, sel, horizons, groups, enc, scen)
        va = build_training_frame(panel, [es_o], sel, horizons, groups, enc, scen)
        Xt, yt, it = xy(tr)
        Xv, yv, iv = xy(va)
        dtr = lgb.Dataset(Xt, yt, init_score=it, categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(Xv, yv, init_score=iv, categorical_feature=cats, reference=dtr, free_raw_data=False)
        evals: dict = {}
        booster = lgb.train(
            p,
            dtr,
            num_boost_round=lg["num_boost_round"],
            valid_sets=[dva],
            valid_names=["es"],
            callbacks=[
                lgb.early_stopping(lg["early_stopping_rounds"], verbose=False),
                lgb.record_evaluation(evals),
            ],
        )
        best = max(1, booster.best_iteration or lg["num_boost_round"])
        metric_name = next(iter(evals["es"]))
        es_score = float(evals["es"][metric_name][best - 1])
        n_rows = len(tr)
        all_o = tr_o
        if cfg["training"]["refit_on_all"]:
            del tr, va, dtr, dva  # release the first frames before building the (larger) refit frame
            gc.collect()
            all_o = training_origins(backtest_origin, cfg)
            full = build_training_frame(panel, all_o, sel, horizons, groups, enc, scen)
            Xf, yf, i_f = xy(full)
            dfull = lgb.Dataset(Xf, yf, init_score=i_f, categorical_feature=cats)
            booster = lgb.train(p, dfull, num_boost_round=best)
            n_rows = len(full)
        info = FitInfo(time.perf_counter() - t0, n_rows, best, len(cols), list(all_o), es_o, None, es_score)
        if measure_memory:
            info.peak_mem_mb = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
        return cls(booster, enc, groups, scen, p, info, name, level_link=link, level_eps=eps)

    # ---------------------------------------------------------------- inference
    def predict(
        self,
        panel: Panel,
        origin: int,
        series_idx: np.ndarray,
        horizons: list[int],
        chunk: int = 4000,
        clip: bool = True,
    ) -> pd.DataFrame:
        """Model forecasts with explicit fallbacks (never silent):
        * all-zero history        -> 0,                       fallback_used='all_zero_history'
        * unseen identifier       -> seasonal naive,          fallback_used='unseen_series'
        * non-finite model output -> seasonal naive,          fallback_used='nonfinite_model_output'
        Negative raw outputs are clipped at 0 and counted in ``negative_raw``."""
        out = []
        sel = np.asarray(series_idx)
        for i in range(0, len(sel), chunk):
            part = sel[i : i + chunk]
            X = build_examples(
                panel,
                origin,
                part,
                horizons,
                self.groups,
                self.enc,
                with_target=False,
                price_scenario=self.price_scenario,
            )
            raw = self.predict_matrix(X)
            unseen = (
                (X[["item_id", "dept_id", "cat_id", "store_id", "state_id"]] < 0).any(axis=1).to_numpy()
                if "item_id" in X.columns
                else np.zeros(len(X), bool)
            )
            out.append(
                pd.DataFrame(
                    {
                        "series_idx": X["series_idx"].to_numpy(),
                        "h": X["h"].to_numpy(),
                        "raw": raw.astype(np.float32),
                        "_unseen": unseen,
                    }
                )
            )
        df = pd.concat(out, ignore_index=True)
        df["forecast"] = np.maximum(df["raw"], 0) if clip else df["raw"]
        df["negative_raw"] = (df["raw"] < 0).astype(np.int8)
        df["fallback_used"] = ""
        bad = ~np.isfinite(df["raw"].to_numpy()) | df["_unseen"].to_numpy()
        if bad.any():
            from .baselines import predict_baseline

            sn = predict_baseline("seasonal_naive", panel, origin, sel, horizons, _SNAIVE_CFG).set_index(
                ["series_idx", "h"]
            )
            keys = pd.MultiIndex.from_frame(df.loc[bad, ["series_idx", "h"]])
            df.loc[bad, "forecast"] = sn.loc[keys, "forecast"].to_numpy()
            df.loc[bad, "fallback_used"] = np.where(
                df.loc[bad, "_unseen"], "unseen_series", "nonfinite_model_output"
            )
        ever = panel.as_of(origin).sales[sel].max(axis=1) > 0
        zero_series = set(sel[~ever].tolist())
        if zero_series:
            m = df["series_idx"].isin(zero_series)
            df.loc[m, "forecast"] = 0.0
            df.loc[m, "fallback_used"] = "all_zero_history"
        return df.drop(columns=["_unseen"])

    def predict_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Point forecasts on the original (units) scale. With a level offset: mu = level * exp(raw_score)."""
        if self.level_link == "log":
            return np.exp(
                self.booster.predict(X[self.feature_cols], raw_score=True)
                + np.log(level_of(X, self.level_eps))
            )
        return self.booster.predict(X[self.feature_cols])

    def predict_ratio_booster(self, booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
        """Quantile boosters trained on y/level are rescaled by the level (identity when no offset is used)."""
        out = booster.predict(X[self.feature_cols])
        return out * level_of(X, self.level_eps) if self.level_link is not None else out

    # ---------------------------------------------------------------- serialization
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / "model.txt"))
        meta = {
            "groups": self.groups,
            "price_scenario": self.price_scenario,
            "params": self.params,
            "encoders": json.loads(self.enc.to_json()),
            "feature_cols": self.feature_cols,
            "feature_version": self.feature_version,
            "name": self.name,
            "level_link": self.level_link,
            "level_eps": self.level_eps,
            "info": {k: v for k, v in self.info.__dict__.items()},
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> GlobalLGBM:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        booster = lgb.Booster(model_file=str(d / "model.txt"))
        enc = Encoders.from_json(json.dumps(meta["encoders"]))
        info = FitInfo(**{k: v for k, v in meta["info"].items() if k in FitInfo.__dataclass_fields__})
        m = cls(
            booster,
            enc,
            meta["groups"],
            meta["price_scenario"],
            meta["params"],
            info,
            meta["name"],
            level_link=meta.get("level_link"),
            level_eps=meta.get("level_eps", 0.1),
        )
        assert m.feature_cols == meta["feature_cols"], "feature column mismatch after load"
        return m


# ------------------------------------------------------------------------------- tuning
def tune(
    panel: Panel, tune_origin: int, cfg: dict, groups: list[str], series_idx: np.ndarray | None = None
) -> tuple[dict, list[dict]]:
    """Bounded random search. Scores = early-stopping loss at origin tune_origin-28; only data <= tune_origin
    is read, and tune_origin is the *first* validation origin, so later validation windows and the holdout are untouched."""
    tc = cfg["lightgbm"]["tuning"]
    rng = np.random.default_rng(cfg["seed"])
    trials = []
    cfg_t = {**cfg, "training": {**cfg["training"], "refit_on_all": False}}
    for i in range(tc["n_trials"]):
        params = {k: v[int(rng.integers(len(v)))] for k, v in tc["space"].items()}
        t0 = time.perf_counter()
        try:
            m = GlobalLGBM.fit(panel, tune_origin, cfg_t, groups, params=params, series_idx=series_idx)
            trials.append(
                {
                    "trial": i,
                    "params": params,
                    "score": m.info.es_score,
                    "best_iteration": m.info.best_iteration,
                    "seconds": time.perf_counter() - t0,
                    "failed": False,
                }
            )
        except Exception as e:  # recorded, never silently dropped
            trials.append(
                {"trial": i, "params": params, "score": float("inf"), "failed": True, "error": repr(e)}
            )
    ok = [t for t in trials if not t["failed"]]
    if not ok:
        return {}, trials
    best = min(ok, key=lambda t: t["score"])
    return best["params"], trials
