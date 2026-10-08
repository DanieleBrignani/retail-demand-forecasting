"""Prediction intervals: quantile LightGBM (0.1/0.5/0.9) + split-conformalised quantile regression (CQR).

Calibration uses a window strictly earlier than the evaluated window (calibration origin = evaluation origin - 28,
models trained with information <= calibration origin). Temporal dependence and non-exchangeability mean
**no finite-sample coverage guarantee is claimed**; coverage is measured, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features.build import build_examples
from .models.lgbm import GlobalLGBM
from .panel import Panel


class QuantileBundle:
    def __init__(
        self, point: GlobalLGBM, boosters: dict[float, lgb.Booster], offsets: dict[str, float] | None = None
    ):
        self.point, self.boosters, self.offsets = point, boosters, offsets or {}

    @classmethod
    def fit(
        cls,
        panel: Panel,
        origin: int,
        cfg: dict,
        groups: list[str],
        series_idx: np.ndarray,
        params: dict | None,
        point: GlobalLGBM,
        quantiles: list[float] | None = None,
    ) -> QuantileBundle:
        """Quantile boosters trained with information <= origin; ``point`` is the (tweedie) point model."""
        boosters = {}
        for q in quantiles or cfg["uncertainty"]["quantiles"]:
            m = GlobalLGBM.fit(
                panel,
                origin,
                cfg,
                groups,
                params=params,
                series_idx=series_idx,
                name=f"lgbm_q{q}",
                objective_override={"objective": "quantile", "alpha": q, "metric": "quantile"},
            )
            boosters[q] = m.booster
        return cls(point, boosters)

    def predict_raw(
        self, panel: Panel, origin: int, series_idx: np.ndarray, horizons: list[int]
    ) -> pd.DataFrame:
        """Uncalibrated quantiles (and crossing diagnostics), clipped at 0 (documented)."""
        sel = np.asarray(series_idx)
        X = build_examples(
            panel,
            origin,
            sel,
            horizons,
            self.point.groups,
            self.point.enc,
            with_target=False,
            price_scenario=self.point.price_scenario,
        )
        df = X[["series_idx", "h"]].copy()
        for q, b in self.boosters.items():
            df[f"q{q}"] = self.point.predict_ratio_booster(b, X).astype(np.float32)
        qs = sorted(self.boosters)
        if 0.5 not in self.boosters:  # serving bundles train only the outer quantiles
            df["q0.5"] = self.point.predict_matrix(X).astype(np.float32)
        lo, med, hi = f"q{qs[0]}", "q0.5", f"q{qs[-1]}"
        df["crossing"] = ((df[lo] > df[med]) | (df[med] > df[hi])).astype(np.int8)
        df["neg_raw_lower"] = (df[lo] < 0).astype(np.int8)
        for c in (lo, med, hi):
            df[c] = np.maximum(df[c], 0)
        # enforce monotone quantiles after recording the crossing
        arr = np.sort(df[[lo, med, hi]].to_numpy(), axis=1)
        df[lo], df[med], df[hi] = arr[:, 0], arr[:, 1], arr[:, 2]
        df.attrs["cols"] = (lo, med, hi)
        return df

    def calibrate(self, cal_pred: pd.DataFrame, actual: np.ndarray, nominal: float) -> None:
        """CQR: per horizon-bucket offset = ceil((n+1)(nominal))/n empirical quantile of max(lo-y, y-hi)."""
        lo, _, hi = cal_pred.attrs["cols"]
        score = np.maximum(cal_pred[lo].to_numpy() - actual, actual - cal_pred[hi].to_numpy())
        buckets = pd.cut(cal_pred["h"], [0, 7, 14, 21, 28], labels=["1-7", "8-14", "15-21", "22-28"])
        for b in buckets.cat.categories:
            s = score[(buckets == b).to_numpy()]
            if len(s):
                k = min(1.0, np.ceil((len(s) + 1) * nominal) / len(s))
                self.offsets[b] = float(np.quantile(s, k))

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        lo, med, hi = df.attrs["cols"]
        b = pd.cut(df["h"], [0, 7, 14, 21, 28], labels=["1-7", "8-14", "15-21", "22-28"]).astype(str)
        off = b.map(self.offsets).fillna(0.0).to_numpy()
        out = df.copy()
        out["lower"] = np.maximum(df[lo].to_numpy() - off, 0)
        out["upper"] = np.maximum(df[hi].to_numpy() + off, 0)
        out["median"] = df[med]
        return out

    # persistence
    def save(self, d: str | Path) -> None:
        d = Path(d)
        self.point.save(d / "point")
        for q, b in self.boosters.items():
            b.save_model(str(d / f"q{q}.txt"))
        (d / "offsets.json").write_text(
            json.dumps({"offsets": self.offsets, "quantiles": sorted(self.boosters)})
        )

    @classmethod
    def load(cls, d: str | Path) -> QuantileBundle:
        d = Path(d)
        meta = json.loads((d / "offsets.json").read_text())
        return cls(
            GlobalLGBM.load(d / "point"),
            {q: lgb.Booster(model_file=str(d / f"q{q}.txt")) for q in meta["quantiles"]},
            meta["offsets"],
        )


def pinball(y: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def interval_metrics(df: pd.DataFrame, nominal: float, label: str) -> pd.DataFrame:
    """df has actual, lower, median, upper, h, volume_segment, intermittency_segment, crossing."""
    rows = []

    def add(scope, seg_type, seg_val, g):
        rows.append(
            {
                "variant": label,
                "scope": scope,
                "segment_type": seg_type,
                "segment_value": seg_val,
                "n": len(g),
                "coverage": float(((g.actual >= g.lower) & (g.actual <= g.upper)).mean()),
                "nominal": nominal,
                "mean_width": float((g.upper - g.lower).mean()),
                "pinball_0.1": pinball(g.actual.to_numpy(), g.lower.to_numpy(), 0.1),
                "pinball_0.5": pinball(g.actual.to_numpy(), g["median"].to_numpy(), 0.5),
                "pinball_0.9": pinball(g.actual.to_numpy(), g.upper.to_numpy(), 0.9),
                "crossing_rate": float(g["crossing"].mean()),
            }
        )

    add("all", "all", "all", df)
    for lo, hi in [(1, 7), (8, 14), (15, 21), (22, 28)]:
        add("horizon", "h_range", f"{lo}-{hi}", df[(df.h >= lo) & (df.h <= hi)])
    for col in ("volume_segment", "intermittency_segment"):
        for v, g in df.groupby(col):
            add("segment", col, str(v), g)
    return pd.DataFrame(rows)
