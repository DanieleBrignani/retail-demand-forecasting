"""ARIMA/SARIMA on a restricted, training-only-selected subset of series.

Per origin and series: candidate orders are scored on an inner validation (fit on days <= o-28 within the
trailing window, forecast the 28 days up to o, score MAE), the best order is refit on days <= o, and
the 28-day forecast is produced. Any failure (exception, non-finite output, non-convergence) is recorded
and replaced by seasonal naive with ``fallback_used='sarima_fit_failed'``; series are never dropped.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from ..panel import Panel
from ..segments import segment_frame
from .baselines import predict_baseline


def select_subset(panel: Panel, ref_origin: int, cfg: dict, volume_edges: list[float]) -> np.ndarray:
    """Stratified (store x category x volume x intermittency) seeded sample using data <= ref_origin only."""
    seg = segment_frame(panel, ref_origin, cfg, volume_edges)
    df = pd.concat([panel.ids[["store_id", "cat_id"]].reset_index(drop=True), seg], axis=1)
    df["stratum"] = (
        df["store_id"] + "|" + df["cat_id"] + "|" + df["volume_segment"] + "|" + df["intermittency_segment"]
    )
    rng = np.random.default_rng(cfg["seed"])
    k = min(cfg["statistical"]["subset_size"], len(df))
    strata = df.groupby("stratum").indices
    order = list(strata)
    rng.shuffle(order)
    picked: list[int] = []
    pools = {s: list(rng.permutation(strata[s])) for s in order}
    while len(picked) < k and any(pools.values()):  # round-robin across strata => broad coverage
        for s in order:
            if pools[s] and len(picked) < k:
                picked.append(int(pools[s].pop()))
    return np.sort(np.array(picked))


def _fit_forecast(y: np.ndarray, steps: int, order, seasonal, maxiter: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(
            y,
            order=tuple(order),
            seasonal_order=tuple(seasonal),
            trend="c" if order[1] == 0 and seasonal[1] == 0 else "n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False, maxiter=maxiter)
        if not res.mle_retvals.get("converged", True):
            raise RuntimeError("not converged")
        fc = res.forecast(steps)
    fc = np.asarray(fc, dtype=float)
    if not np.all(np.isfinite(fc)):
        raise RuntimeError("non-finite forecast")
    return fc


class SarimaForecaster:
    def __init__(self, cfg: dict, subset: np.ndarray, name: str = "sarima"):
        self.cfg, self.subset, self.name = cfg, np.asarray(subset), name
        self.records: list[dict] = []  # per fit: origin, series, chosen order, fail flag, seconds

    def predict(self, panel: Panel, origin: int, series_idx: np.ndarray, horizons: list[int]) -> pd.DataFrame:
        sc = self.cfg["statistical"]
        H = max(horizons)
        sel = np.intersect1d(np.asarray(series_idx), self.subset)
        a = panel.as_of(origin)
        sn = predict_baseline("seasonal_naive", panel, origin, sel, horizons, self.cfg).set_index(
            ["series_idx", "h"]
        )
        rows = []
        for s in sel:
            y = a.sales[s, -sc["fit_window_days"] :].astype(float)
            t0 = time.perf_counter()
            rec = {
                "origin": origin,
                "series_idx": int(s),
                "order": None,
                "failed": False,
                "negative_raw": 0,
                "error": "",
            }
            try:
                if not (y > 0).any():
                    raise RuntimeError("all_zero_history")
                best, best_err = None, np.inf
                if len(y) > 2 * H + 20:
                    for cand in sc["candidate_orders"]:
                        try:
                            f = _fit_forecast(y[:-H], H, cand["order"], cand["seasonal_order"], sc["maxiter"])
                            err = np.abs(np.maximum(f, 0) - y[-H:]).mean()
                            if err < best_err:
                                best, best_err = cand, err
                        except Exception:
                            continue
                if best is None:
                    best = sc["candidate_orders"][0]
                rec["order"] = f"{best['order']}x{best['seasonal_order']}"
                fc = _fit_forecast(y, H, best["order"], best["seasonal_order"], sc["maxiter"])
                rec["negative_raw"] = int((fc < 0).sum())
                fc = np.maximum(fc, 0)
                fb = ""
            except Exception as e:
                rec["failed"], rec["error"] = True, str(e)[:120]
                fc = None
                fb = "sarima_all_zero_history" if str(e) == "all_zero_history" else "sarima_fit_failed"
            rec["seconds"] = time.perf_counter() - t0
            self.records.append(rec)
            for h in horizons:
                if fc is None:
                    v = float(sn.loc[(s, h), "forecast"])
                else:
                    v = float(fc[h - 1])
                rows.append((int(s), h, v, fb))
        return pd.DataFrame(rows, columns=["series_idx", "h", "forecast", "fallback_used"]).astype(
            {"series_idx": "int32", "h": "int16", "forecast": "float32"}
        )
