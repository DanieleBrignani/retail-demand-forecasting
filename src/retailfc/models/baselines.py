"""Baselines. All read history only through ``panel.as_of(origin)``.

Forecast frames are long: columns ``series_idx, h, forecast, fallback_used`` (+ optional ``lower, upper``).

Fallback policy (documented in docs/limitations.md):
* ``all_zero_history``  -> no positive sale observed by the origin: forecast 0 (an explicit, flagged fallback).
* ``short_history``     -> fewer observed days since the first positive sale than the window: the window is
                           truncated to the available days; flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..panel import Panel

BASELINES = ("naive", "seasonal_naive", "mean", "seasonal_mean")


def _frame(sel: np.ndarray, horizons: list[int], fc: np.ndarray, fb: np.ndarray) -> pd.DataFrame:
    m, H = fc.shape
    return pd.DataFrame(
        {
            "series_idx": np.tile(sel.astype(np.int32), H),
            "h": np.repeat(np.asarray(horizons, dtype=np.int16), m),
            "forecast": fc.T.reshape(-1).astype(np.float32),
            "fallback_used": np.tile(fb, H),
        }
    )


def _history_state(sales: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    pos = sales > 0
    ever = pos.any(axis=1)
    first = np.where(ever, np.argmax(pos, axis=1), sales.shape[1])
    avail = sales.shape[1] - first  # days since first positive sale (inclusive)
    fb = np.full(len(sales), "", dtype=object)
    fb[avail < window] = "short_history"
    fb[~ever] = "all_zero_history"
    return avail, fb


def predict_baseline(
    name: str, panel: Panel, origin: int, series_idx: np.ndarray, horizons: list[int], cfg: dict
) -> pd.DataFrame:
    a = panel.as_of(origin)
    sel = np.asarray(series_idx)
    H = len(horizons)
    hist = a.sales[sel].astype(np.float32)  # (m, o)
    m, o = hist.shape
    b = cfg["baselines"]
    if name == "naive":
        fc = np.repeat(hist[:, -1:], H, axis=1)
        _, fb = _history_state(hist, 1)
    elif name == "seasonal_naive":
        cols = [origin + h - 7 * int(np.ceil(h / 7)) - 1 for h in horizons]  # 0-based, all <= o-1
        if min(cols) < 0:  # < 7 days of history: fall back to naive
            fc = np.repeat(hist[:, -1:], H, axis=1)
            fb = np.full(m, "short_history", dtype=object)
        else:
            fc = hist[:, cols]
            _, fb = _history_state(hist, 7)
    elif name == "mean":
        w = min(b["mean_window_days"], o)
        fc = np.repeat(hist[:, -w:].mean(axis=1, keepdims=True), H, axis=1)
        _, fb = _history_state(hist, b["mean_window_days"])
    elif name == "seasonal_mean":
        weeks = b["seasonal_mean_weeks"]
        fc = np.zeros((m, H), dtype=np.float32)
        for j, h in enumerate(horizons):
            last = origin + h - 7 * int(np.ceil(h / 7))  # 1-based day, <= o
            days = [last - 7 * k for k in range(weeks) if last - 7 * k >= 1]
            fc[:, j] = hist[:, [d - 1 for d in days]].mean(axis=1)
        _, fb = _history_state(hist, 7 * weeks)
    else:
        raise ValueError(f"unknown baseline {name}")
    ever = (hist > 0).any(axis=1)
    fc[~ever] = 0.0
    return _frame(sel, horizons, np.maximum(fc, 0), fb)
