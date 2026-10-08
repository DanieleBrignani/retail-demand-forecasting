"""Origin-anchored feature construction for the direct global strategy.

One call builds all (series x horizon) rows for a single forecast origin. Row order is h-major:
``row = hi * m + i`` for horizon index ``hi`` and series position ``i``.
Sales features come exclusively from ``panel.as_of(origin)`` (days <= origin);
the training target is read separately from the full panel and never enters a feature.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..panel import ID_COLS, Panel
from .spec import columns_for

LOOKBACK = 365


class CalendarCoverageError(ValueError):
    """Target dates fall outside the calendar: refuse rather than fabricate calendar features."""


@dataclass
class Encoders:
    ids: dict[str, dict[str, int]]
    event_type: dict[str, int]
    event_name: dict[str, int]
    fit_cutoff_day: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "ids": self.ids,
                "event_type": self.event_type,
                "event_name": self.event_name,
                "fit_cutoff_day": self.fit_cutoff_day,
            }
        )

    @classmethod
    def from_json(cls, s: str) -> Encoders:
        d = json.loads(s)
        return cls(d["ids"], d["event_type"], d["event_name"], d["fit_cutoff_day"])


def fit_encoders(panel: Panel, cutoff_day: int) -> Encoders:
    """Fit categorical vocabularies using only information up to ``cutoff_day`` (training data)."""
    known = panel.ids.iloc[
        panel.eligible_at(cutoff_day)
    ]  # products observed by the cutoff; later products stay unseen (-1)
    ids = {c: {v: i for i, v in enumerate(sorted(known[c].unique()))} for c in ID_COLS}
    cal = panel.calendar.iloc[:cutoff_day]

    def vocab(col: str) -> dict[str, int]:
        vals = sorted(set(cal[col].fillna("none").astype(str)) | {"none"})
        return {v: i for i, v in enumerate(vals)}

    return Encoders(ids, vocab("event_type_1"), vocab("event_name_1"), cutoff_day)


def _encode(series: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    return series.map(mapping).fillna(-1).to_numpy(dtype=np.int32)


def history_features(tail: np.ndarray, origin: int) -> dict[str, np.ndarray]:
    """tail: (m, L) last L<=365 observed days ending exactly at the origin."""
    m, L = tail.shape
    t = tail.astype(np.float32)
    nan = np.full(m, np.nan, dtype=np.float32)

    def lag(k: int) -> np.ndarray:
        return t[:, -(k + 1)] if origin > k and L > k else nan

    def roll(w: int, fn) -> np.ndarray:
        return fn(t[:, -w:], axis=1).astype(np.float32) if origin >= w and L >= w else nan

    pos = t > 0
    ever = pos.any(axis=1)
    last_pos_from_end = np.argmax(pos[:, ::-1], axis=1)  # 0 => positive on day o
    first_pos_from_start = np.argmax(pos, axis=1)
    days_since = np.where(ever, last_pos_from_end, np.nan).astype(np.float32)
    # age uses the lookback window only when history is longer than it (capped at L): documented cap
    age = np.where(ever, L - first_pos_from_start, 0).astype(np.float32)
    return {
        "last_sale": t[:, -1],
        "lag_7": lag(7),
        "lag_14": lag(14),
        "lag_28": lag(28),
        "roll_mean_7": roll(7, np.mean),
        "roll_mean_14": roll(14, np.mean),
        "roll_mean_28": roll(28, np.mean),
        "roll_std_7": roll(7, np.std),
        "roll_std_28": roll(28, np.std),
        "zero_prop_28": (1.0 - roll(28, lambda x, axis: (x > 0).mean(axis=axis))) if origin >= 28 else nan,
        "days_since_sale": days_since,
        "age_since_first_sale": age,
    }


def seasonal_lags(tail: np.ndarray, origin: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Values at the last same-weekday-as-target day <= origin, and one week before it."""
    m, L = tail.shape
    day = origin + h - 7 * int(np.ceil(h / 7))  # in (origin-7, origin]
    assert origin - 7 < day <= origin
    out = []
    for dd in (day, day - 7):
        back = origin - dd  # days before origin
        if dd >= 1 and back < L:
            out.append(tail[:, L - 1 - back].astype(np.float32))
        else:
            out.append(np.full(m, np.nan, dtype=np.float32))
    return out[0], out[1]


def build_examples(
    panel: Panel,
    origin: int,
    series_idx: np.ndarray,
    horizons: list[int],
    groups: list[str],
    enc: Encoders,
    *,
    with_target: bool = True,
    price_scenario: str = "observed",
) -> pd.DataFrame:
    a = panel.as_of(origin)
    sel = np.asarray(series_idx, dtype=np.int64)
    m, H = len(sel), len(horizons)
    cols = columns_for(groups, price_scenario)
    if origin + max(horizons) > len(panel.calendar):
        raise CalendarCoverageError(
            f"calendar ends at day {len(panel.calendar)}; need day {origin + max(horizons)}"
        )

    L = min(origin, LOOKBACK)
    tail = a.sales[:, origin - L : origin][sel]
    hist = history_features(tail, origin)
    data: dict[str, np.ndarray] = {}
    data["series_idx"] = np.tile(sel.astype(np.int32), H)
    data["origin"] = np.full(m * H, origin, dtype=np.int32)
    hs = np.repeat(np.asarray(horizons, dtype=np.int16), m)
    data["h"] = hs
    for k, v in hist.items():
        data[k] = np.tile(v, H)
    s1, s2 = zip(*[seasonal_lags(tail, origin, h) for h in horizons], strict=True)
    data["seas_lag_1"] = np.concatenate(s1)
    data["seas_lag_2"] = np.concatenate(s2)

    ids = panel.ids.iloc[sel]
    for c in ID_COLS:
        data[c] = np.tile(_encode(ids[c], enc.ids[c]), H)

    days = origin + np.asarray(horizons)  # 1-based target days
    cal = panel.calendar.iloc[days - 1]
    dd = np.repeat
    data["t_dow"] = dd(cal["wday"].to_numpy(dtype=np.int16), m)
    data["t_month"] = dd(cal["month"].to_numpy(dtype=np.int16), m)
    data["t_dom"] = dd(pd.to_datetime(cal["date"]).dt.day.to_numpy(dtype=np.int16), m)
    data["t_event_type_1"] = dd(_encode(cal["event_type_1"].fillna("none").astype(str), enc.event_type), m)
    data["t_event_name_1"] = dd(_encode(cal["event_name_1"].fillna("none").astype(str), enc.event_name), m)

    if "price" in groups:
        pf, age = a.ffilled_price()
        w = (
            a.price_last_block
        )  # latest COMPLETED block: a block price is known only after the block has ended
        if w < 0:  # nothing completed yet -> price unavailable
            cur = np.full(m, np.nan, dtype=np.float32)
            prev, cur_age = cur.copy(), cur.copy()
            mx = np.full(m, np.nan)
        else:
            cur, cur_age = pf[sel, w], age[sel, w]
            prev = pf[sel, w - 4] if w >= 4 else np.full(m, np.nan, dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mx = np.nanmax(np.where(np.isnan(a.price[sel]), -np.inf, a.price[sel]), axis=1)
                mx = np.where(np.isfinite(mx), mx, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chg = cur / prev - 1
            rel = cur / mx
        data["price_origin"] = np.tile(cur, H)
        data["price_age_weeks"] = np.tile(cur_age, H)
        data["price_missing"] = np.tile(np.isnan(cur).astype(np.int8), H)
        data["price_chg_4w"] = np.tile(chg.astype(np.float32), H)
        data["price_rel_max"] = np.tile(rel.astype(np.float32), H)
        if price_scenario == "planned":
            # SCENARIO ONLY: reads prices of the target week (future relative to the origin).
            tp = np.concatenate([panel.price_weekly[sel, panel.day_week_idx[d - 1]] for d in days])
            data["target_price"] = tp
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                data["target_price_chg"] = tp / np.tile(cur, H) - 1

    df = pd.DataFrame({k: data[k] for k in ["series_idx", "origin", *cols]})
    if with_target:
        tgt = np.full((m, H), np.nan, dtype=np.float32)
        for j, d in enumerate(days):
            if d <= panel.n_days:
                tgt[:, j] = panel.sales[sel, d - 1]
        df["y"] = tgt.T.reshape(-1)
    return df


def build_training_frame(
    panel: Panel,
    origins: list[int],
    series_idx: np.ndarray,
    horizons: list[int],
    groups: list[str],
    enc: Encoders,
    price_scenario: str = "observed",
) -> pd.DataFrame:
    parts = []
    for o in origins:
        # training rows exist only for products observed by the training origin (no knowledge of later launches)
        sel_o = np.intersect1d(np.asarray(series_idx), panel.eligible_at(o))
        if len(sel_o) == 0:
            continue
        df = build_examples(
            panel, o, sel_o, horizons, groups, enc, with_target=True, price_scenario=price_scenario
        )
        parts.append(df[df["y"].notna()])
    return pd.concat(parts, ignore_index=True)
