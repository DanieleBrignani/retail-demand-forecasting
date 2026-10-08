"""Dense in-memory view of the item x store x day panel and an *as-of* view that enforces the information boundary.

Day numbers are 1-based (``d_k`` -> k).  ``sales[:, k-1]`` is day k.  Only the first ``n_days`` calendar
days have sales; calendar rows beyond that exist (future dates) but have no sales.
Prices are kept per 7-day block (``price_weekly[series, block]``; NaN = no price recorded) and are never
zero-filled or back-filled.  A block price is the median over ALL the block's transaction lines, so it only becomes
known when the block has ended: ``as_of`` exposes completed blocks only (see ``AsOf.price_last_block``).
It is a transaction-derived price, not an externally observed list price.

All feature/baseline/model code must read data through :meth:`Panel.as_of`, which exposes
sales up to the origin and prices up to the origin's week only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
SALES_DTYPE = np.int32  # daily units of ONE series-day can exceed 32,767 (the real data has 74,215 and 80,995): int16 silently wraps


def read_sales_csv(path, n_days: int) -> pd.DataFrame:
    """Read ``sales_daily.csv`` without any narrowing: values are parsed as int64, then cast to int32 only after checking the
    range, so a large order can never wrap around (a former int16 read turned 74,215 into 8,679 and lost 131,072 units)."""
    sales = pd.read_csv(path, dtype={f"d_{i}": "int64" for i in range(1, n_days + 1)})
    cols = [c for c in sales.columns if str(c).startswith("d_")]
    mx, mn = int(sales[cols].to_numpy().max()), int(sales[cols].to_numpy().min())
    lim = np.iinfo(SALES_DTYPE)
    if mx > lim.max or mn < 0:
        raise ValueError(f"sales out of the supported range [0, {lim.max}]: min {mn}, max {mx}")
    sales[cols] = sales[cols].astype(SALES_DTYPE)
    return sales


@dataclass
class Panel:
    ids: pd.DataFrame  # n rows: item_id, dept_id, cat_id, store_id, state_id
    sales: np.ndarray  # (n, n_days) int32 units per series-day
    calendar: pd.DataFrame  # one row per day (observed + future), index 0 == d_1
    price_weekly: np.ndarray  # (n, n_weeks) float32, NaN = absent
    dataset_version: str = "unversioned"
    week_ids: np.ndarray = field(init=False)
    day_week_idx: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.ids = self.ids.reset_index(drop=True)
        self.calendar = self.calendar.reset_index(drop=True)
        self.week_ids = np.sort(self.calendar["week_id"].unique())
        self.day_week_idx = np.searchsorted(self.week_ids, self.calendar["week_id"].to_numpy())
        assert self.price_weekly.shape == (len(self.ids), len(self.week_ids)), "price matrix shape mismatch"
        assert self.sales.shape[0] == len(self.ids)
        assert self.sales.shape[1] <= len(self.calendar), "sales longer than calendar"
        self.dates = pd.to_datetime(self.calendar["date"]).to_numpy()
        # states come from the data (whatever ``state_id`` holds)
        self.state_names = sorted(self.ids["state_id"].unique())
        self.state_code = self.ids["state_id"].map({s: i for i, s in enumerate(self.state_names)}).to_numpy()

    @property
    def n_series(self) -> int:
        return len(self.ids)

    @property
    def n_days(self) -> int:
        return self.sales.shape[1]

    # ---- series-universe policy (docs/data_contract.md): a product exists at origin o iff it has a recorded sale on or before o
    def product_first_day(self) -> np.ndarray:
        """Per series: 1-based day of the first recorded sale of its PRODUCT in any channel (``n_days + 1`` = never sold)."""
        if getattr(self, "_pfd", None) is None:
            pos = self.sales > 0
            first = np.where(pos.any(axis=1), pos.argmax(axis=1) + 1, self.n_days + 1)
            by_item = pd.Series(first).groupby(self.ids["item_id"].to_numpy()).transform("min").to_numpy()
            self._pfd = by_item.astype(np.int64)
        return self._pfd

    def eligible_at(self, origin: int) -> np.ndarray:
        """Indices of the series whose product was observed (>= 1 recorded sale) by ``origin``.  No external catalogue exists,
        so products that first sell later are NOT part of the universe at ``origin`` (cold start; evaluated separately)."""
        return np.flatnonzero(self.product_first_day() <= origin)

    def subset(self, series_idx: np.ndarray) -> Panel:
        series_idx = np.asarray(series_idx)
        return Panel(
            self.ids.iloc[series_idx],
            self.sales[series_idx],
            self.calendar,
            self.price_weekly[series_idx],
            self.dataset_version,
        )

    def as_of(self, origin: int) -> AsOf:
        if not (1 <= origin <= self.n_days):
            raise ValueError(f"origin {origin} outside observed range 1..{self.n_days}")
        w = int(self.day_week_idx[origin - 1])
        # the origin day's block is complete only if the NEXT calendar day belongs to the next block
        complete = origin < len(self.day_week_idx) and int(self.day_week_idx[origin]) != w
        last = w if complete else w - 1  # index of the latest completed block (-1: none yet)
        return AsOf(
            panel=self,
            origin=origin,
            sales=self.sales[:, :origin],
            price=self.price_weekly[:, : last + 1],
            week_idx=w,
            price_last_block=last,
        )

    def date_of(self, day: int) -> np.datetime64:
        return self.dates[day - 1]

    def key_frame(self) -> pd.DataFrame:
        return self.ids[["item_id", "store_id"]]


@dataclass
class AsOf:
    """Everything the forecasting process may know at ``origin`` (calendar is known in advance)."""

    panel: Panel
    origin: int
    sales: np.ndarray  # (n, origin)
    price: np.ndarray  # (n, price_last_block+1): prices of COMPLETED blocks only
    week_idx: int  # block containing the origin day
    price_last_block: int  # latest completed block (-1 = none): the price information boundary

    def ffilled_price(self) -> tuple[np.ndarray, np.ndarray]:
        """Last observed price carried forward (never backward) and its age in weeks (NaN if none yet)."""
        p = self.price
        valid = ~np.isnan(p)
        W = p.shape[1]
        if W == 0:  # no completed block yet: every price is unavailable (NaN), never zero
            return p.astype(np.float32), p.astype(np.float32)
        idx = np.where(valid, np.arange(W)[None, :], -1)
        idx = np.maximum.accumulate(idx, axis=1)
        safe = np.where(idx < 0, 0, idx)
        filled = np.take_along_axis(p, safe, axis=1)
        filled[idx < 0] = np.nan
        age = (np.arange(W)[None, :] - idx).astype(np.float32)
        age[idx < 0] = np.nan
        return filled.astype(np.float32), age


# ----------------------------------------------------------------------------- construction
def build_panel(
    calendar: pd.DataFrame,
    sales_wide: pd.DataFrame,
    prices: pd.DataFrame,
    dataset_version: str = "unversioned",
) -> Panel:
    dcols = [c for c in sales_wide.columns if str(c).startswith("d_")]
    ids = sales_wide[ID_COLS].copy()
    raw_mat = sales_wide[dcols].to_numpy(dtype=np.int64)
    if raw_mat.max() > np.iinfo(SALES_DTYPE).max or raw_mat.min() < 0:
        raise ValueError("sales outside the supported integer range")
    mat = raw_mat.astype(SALES_DTYPE)
    if int(mat.sum(dtype=np.int64)) != int(
        raw_mat.sum()
    ):  # integrity: the cast must not change a single unit
        raise ValueError("sales changed while building the panel")
    cal = calendar.copy()
    cal["day_num"] = cal["d"].str.slice(2).astype(int)
    cal = cal.sort_values("day_num").reset_index(drop=True)
    week_ids = np.sort(cal["week_id"].unique())
    key = pd.MultiIndex.from_frame(ids[["store_id", "item_id"]])
    row = key.get_indexer(pd.MultiIndex.from_frame(prices[["store_id", "item_id"]]))
    col = np.searchsorted(week_ids, prices["week_id"].to_numpy())
    keep = row >= 0
    pw = np.full((len(ids), len(week_ids)), np.nan, dtype=np.float32)
    pw[row[keep], col[keep]] = prices["sell_price"].to_numpy(dtype=np.float32)[keep]
    return Panel(ids, mat, cal, pw, dataset_version)


def load_panel_from_raw(version: str | None = None, *, max_series: int | None = None, seed: int = 0) -> Panel:
    from .ingestion import raw

    version = version or raw.latest_version()
    mf = raw.verify_version(version)
    d = raw.raw_root() / version
    sales_name = "sales_daily.csv"
    cal = pd.read_csv(d / "calendar.csv")
    sales = read_sales_csv(d / sales_name, mf["observed_sales"]["n_days"])
    prices = pd.read_csv(d / "sell_prices.csv", dtype={"sell_price": "float32"})
    panel = build_panel(cal, sales, prices, version)
    if max_series and max_series < panel.n_series:
        rng = np.random.default_rng(seed)
        panel = panel.subset(np.sort(rng.choice(panel.n_series, max_series, replace=False)))
    return panel


def cache_path(version: str) -> Path:
    from . import config as C

    return C.data_dir() / "interim" / f"{version}.npz"
