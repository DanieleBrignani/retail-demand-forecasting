"""SYNTHETIC retail-shaped data for tests ONLY. Never used for reported benchmark results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STORES = [("UK_1", "UK"), ("INT_1", "INT")]
DEPT, CAT = "RETAIL_1", "RETAIL"


def make_calendar(n_obs: int, n_future: int = 56) -> pd.DataFrame:
    n = n_obs + n_future
    dates = pd.date_range("2010-01-02", periods=n, freq="D")
    cal = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "d": [f"d_{i}" for i in range(1, n + 1)]})
    cal["week_id"] = 1001 + (np.arange(n) // 7)
    cal["weekday"] = dates.day_name()
    cal["wday"] = ((dates.dayofweek + 2) % 7) + 1  # Saturday = 1
    cal["month"] = dates.month
    cal["year"] = dates.year
    ev = np.full(n, None, dtype=object)
    et = np.full(n, None, dtype=object)
    for i in range(20, n, 45):
        ev[i], et[i] = ("Spring Holiday" if (i // 45) % 2 == 0 else "Winter Holiday"), "National"
    cal["event_name_1"], cal["event_type_1"] = ev, et
    cal["event_name_2"], cal["event_type_2"] = None, None
    return cal


def make_dataset(n_obs: int = 300, n_items: int = 36, seed: int = 0, n_future: int = 56):
    """Returns (calendar, sales_wide, prices). Contains: intermittent series, a late-launch series
    (leading zeros), an all-zero series, and series with missing price weeks."""
    rng = np.random.default_rng(seed)
    cal = make_calendar(n_obs, n_future)
    dow = cal["wday"].to_numpy()[:n_obs]
    rows, mat = [], []
    for k in range(n_items):
        item = f"{DEPT}_{k + 1:03d}"
        for store, state in STORES:
            base = rng.choice([0.2, 0.6, 2.0, 6.0, 15.0])
            weekly = 1 + 0.4 * np.sin(2 * np.pi * dow / 7 + rng.uniform(0, 6))
            trend = 1 + 0.001 * np.arange(n_obs) * rng.uniform(-0.5, 1)
            y = rng.poisson(np.maximum(base * weekly * trend, 0.01)).astype(np.int16)
            rows.append(
                {
                    "id": f"{item}_{store}",
                    "item_id": item,
                    "dept_id": DEPT,
                    "cat_id": CAT,
                    "store_id": store,
                    "state_id": state,
                }
            )
            mat.append(y)
    mat = np.stack(mat)
    mat[3, :120] = 0  # late launch (leading zeros)
    mat[5, :] = 0  # all-zero series
    ids = pd.DataFrame(rows)
    sales = pd.concat([ids, pd.DataFrame(mat, columns=[f"d_{i}" for i in range(1, n_obs + 1)])], axis=1)
    weeks = np.sort(cal["week_id"].unique())
    prow = []
    for i, r in ids.iterrows():
        p0 = rng.uniform(1, 8)
        for j, w in enumerate(weeks):
            if i == 7 and j < 10:
                continue  # price listed only from week 10 (missing early prices)
            if i == 9 and 20 <= j < 25:
                continue  # stale-price gap
            prow.append((r["store_id"], r["item_id"], int(w), round(p0 * (0.8 if j % 13 == 0 else 1.0), 2)))
    prices = pd.DataFrame(prow, columns=["store_id", "item_id", "week_id", "sell_price"])
    return cal, sales, prices


def write_csvs(folder: Path, n_obs: int = 300, sales_name: str = "sales_daily.csv", **kw) -> Path:
    cal, sales, prices = make_dataset(n_obs, **kw)
    folder.mkdir(parents=True, exist_ok=True)
    cal.to_csv(folder / "calendar.csv", index=False)
    sales.to_csv(folder / sales_name, index=False)
    prices.to_csv(folder / "sell_prices.csv", index=False)
    return folder


def make_panel(n_obs: int = 300, **kw):
    from retailfc.panel import build_panel

    cal, sales, prices = make_dataset(n_obs, **kw)
    return build_panel(cal, sales, prices, "synthetic-test")
