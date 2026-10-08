"""Adapter: UCI *Online Retail II* transactions -> the platform's canonical CSVs (calendar, sell_prices, sales_daily).

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0), one UK online giftware retailer,
2009-12-01 .. 2011-12-09. It has no physical stores, regions or product categories.

Every modelling-relevant choice is explicit and recorded in ``adapter_report.json``:

* Sales = sum of ``Quantity`` per product code, store and calendar day, over invoices that are not cancellations
  (``Invoice`` not starting with ``C``) with ``Quantity > 0`` and ``Price > 0``. Cancellations/returns are excluded, not netted.
* Product codes = ``^\\d{5}[A-Za-z]{0,2}$``. Non-product codes (postage, fees, adjustments, bank charges, gift-voucher
  style codes, ...) are excluded and counted.
* "Stores" are two sales channels, not physical stores: ``UK_1`` (Country == United Kingdom) and ``INT_1`` (every other
  country). ``state_id`` = ``UK`` / ``INT``. There is a single department (``RETAIL_1``) and category (``RETAIL``): the source
  has no category, and none is invented.
* Every product code appears in both channels (zero-filled when never sold there), i.e. a complete item x channel panel.
  No series is dropped on the basis of future sales.
* Days without any transaction (closures, Saturdays, bank holidays) are zero for every series: they are structural zeros, not
  missing data, and the weekly/holiday features can explain them.
* Prices = median ``Price`` of the week's sale lines per (store, item); weeks without sales have **no price row** (never 0).
* Calendar events = England & Wales public holidays from the ``holidays`` package (an *assumption*: the source has no event
  calendar).
* ``week_id`` = 1 + k for the k-th consecutive 7-day block starting at the first date (a plain week key, not a
  retail fiscal week).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCT_RE = re.compile(r"^\d{5}[A-Za-z]{0,2}$")
SOURCE_URL = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
LICENSE = "CC BY 4.0"
UK = "United Kingdom"
DEPT, CAT = "RETAIL_1", "RETAIL"
STORES = {"UK_1": "UK", "INT_1": "INT"}


def read_xlsx(path: str | Path) -> pd.DataFrame:
    """Read both sheets of online_retail_II.xlsx. Mixed-type columns are normalised to text."""
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    for c in ("Invoice", "StockCode", "Description", "Country"):
        df[c] = df[c].astype(str)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    df = df.assign(
        StockCode=df["StockCode"].str.strip()
    )  # the source has codes such as '47503J ' with a trailing blank
    cancel = df["Invoice"].str.startswith("C")
    qty_bad = ~(df["Quantity"] > 0)  # non-positive OR missing quantity
    missing_ts = df["InvoiceDate"].isna() | df["Country"].isin(["nan", "None", ""])
    price_bad = ~(df["Price"] > 0)
    product = df["StockCode"].str.match(PRODUCT_RE)
    keep = ~cancel & ~qty_bad & ~price_bad & product & ~missing_ts
    report = {
        "rows_in": int(n0),
        "excluded_cancellation_invoices": int(cancel.sum()),
        "excluded_nonpositive_quantity": int(qty_bad.sum()),
        "excluded_nonpositive_or_missing_price": int(price_bad.sum()),
        "excluded_non_product_codes": int((~product).sum()),
        "excluded_missing_timestamp_or_country": int(missing_ts.sum()),
        "rows_kept": int(keep.sum()),
        "non_product_codes_seen": sorted(set(df.loc[~product, "StockCode"]))[:60],
        "note": "exclusion categories overlap; rows_kept is the intersection of all keep-conditions",
    }
    out = df.loc[keep, ["StockCode", "Quantity", "InvoiceDate", "Price", "Country"]].copy()
    out["StockCode"] = out["StockCode"].str.upper()
    out["store_id"] = np.where(out["Country"] == UK, "UK_1", "INT_1")
    out["date"] = pd.to_datetime(out["InvoiceDate"]).dt.normalize()
    return out, report


def build_calendar(first: pd.Timestamp, n_obs: int, n_future: int) -> pd.DataFrame:
    import holidays

    n = n_obs + n_future
    dates = pd.date_range(first, periods=n, freq="D")
    hol = holidays.country_holidays("GB", subdiv="ENG", years=sorted({d.year for d in dates}))
    names = np.array([hol.get(d.date()) for d in dates], dtype=object)
    cal = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "d": [f"d_{i}" for i in range(1, n + 1)]})
    cal["week_id"] = 1 + np.arange(n) // 7
    cal["weekday"] = dates.day_name()
    cal["wday"] = ((dates.dayofweek + 2) % 7) + 1  # Saturday = 1 (weekday code; any consistent coding works)
    cal["month"] = dates.month
    cal["year"] = dates.year
    cal["event_name_1"] = names
    cal["event_type_1"] = np.where(pd.isna(names), None, "National")
    cal["event_name_2"], cal["event_type_2"] = None, None
    return cal


def to_canonical_frames(
    tx: pd.DataFrame, n_future: int = 56
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    first, last = tx["date"].min(), tx["date"].max()
    n_obs = int((last - first).days) + 1
    day = (tx["date"] - first).dt.days.to_numpy()
    items = sorted(tx["StockCode"].unique())
    item_ix = {c: i for i, c in enumerate(items)}
    stores = list(STORES)
    store_ix = {s: i for i, s in enumerate(stores)}
    ii = tx["StockCode"].map(item_ix).to_numpy()
    si = tx["store_id"].map(store_ix).to_numpy()

    mat = np.zeros((len(items) * len(stores), n_obs), dtype=np.int64)
    row = ii * len(stores) + si
    np.add.at(mat, (row, day), tx["Quantity"].to_numpy(dtype=np.int64))

    rows = []
    for c in items:
        for s in stores:
            item_id = f"{DEPT}_{c}"
            rows.append(
                {
                    "id": f"{item_id}_{s}",
                    "item_id": item_id,
                    "dept_id": DEPT,
                    "cat_id": CAT,
                    "store_id": s,
                    "state_id": STORES[s],
                }
            )
    ids = pd.DataFrame(rows)
    sales = pd.concat([ids, pd.DataFrame(mat, columns=[f"d_{i}" for i in range(1, n_obs + 1)])], axis=1)

    wk = 1 + day // 7
    pr = (
        pd.DataFrame(
            {
                "store_id": tx["store_id"].to_numpy(),
                "item": tx["StockCode"].to_numpy(),
                "week_id": wk,
                "sell_price": tx["Price"].to_numpy(),
            }
        )
        .groupby(["store_id", "item", "week_id"], as_index=False)["sell_price"]
        .median()
    )
    pr["sell_price"] = pr["sell_price"].round(2).clip(lower=0.01)
    pr["item_id"] = DEPT + "_" + pr["item"]
    prices = pr[["store_id", "item_id", "week_id", "sell_price"]]

    cal = build_calendar(first, n_obs, n_future)
    report = {
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "calendar_days_observed": n_obs,
        "calendar_days_future": n_future,
        "n_products": len(items),
        "n_series": len(ids),
        "days_with_any_transaction": int(len(set(day.tolist()))),
        "zero_share_of_series_days": float((mat == 0).mean()),
        "series_all_zero": int((mat.sum(axis=1) == 0).sum()),
        "uk_row_share": float((tx["store_id"] == "UK_1").mean()),
    }
    return cal, sales, prices, report


def adapt(
    xlsx: str | Path, out_dir: str | Path, n_future: int = 56, transactions: pd.DataFrame | None = None
) -> dict:
    tx_raw = transactions if transactions is not None else read_xlsx(xlsx)
    tx, rep_clean = clean(tx_raw)
    cal, sales, prices, rep = to_canonical_frames(tx, n_future)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cal.to_csv(out / "calendar.csv", index=False)
    sales.to_csv(out / "sales_daily.csv", index=False)
    prices.to_csv(out / "sell_prices.csv", index=False)
    report = {
        "source": SOURCE_URL,
        "license": LICENSE,
        "cleaning": rep_clean,
        "panel": rep,
        "assumptions": [
            "stores are sales channels (UK vs international), not physical stores",
            "single department/category: the source has none",
            "events = England & Wales public holidays (holidays package), not source data",
            "week_id = 7-day blocks from the first date",
            "cancellations/returns excluded, not netted",
        ],
    }
    (out / "adapter_report.json").write_text(json.dumps(report, indent=2))
    return report
