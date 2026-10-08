"""Source validation for the three canonical input files (calendar, daily sales, weekly prices). Pure functions over DataFrames; no I/O.

Severity: ``error`` checks block ingestion; ``warn`` checks are recorded in the manifest.
Zero sales are valid observations; missing observations (NaN / absent d_ column) are errors.
Absent prices are *not* imputed anywhere -- ``price_coverage`` only reports them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SCHEMA_VERSION = "retail-raw-v1"

CALENDAR_COLS = [
    "date",
    "week_id",
    "weekday",
    "wday",
    "month",
    "year",
    "d",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]
SALES_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
PRICE_COLS = ["store_id", "item_id", "week_id", "sell_price"]
_D_RE = re.compile(r"^d_(\d+)$")


@dataclass
class Check:
    name: str
    passed: bool
    severity: str = "error"
    detail: str = ""


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", severity: str = "error") -> None:
        self.checks.append(Check(name, bool(passed), severity, detail))

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}

    def merge(self, other: ValidationReport) -> ValidationReport:
        self.checks.extend(other.checks)
        return self


def _cols(rep: ValidationReport, df: pd.DataFrame, expected: list[str], label: str) -> bool:
    missing = [c for c in expected if c not in df.columns]
    rep.add(f"{label}.required_columns", not missing, f"missing={missing}")
    extra = [c for c in df.columns if c not in expected and not _D_RE.match(str(c))]
    rep.add(f"{label}.unexpected_columns", not extra, f"extra={extra[:10]}", severity="warn")
    return not missing


def validate_calendar(cal: pd.DataFrame) -> ValidationReport:
    rep = ValidationReport()
    if not _cols(rep, cal, CALENDAR_COLS, "calendar"):
        return rep
    dates = pd.to_datetime(cal["date"], errors="coerce")
    rep.add("calendar.date_parseable", dates.notna().all(), f"bad={int(dates.isna().sum())}")
    rep.add("calendar.unique_date", dates.is_unique, f"dup={int(dates.duplicated().sum())}")
    rep.add("calendar.unique_d", cal["d"].is_unique, f"dup={int(cal['d'].duplicated().sum())}")
    nums = cal["d"].astype(str).str.extract(_D_RE)[0]
    rep.add("calendar.d_format", nums.notna().all(), "d must look like d_<int>")
    if nums.notna().all() and dates.notna().all():
        n = nums.astype(int).to_numpy()
        order = np.argsort(dates.to_numpy())
        rep.add(
            "calendar.d_matches_date_order",
            bool(np.all(np.diff(n[order]) == 1)) and n[order][0] == 1,
            "d_k must increase by 1 per consecutive day, starting at d_1",
        )
        gaps = np.diff(dates.to_numpy()[order]).astype("timedelta64[D]").astype(int)
        rep.add("calendar.no_missing_dates", bool(np.all(gaps == 1)), f"gaps={int((gaps != 1).sum())}")
        rep.add(
            "calendar.weekday_consistent",
            bool((dates.dt.day_name().to_numpy() == cal["weekday"].to_numpy()).all()),
            "weekday name vs date",
        )
        # week id -> exactly 7 days
        per_week = cal.groupby("week_id").size()
        rep.add(
            "calendar.week_at_most_7_days",
            bool((per_week <= 7).all()),
            f"weeks>7: {int((per_week > 7).sum())}",
        )
        rep.add(
            "calendar.partial_edge_weeks",
            bool((per_week.iloc[1:-1] == 7).all()),
            f"partial weeks (first/last may be partial): {int((per_week < 7).sum())}",
            severity="warn",
        )
    return rep


def validate_sales(sales: pd.DataFrame, calendar: pd.DataFrame | None = None) -> ValidationReport:
    rep = ValidationReport()
    if not _cols(rep, sales, SALES_ID_COLS, "sales"):
        return rep
    dcols = [c for c in sales.columns if _D_RE.match(str(c))]
    rep.add("sales.has_day_columns", len(dcols) > 0, f"n_day_cols={len(dcols)}")
    if not dcols:
        return rep
    nums = np.array([int(_D_RE.match(c).group(1)) for c in dcols])
    rep.add(
        "sales.day_columns_contiguous",
        bool(np.array_equal(nums, np.arange(1, len(nums) + 1))),
        "d_1..d_N consecutive and ordered",
    )
    rep.add(
        "sales.unique_item_store",
        not sales.duplicated(["item_id", "store_id"]).any(),
        f"dup={int(sales.duplicated(['item_id', 'store_id']).sum())}",
    )
    rep.add("sales.unique_id", sales["id"].is_unique, "")
    vals = sales[dcols].to_numpy()
    rep.add("sales.no_missing_observations", not pd.isna(vals).any(), f"nan={int(pd.isna(vals).sum())}")
    rep.add("sales.no_negative", not (vals < 0).any(), f"neg={int((vals < 0).sum())}")
    rep.add("sales.integer_valued", bool(np.all(np.mod(np.nan_to_num(vals), 1) == 0)), "")
    for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        rep.add(
            f"sales.{c}_not_null", sales[c].notna().all() and (sales[c].astype(str).str.len() > 0).all(), ""
        )
    s = sales
    rep.add(
        "sales.dept_prefix_of_item",
        bool(s.apply(lambda r: r["item_id"].startswith(r["dept_id"]), axis=1).all()) if len(s) else True,
        "item_id = <dept_id>_<n>",
    )
    rep.add(
        "sales.state_prefix_of_store",
        bool(s.apply(lambda r: r["store_id"].startswith(r["state_id"]), axis=1).all()) if len(s) else True,
        "store_id = <state>_<n>",
    )
    if calendar is not None:
        ndays = len(calendar)
        rep.add(
            "sales.calendar_coverage", len(dcols) <= ndays, f"sales_days={len(dcols)} calendar_days={ndays}"
        )
    rep.add("sales.zero_share_reported", True, f"zero_share={(vals == 0).mean():.4f}", severity="warn")
    return rep


def validate_prices(
    prices: pd.DataFrame, calendar: pd.DataFrame | None = None, sales_ids: pd.DataFrame | None = None
) -> ValidationReport:
    rep = ValidationReport()
    if not _cols(rep, prices, PRICE_COLS, "prices"):
        return rep
    key = ["store_id", "item_id", "week_id"]
    rep.add(
        "prices.unique_store_item_week",
        not prices.duplicated(key).any(),
        f"dup={int(prices.duplicated(key).sum())}",
    )
    sp = prices["sell_price"]
    rep.add("prices.price_not_null", sp.notna().all(), f"null={int(sp.isna().sum())}")
    rep.add("prices.price_positive", bool((sp.dropna() > 0).all()), f"nonpos={int((sp.dropna() <= 0).sum())}")
    rep.add("prices.price_finite", bool(np.isfinite(sp.dropna().to_numpy()).all()), "")
    if calendar is not None:
        weeks = set(calendar["week_id"].unique())
        bad = ~prices["week_id"].isin(weeks)
        rep.add("prices.week_in_calendar", not bad.any(), f"orphan_rows={int(bad.sum())}")
        # join cardinality: price row -> calendar week is many days per week; ensure days-per-week constant
        rep.add(
            "prices.join_week_cardinality",
            bool((calendar.groupby("week_id").size() <= 7).all()),
            "each price-week joins at most 7 calendar days (no fan-out)",
        )
    if sales_ids is not None:
        pairs = set(zip(sales_ids["store_id"], sales_ids["item_id"], strict=False))
        pp = set(zip(prices["store_id"], prices["item_id"], strict=False))
        orphan = len(pp - pairs)
        rep.add("prices.referential_integrity", orphan == 0, f"price (store,item) not in sales: {orphan}")
        rep.add(
            "prices.series_without_price",
            True,
            f"sales series with no price rows: {len(pairs - pp)}",
            severity="warn",
        )
    return rep


def validate_all(calendar: pd.DataFrame, sales: pd.DataFrame, prices: pd.DataFrame) -> ValidationReport:
    rep = validate_calendar(calendar)
    rep.merge(validate_sales(sales, calendar))
    rep.merge(
        validate_prices(
            prices,
            calendar,
            sales[[c for c in ("store_id", "item_id") if c in sales.columns]]
            if {"store_id", "item_id"} <= set(sales.columns)
            else None,
        )
    )
    return rep


def price_coverage(prices: pd.DataFrame, calendar: pd.DataFrame, n_series: int) -> dict:
    """Share of (series, week) cells with a price; absent cells stay absent (never zero-filled)."""
    n_weeks = calendar["week_id"].nunique()
    return {
        "price_rows": int(len(prices)),
        "cells": int(n_series * n_weeks),
        "coverage": float(len(prices) / max(1, n_series * n_weeks)),
    }
