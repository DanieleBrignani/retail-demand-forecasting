"""Single source of truth for features: definition, time reference, availability, missing policy.

``docs/feature_dictionary.md`` is generated from this registry (``retailfc docs feature-dictionary``).
Notation: o = forecast origin day (last day with sales), h = horizon (1..28), t = o + h target day,
y_d = sales on day d. All sales references are <= o.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

FEATURE_CODE_VERSION = "1"
PRICE_CODE_VERSION = "2"  # fv2: price features use COMPLETED 7-day blocks only (fv1 leaked future transactions of the origin's block)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    definition: str
    time_ref: str
    availability: str
    missing: str
    meaning: str
    categorical: bool = False


def _h(
    name,
    definition,
    time_ref,
    missing,
    meaning,
    group="history",
    availability="Sales up to and including the origin day o.",
):
    return FeatureSpec(name, group, definition, time_ref, availability, missing, meaning)


_ID_AV = "Static catalogue attributes; encoders fitted on training data only, unseen id -> -1 (treated as missing)."
_CAL_AV = (
    "Calendar is published in advance; assumed known >= 28 days ahead of the origin "
    "(the calendar table covers the target dates)."
)

SPECS: list[FeatureSpec] = [
    _h("last_sale", "y_o", "day o", "never missing (o >= 1)", "Most recent demand signal."),
    _h("lag_7", "y_{o-7}", "day o-7", "NaN if o <= 7", "Same weekday as the origin one week earlier."),
    _h("lag_14", "y_{o-14}", "day o-14", "NaN if o <= 14", "Two weeks back."),
    _h("lag_28", "y_{o-28}", "day o-28", "NaN if o <= 28", "Four weeks back."),
    _h("roll_mean_7", "mean(y_{o-6..o})", "days o-6..o", "NaN if o < 7", "Short-term level."),
    _h("roll_mean_14", "mean(y_{o-13..o})", "days o-13..o", "NaN if o < 14", "Medium-term level."),
    _h("roll_mean_28", "mean(y_{o-27..o})", "days o-27..o", "NaN if o < 28", "Monthly level."),
    _h("roll_std_7", "std(y_{o-6..o}), ddof=0", "days o-6..o", "NaN if o < 7", "Short-term volatility."),
    _h("roll_std_28", "std(y_{o-27..o}), ddof=0", "days o-27..o", "NaN if o < 28", "Monthly volatility."),
    _h(
        "zero_prop_28",
        "share of zero-sales days in o-27..o",
        "days o-27..o",
        "NaN if o < 28",
        "Intermittency; zero sales do not prove zero demand.",
    ),
    _h(
        "days_since_sale",
        "o - (last day <= o with y>0)",
        "days <= o",
        "NaN if no positive sale yet",
        "Recency of demand; large values may mean inactivity or stock-out (not observable).",
    ),
    _h(
        "age_since_first_sale",
        "o - (first day with y>0) + 1; 0 if none",
        "days <= o",
        "0 if none",
        "Apparent time on sale; first positive sale is not necessarily product launch.",
    ),
    _h(
        "seas_lag_1",
        "y_{t-7*ceil(h/7)}",
        "last same-weekday-as-target day <= o",
        "NaN if day < 1",
        "Most recent observation of the target's weekday; anchored to the origin, never to t.",
    ),
    _h(
        "seas_lag_2",
        "y_{t-7*ceil(h/7)-7}",
        "same weekday, one week earlier",
        "NaN if day < 1",
        "Weekly pattern persistence.",
    ),
    FeatureSpec(
        "item_id", "ids", "integer code of item_id", "static", _ID_AV, "unseen -> -1", "Item identity.", True
    ),
    FeatureSpec(
        "dept_id", "ids", "integer code of dept_id", "static", _ID_AV, "unseen -> -1", "Department.", True
    ),
    FeatureSpec(
        "cat_id", "ids", "integer code of cat_id", "static", _ID_AV, "unseen -> -1", "Category.", True
    ),
    FeatureSpec(
        "store_id", "ids", "integer code of store_id", "static", _ID_AV, "unseen -> -1", "Store.", True
    ),
    FeatureSpec(
        "state_id", "ids", "integer code of state_id", "static", _ID_AV, "unseen -> -1", "State.", True
    ),
    FeatureSpec(
        "h",
        "horizon",
        "forecast horizon in days (1..28)",
        "t - o",
        "Known by construction.",
        "never missing",
        "Lets the direct global model learn horizon-specific behaviour.",
    ),
    FeatureSpec(
        "t_dow",
        "calendar",
        "weekday of target day (wday 1..7)",
        "day t",
        _CAL_AV,
        "never missing",
        "Weekly seasonality.",
        True,
    ),
    FeatureSpec(
        "t_month",
        "calendar",
        "month of target day",
        "day t",
        _CAL_AV,
        "never missing",
        "Yearly seasonality.",
        True,
    ),
    FeatureSpec(
        "t_dom",
        "calendar",
        "day of month of target day",
        "day t",
        _CAL_AV,
        "never missing",
        "Pay-cycle effects.",
    ),
    FeatureSpec(
        "t_event_type_1",
        "events",
        "event_type_1 code of target day (vocabulary fit on training calendar)",
        "day t",
        _CAL_AV,
        "no event -> code of 'none'; unseen -> -1",
        "Holiday/sporting/cultural/religious type.",
        True,
    ),
    FeatureSpec(
        "t_event_name_1",
        "events",
        "event_name_1 code of target day",
        "day t",
        _CAL_AV,
        "no event -> code of 'none'; unseen -> -1",
        "Named event effects.",
        True,
    ),
    FeatureSpec(
        "price_origin",
        "price",
        "median transaction price of the latest COMPLETED 7-day block (ended on or before o), carried forward",
        "completed blocks <= o",
        "fv2 (price-availability fix): a block price is known only after its block has ended; the origin's own block is excluded unless o is its last day. Transaction-derived, not a list price.",
        "NaN if never listed by o (never filled with 0, never back-filled)",
        "Current price level.",
    ),
    FeatureSpec(
        "price_age_weeks",
        "price",
        "completed blocks since the carried-forward price was last recorded (0 = latest completed block)",
        "completed blocks <= o",
        "as above",
        "NaN if never listed",
        "Price staleness.",
    ),
    FeatureSpec(
        "price_missing",
        "price",
        "1 if no price recorded in any completed block by o else 0",
        "completed blocks <= o",
        "as above",
        "never missing",
        "Availability indicator (not an imputation).",
    ),
    FeatureSpec(
        "price_chg_4w",
        "price",
        "price_origin / price_origin_{4 weeks earlier} - 1",
        "completed blocks <= o",
        "as above",
        "NaN if either price unavailable",
        "Recent promotion / price-change signal.",
    ),
    FeatureSpec(
        "price_rel_max",
        "price",
        "price_origin / max recorded block price among completed blocks",
        "completed blocks <= o",
        "as above",
        "NaN if never listed",
        "Discount depth relative to history.",
    ),
    FeatureSpec(
        "target_price",
        "price_planned",
        "listed price of the target week (SCENARIO: planned prices known)",
        "week(t)",
        "Only in the labelled 'planned' scenario; assumes the retailer's price plan is known "
        "28 days ahead -- NOT proven for this dataset.",
        "NaN if not listed",
        "Planned price.",
    ),
    FeatureSpec(
        "target_price_chg",
        "price_planned",
        "target_price / price_origin - 1 (SCENARIO: planned)",
        "week(t)",
        "as above",
        "NaN if either missing",
        "Planned price change.",
    ),
]

SPEC_BY_NAME = {s.name: s for s in SPECS}
GROUP_ORDER = ["history", "ids", "horizon", "calendar", "events", "price", "price_planned"]


def columns_for(groups: list[str], price_scenario: str = "observed") -> list[str]:
    gs = list(groups)
    if "price" in gs and price_scenario == "planned":
        gs.append("price_planned")
    return [s.name for g in GROUP_ORDER if g in gs for s in SPECS if s.group == g]


def categorical_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if SPEC_BY_NAME[c].categorical]


def feature_version(cols: list[str], price_scenario: str = "observed") -> str:
    code = (
        PRICE_CODE_VERSION
        if any(SPEC_BY_NAME[c].group.startswith("price") for c in cols)
        else FEATURE_CODE_VERSION
    )
    blob = json.dumps(
        {
            "code": code,
            "cols": cols,
            "scenario": price_scenario,
            "specs": [asdict(SPEC_BY_NAME[c]) for c in cols],
        },
        sort_keys=True,
    ).encode()
    return f"fv{code}-{hashlib.sha256(blob).hexdigest()[:8]}"


def group_of(col: str) -> str:
    return SPEC_BY_NAME[col].group
