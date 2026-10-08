"""Information boundary of prices, tested from ORIGINAL TRANSACTION ROWS (adapter -> ingest -> panel -> features).

The fixtures are tiny hand-made transaction tables (unit tests, not a benchmark). Each test builds the source, runs the real
adapter, builds features at an EARLIER origin, changes ONLY information that lies after that origin, rebuilds, and compares the
earlier features by business key (item_id, store_id) -- never by row position, because the series universe may change.

Defect reproduced (regression): the adapter's price of a 7-day block is the median of ALL its lines; the panel used to expose the
block of the origin even when the origin was inside it, so a future day-4 price changed the feature at origin day 1.
Policy now (docs/data_contract.md, "price availability"): a block price is visible only once the block has ended; features use the
latest COMPLETED block price (carried forward, never backfilled), with its age; missing stays missing.
"""

import numpy as np
import pandas as pd
import pytest

from retailfc.adapters import online_retail as OR
from retailfc.features.build import build_examples, fit_encoders
from retailfc.ingestion import raw
from retailfc.panel import load_panel_from_raw

COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
FIRST = pd.Timestamp("2010-12-01")  # block 1 = days 1-7, block 2 = days 8-14, ...
GROUPS = ["history", "ids", "horizon", "calendar", "events", "price"]
UK = "United Kingdom"


def day(n: int) -> str:
    return str((FIRST + pd.Timedelta(days=n - 1)).date()) + " 10:00"


def make_tx(extra: list[tuple], n_days: int = 28) -> pd.DataFrame:
    """A filler product sells every day (so the observation window is fixed) plus the rows under test.
    extra rows: (code, qty, day_number, price, country)."""
    rows = [("F%05d" % d, "22423", 1, day(d), 5.0, UK) for d in range(1, n_days + 1)]
    for i, (code, qty, d, price, country) in enumerate(extra):
        rows.append((f"X{i:05d}", code, qty, day(d), price, country))
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Country"])
    df["Description"], df["Customer ID"] = "x", 1.0
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df[COLS]


def build(tx: pd.DataFrame, tmp_path, name: str):
    out = tmp_path / name
    OR.adapt(None, out, n_future=56, transactions=tx)
    v = raw.ingest_raw(out, label="retail")["dataset_version"]
    return load_panel_from_raw(v)


def features(panel, origin: int, horizons=(1, 2)) -> pd.DataFrame:
    """Features at `origin` for every series, indexed by business key."""
    enc = fit_encoders(panel, origin)
    df = build_examples(
        panel, origin, np.arange(panel.n_series), list(horizons), GROUPS, enc, with_target=False
    )
    ids = panel.ids.iloc[df["series_idx"].to_numpy()].reset_index(drop=True)
    # integer vocabulary codes depend on the universe the encoder was fitted on; they are compared separately (universe tests)
    df = df.reset_index(drop=True).drop(columns=["item_id", "dept_id", "cat_id", "store_id", "state_id"])
    df.insert(0, "item_id", ids["item_id"])
    df.insert(1, "store_id", ids["store_id"])
    return df.drop(columns=["series_idx"]).set_index(["item_id", "store_id", "h"]).sort_index()


def row(df: pd.DataFrame, item: str, store: str = "UK_1", h: int = 1) -> pd.Series:
    return df.loc[(f"RETAIL_1_{item}", store, h)]


def assert_same_on_common_keys(a: pd.DataFrame, b: pd.DataFrame, ignore=()):
    keys = a.index.intersection(b.index)
    assert len(keys) > 0
    cols = [c for c in a.columns if c not in ignore]
    pd.testing.assert_frame_equal(a.loc[keys, cols], b.loc[keys, cols], check_exact=False)


# ----------------------------------------------------------------------------- the reproduced defect
@pytest.mark.parametrize("origin", [1, 3])
def test_future_price_inside_the_same_block_does_not_change_an_earlier_origin_feature(tmp_env, origin):
    base = [("85123A", 2, 1, 10.0, UK), ("85123A", 2, 4, 10.0, UK)]
    changed = [
        ("85123A", 2, 1, 10.0, UK),
        ("85123A", 2, 4, 100.0, UK),
    ]  # only the FUTURE (day 4) price differs
    p0, p1 = build(make_tx(base), tmp_env, "a"), build(make_tx(changed), tmp_env, "b")
    f0, f1 = features(p0, origin), features(p1, origin)
    a, b = row(f0, "85123A"), row(f1, "85123A")
    for c in ("price_origin", "price_age_weeks", "price_missing", "price_chg_4w", "price_rel_max"):
        assert (a[c] == b[c]) or (np.isnan(a[c]) and np.isnan(b[c])), (c, a[c], b[c])
    assert_same_on_common_keys(f0, f1)


def test_the_block_price_becomes_visible_only_after_the_block_has_ended(tmp_env):
    tx = make_tx([("85123A", 2, 1, 10.0, UK), ("85123A", 2, 4, 30.0, UK)])  # block-1 median = 20
    p = build(tx, tmp_env, "a")
    inside, at_end, next_day = (
        row(features(p, 3), "85123A"),
        row(features(p, 7), "85123A"),
        row(features(p, 8), "85123A"),
    )
    assert inside["price_missing"] == 1 and np.isnan(
        inside["price_origin"]
    )  # unavailable, not zero, not future
    assert (
        at_end["price_origin"] == pytest.approx(20.0) and at_end["price_age_weeks"] == 0
    )  # origin = last day of block
    assert (
        next_day["price_origin"] == pytest.approx(20.0) and next_day["price_age_weeks"] == 0
    )  # block 2 incomplete


def test_missing_past_prices_followed_by_future_prices_stay_missing(tmp_env):
    tx = make_tx([("85123A", 2, 9, 7.0, UK), ("85123A", 2, 16, 8.0, UK)])  # first sale in block 2
    p = build(tx, tmp_env, "a")
    for origin in (
        3,
        7,
        10,
        14,
    ):  # blocks 1-2 are complete only at 7 / 14; the block-2 price exists from origin 14
        r = row(features(p, origin), "85123A")
        if origin < 14:
            assert r["price_missing"] == 1 and np.isnan(r["price_origin"]), (
                origin
            )  # no backfilling from day 9 / 16
        else:
            assert r["price_origin"] == pytest.approx(7.0)
    assert (
        p.price_weekly[(p.ids.item_id == "RETAIL_1_85123A").to_numpy()] != 0
    ).all()  # no zero used as "unavailable"


def test_future_quantities_and_future_transactions_do_not_change_earlier_features(tmp_env):
    base = [("85123A", 2, 1, 10.0, UK), ("85123A", 3, 5, 10.0, UK)]
    more = [*base, ("85123A", 400, 12, 99.0, UK), ("85123A", 7, 20, 3.0, "France")]  # added after origin 8
    fewer = base[:1]  # removes the day-5 sale (still before origin 8)
    p0, p1 = build(make_tx(base), tmp_env, "a"), build(make_tx(more), tmp_env, "b")
    assert_same_on_common_keys(features(p0, 8), features(p1, 8))
    qty = [("85123A", 2, 1, 10.0, UK), ("85123A", 3000, 5, 10.0, UK)]
    p2 = build(make_tx(qty), tmp_env, "c")
    assert features(p0, 4).equals(features(p2, 4))  # day-5 quantity is in the future of origin 4
    p3 = build(make_tx(fewer), tmp_env, "d")
    f0, f3 = features(p0, 4), features(p3, 4)
    assert_same_on_common_keys(f0, f3)  # removing a later transaction changes nothing before it


def test_a_price_listed_only_in_a_later_block_does_not_leak_through_the_running_max(tmp_env):
    tx = make_tx([("85123A", 2, 1, 10.0, UK), ("85123A", 2, 9, 500.0, UK)])
    p = build(tx, tmp_env, "a")
    r = row(features(p, 7), "85123A")
    assert r["price_origin"] == pytest.approx(10.0) and r["price_rel_max"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- series-universe policy (from original transactions)
def test_a_product_first_sold_after_the_origin_is_not_in_the_universe_vocabulary_or_training(tmp_env):
    from retailfc.features.build import build_training_frame

    old = [("85123A", 2, 1, 10.0, UK)]
    late = [*old, ("21730", 5, 20, 4.0, UK)]  # a product that does not exist before day 20
    p_old, p_late = build(make_tx(old), tmp_env, "a"), build(make_tx(late), tmp_env, "b")
    origin = 10
    assert "RETAIL_1_21730" not in set(p_late.ids.iloc[p_late.eligible_at(origin)]["item_id"])
    assert "RETAIL_1_21730" in set(p_late.ids.iloc[p_late.eligible_at(20)]["item_id"])
    enc_old, enc_late = fit_encoders(p_old, origin), fit_encoders(p_late, origin)
    assert (
        enc_old.ids["item_id"] == enc_late.ids["item_id"]
    )  # vocabularies (and codes of known products) are identical
    assert "RETAIL_1_21730" not in enc_late.ids["item_id"]
    tf = build_training_frame(p_late, [8, 9, 10], np.arange(p_late.n_series), [1], GROUPS, enc_late)
    late_idx = np.flatnonzero((p_late.ids.item_id == "RETAIL_1_21730").to_numpy())
    assert (
        not tf["series_idx"].isin(late_idx).any()
    )  # no training rows for a product that has not launched yet
    # the late product, once forecast anyway, is an UNSEEN identifier (-1): documented cold-start, never a silently known id
    df = build_examples(p_late, origin, late_idx, [1], GROUPS, enc_late, with_target=False)
    assert (df["item_id"] == -1).all()


def test_earlier_features_are_identical_when_a_later_product_is_added_or_removed(tmp_env):
    base = [("85123A", 2, 1, 10.0, UK)]
    late = [*base, ("21730", 5, 20, 4.0, UK), ("21731", 1, 22, 9.0, "France")]
    f0, f1 = (
        features(build(make_tx(base), tmp_env, "a"), 10),
        features(build(make_tx(late), tmp_env, "b"), 10),
    )
    assert set(f0.index) < set(f1.index)  # the universe of the second panel is larger ...
    keys = [k for k in f1.index if k in f0.index]
    # ... but the products known at the origin get identical features, compared by business key (vocabulary codes are checked above)
    pd.testing.assert_frame_equal(f0.loc[keys], f1.loc[keys])


# ----------------------------------------------------------------------------- large quantities must never wrap (int16 regression)
BIG = [("85123A", 74215, 5, 2.0, UK), ("85123A", 80995, 20, 2.0, UK), ("21730", 40000, 6, 1.0, "France")]


def _source_units(tx: pd.DataFrame) -> int:
    from retailfc.adapters import online_retail as OR

    kept, _ = OR.clean(tx)
    return int(kept["Quantity"].sum())


def test_large_daily_quantities_survive_adapter_ingestion_and_panel_without_wrapping(tmp_env):
    """Real data has single series-days of 74,215 and 80,995 units; an int16 read turned them into 8,679 and 15,459 (lost 131,072 units)."""
    tx = make_tx(BIG)
    panel = build(tx, tmp_env, "big")
    assert panel.sales.dtype == np.int32
    assert int(panel.sales.max()) == 80995
    assert int(panel.sales.sum(dtype=np.int64)) == _source_units(
        tx
    )  # every unit of the source reaches the panel
    i = int(np.flatnonzero((panel.ids.item_id == "RETAIL_1_85123A") & (panel.ids.store_id == "UK_1"))[0])
    assert panel.sales[i, 4] == 74215 and panel.sales[i, 19] == 80995


def test_the_sales_reader_refuses_values_it_cannot_hold(tmp_path):
    from retailfc.panel import read_sales_csv

    f = tmp_path / "s.csv"
    pd.DataFrame({"id": ["a"], "d_1": [2**31]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="out of the supported range"):
        read_sales_csv(f, 1)


def test_weekly_aggregation_keeps_large_quantities(tmp_env):
    from retailfc import weekly as W

    tx = make_tx(BIG)
    panel = build(tx, tmp_env, "bigw")
    Y, _ids, _starts, _p = W.to_weekly(panel)
    assert float(Y.max()) >= 74215 and int(round(float(Y.sum()))) <= _source_units(tx)
    assert int(round(float(Y[:, :4].sum()))) == int(panel.sales[:, :28].sum(dtype=np.int64))
