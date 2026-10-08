"""Adapter tests on a tiny HAND-MADE transaction table (not real data, not a benchmark)."""

import numpy as np
import pandas as pd
import pytest

from retailfc.adapters import online_retail as OR
from retailfc.features.build import build_examples, fit_encoders
from retailfc.ingestion import raw, validate
from retailfc.panel import load_panel_from_raw

COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def tx_table():
    rows = [
        # invoice, code, qty, timestamp, price, country
        ("500001", "85123A", 6, "2010-12-20 09:00", 2.00, "United Kingdom"),
        (
            "500002",
            "85123A",
            4,
            "2010-12-20 15:00",
            2.50,
            "United Kingdom",
        ),  # same day: summed, price median 2.25
        (
            "500003",
            "85123A",
            3,
            "2010-12-22 10:00",
            2.00,
            "France",
        ),  # other channel, 2 days later (gap day = closed)
        ("500004", "22423", 10, "2010-12-25 11:00", 12.0, "United Kingdom"),  # Christmas day (holiday event)
        ("C500005", "22423", -5, "2010-12-26 11:00", 12.0, "United Kingdom"),  # cancellation: excluded
        ("500006", "22423", -2, "2010-12-26 12:00", 12.0, "United Kingdom"),  # negative quantity: excluded
        ("500007", "22423", 5, "2010-12-26 13:00", 0.0, "United Kingdom"),  # zero price: excluded
        ("500008", "POST", 1, "2010-12-26 14:00", 18.0, "United Kingdom"),  # non-product code: excluded
        ("500009", "DCGS0003", 1, "2010-12-27 14:00", 5.0, "United Kingdom"),  # non-product code: excluded
        ("500010", "21730", 2, "2010-12-30 10:00", 4.0, "Germany"),
    ]
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Country"])
    df["Description"], df["Customer ID"] = "x", 1.0
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df[COLS]


@pytest.fixture()
def built(tmp_path):
    rep = OR.adapt(None, tmp_path / "out", n_future=56, transactions=tx_table())
    return tmp_path / "out", rep


def sales_row(out, item, store):
    s = pd.read_csv(out / "sales_daily.csv")
    r = s[(s.item_id == f"RETAIL_1_{item}") & (s.store_id == store)].iloc[0]
    return r[[c for c in s.columns if c.startswith("d_")]].astype(int).to_numpy()


def test_cleaning_rules_and_their_counts_are_recorded(built):
    _, rep = built
    c = rep["cleaning"]
    assert c["rows_in"] == 10 and c["rows_kept"] == 5
    assert c["excluded_cancellation_invoices"] == 1 and c["excluded_nonpositive_quantity"] == 2
    assert c["excluded_nonpositive_or_missing_price"] == 1 and c["excluded_non_product_codes"] == 2
    assert "POST" in c["non_product_codes_seen"]


def test_sales_are_summed_per_day_and_cancellations_are_not_netted(built):
    out, _ = built
    uk = sales_row(out, "85123A", "UK_1")
    assert uk[0] == 10 and uk.sum() == 10  # 2010-12-20 is d_1: 6 + 4
    assert sales_row(out, "22423", "UK_1")[5] == 10  # 25 Dec = d_6; the cancellation did not reduce it
    assert sales_row(out, "85123A", "INT_1")[2] == 3  # France on d_3 -> international channel


def test_panel_is_complete_item_by_channel_with_structural_zeros(built):
    out, rep = built
    s = pd.read_csv(out / "sales_daily.csv")
    assert len(s) == 3 * 2 == rep["panel"]["n_series"]  # 3 products x 2 channels, zero-filled
    assert (
        sales_row(out, "21730", "UK_1").sum() == 0
    )  # never sold in the UK channel: kept as an all-zero series
    assert sales_row(out, "85123A", "UK_1")[1] == 0  # closed day is a zero observation, not missing


def test_prices_are_weekly_medians_and_absent_weeks_have_no_row(built):
    out, _ = built
    p = pd.read_csv(out / "sell_prices.csv")
    r = p[(p.item_id == "RETAIL_1_85123A") & (p.store_id == "UK_1")]
    assert list(r.week_id) == [1] and r.sell_price.iloc[0] == pytest.approx(2.25)
    assert (p.sell_price > 0).all() and not p.duplicated(["store_id", "item_id", "week_id"]).any()
    assert not ((p.item_id == "RETAIL_1_21730") & (p.store_id == "UK_1")).any()  # no sales -> no price row


def test_calendar_has_future_days_and_holiday_events(built):
    out, _ = built
    cal = pd.read_csv(out / "calendar.csv")
    n_obs = 11  # 2010-12-20 .. 2010-12-30
    assert len(cal) == n_obs + 56 and cal.d.iloc[0] == "d_1" and cal.date.iloc[0] == "2010-12-20"
    xmas = cal[cal.date == "2010-12-25"].iloc[0]
    assert (
        isinstance(xmas.event_name_1, str)
        and "Christmas" in xmas.event_name_1
        and xmas.event_type_1 == "National"
    )


def test_output_passes_validation_and_ingests_with_its_own_label(built, tmp_env):
    out, _ = built
    cal, sales, prices = (
        pd.read_csv(out / f) for f in ("calendar.csv", "sales_daily.csv", "sell_prices.csv")
    )
    assert validate.validate_all(cal, sales, prices).ok
    mf = raw.ingest_raw(out, source_note="UCI Online Retail II (CC BY 4.0), adapted", label="retail")
    assert mf["dataset_version"].startswith("retail-") and "Online Retail II" in mf["source"]
    assert mf["observed_sales"]["n_days"] == 11 and mf["observed_sales"]["first_date"] == "2010-12-20"
    raw.verify_version(mf["dataset_version"])


def test_panel_and_features_work_with_adapter_states(built, tmp_env):
    out, _ = built
    v = raw.ingest_raw(out, label="retail")["dataset_version"]
    panel = load_panel_from_raw(v)
    assert panel.state_names == ["INT", "UK"]
    enc = fit_encoders(panel, panel.n_days)
    df = build_examples(
        panel,
        8,
        np.arange(panel.n_series),
        [1, 2],
        ["history", "ids", "horizon", "calendar", "events"],
        enc,
    )
    assert len(df) == panel.n_series * 2 and "t_snap" not in df.columns
    assert df["t_event_name_1"].nunique() >= 1


def test_codes_with_stray_whitespace_are_kept_as_products():
    df = tx_table()
    df.loc[0, "StockCode"] = "85123A "  # trailing blank exists in the real file (e.g. '47503J ')
    cleaned, rep = OR.clean(df)
    assert "85123A" in set(cleaned.StockCode) and rep["rows_kept"] == 5
    assert "85123A " not in rep["non_product_codes_seen"]
