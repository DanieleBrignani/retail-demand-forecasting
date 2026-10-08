import json
import os
import stat

import pandas as pd
import pytest

from retailfc.ingestion import raw, validate
from retailfc.panel import load_panel_from_raw
from tests.synth import make_dataset, write_csvs


def test_ingest_records_manifest_and_makes_files_read_only(tmp_env):
    src = write_csvs(tmp_env / "src", 120, n_items=12)
    mf = raw.ingest_raw(src)
    assert mf["status"] == "ingested" and mf["dataset_version"].startswith("retail-")
    assert set(mf["files"]) == {"calendar.csv", "sell_prices.csv", "sales_daily.csv"}
    assert all(len(v["sha256"]) == 64 for v in mf["files"].values())
    assert mf["observed_sales"]["n_days"] == 120 and mf["observed_sales"]["last_d"] == "d_120"
    assert mf["schema_version"] == validate.SCHEMA_VERSION and mf["ingested_at"]
    assert mf["calendar_range"]["n_days"] > mf["observed_sales"]["n_days"]  # future calendar days exist
    f = raw.raw_root() / mf["dataset_version"] / "calendar.csv"
    assert not os.access(f, os.W_OK) or not (f.stat().st_mode & stat.S_IWRITE)


def test_reingest_same_bytes_is_verified_noop_and_changed_bytes_get_new_version(tmp_env):
    src = write_csvs(tmp_env / "src", 120, n_items=12)
    v1 = raw.ingest_raw(src)["dataset_version"]
    again = raw.ingest_raw(src)
    assert again["status"] == "already_ingested" and again["dataset_version"] == v1
    # change content -> a new version; old one untouched
    p = src / "sales_daily.csv"
    df = pd.read_csv(p)
    df.loc[0, "d_5"] += 1
    df.to_csv(p, index=False)
    v2 = raw.ingest_raw(src)["dataset_version"]
    assert v2 != v1 and {v1, v2} <= set(raw.list_versions())
    raw.verify_version(v1)


def test_tampered_stored_version_is_detected_not_overwritten(tmp_env):
    src = write_csvs(tmp_env / "src", 120, n_items=12)
    v = raw.ingest_raw(src)["dataset_version"]
    f = raw.raw_root() / v / "sell_prices.csv"
    os.chmod(f, stat.S_IWRITE | stat.S_IREAD)
    f.write_text(f.read_text() + "x")
    with pytest.raises(raw.RawIngestError, match="differ"):
        raw.ingest_raw(src)


def test_missing_files_give_actionable_error(tmp_env):
    (tmp_env / "empty").mkdir()
    with pytest.raises(raw.RawIngestError, match="adapt-online-retail"):
        raw.ingest_raw(tmp_env / "empty")


def test_panel_roundtrip_matches_source(tmp_env):
    src = write_csvs(tmp_env / "src", 120, n_items=12)
    v = raw.ingest_raw(src)["dataset_version"]
    p = load_panel_from_raw(v)
    sales = pd.read_csv(src / "sales_daily.csv")
    assert p.sales.shape == (len(sales), 120)
    assert (p.sales[0] == sales.iloc[0][[f"d_{i}" for i in range(1, 121)]].to_numpy()).all()
    assert p.dataset_version == v


@pytest.fixture(scope="module")
def good():
    return make_dataset(120, n_items=12)


def test_valid_data_passes(good):
    cal, sales, prices = good
    assert validate.validate_all(cal, sales, prices).ok


def _fails(rep, name):
    return any(c.name == name and not c.passed for c in rep.checks)


def test_duplicate_rows_detected(good):
    cal, sales, prices = good
    assert _fails(
        validate.validate_sales(pd.concat([sales, sales.iloc[[0]]]), cal), "sales.unique_item_store"
    )
    assert _fails(
        validate.validate_prices(pd.concat([prices, prices.iloc[[0]]]), cal), "prices.unique_store_item_week"
    )
    assert _fails(validate.validate_calendar(pd.concat([cal, cal.iloc[[0]]])), "calendar.unique_date")


def test_negative_and_missing_sales_detected(good):
    cal, sales, _ = good
    s = sales.copy()
    s.loc[0, "d_3"] = -1
    assert _fails(validate.validate_sales(s, cal), "sales.no_negative")
    s = sales.copy()
    s["d_4"] = s["d_4"].astype(float)
    s.loc[0, "d_4"] = None
    assert _fails(validate.validate_sales(s, cal), "sales.no_missing_observations")


def test_missing_calendar_dates_and_day_columns_detected(good):
    cal, sales, _ = good
    assert _fails(validate.validate_calendar(cal.drop(index=10)), "calendar.no_missing_dates")
    assert _fails(validate.validate_sales(sales.drop(columns=["d_7"]), cal), "sales.day_columns_contiguous")


def test_invalid_prices_and_orphans_detected(good):
    cal, sales, prices = good
    p = prices.copy()
    p.loc[0, "sell_price"] = 0
    assert _fails(validate.validate_prices(p, cal), "prices.price_positive")
    p = prices.copy()
    p.loc[0, "item_id"] = "UNKNOWN_1"
    assert _fails(
        validate.validate_prices(p, cal, sales[["store_id", "item_id"]]), "prices.referential_integrity"
    )
    p = prices.copy()
    p.loc[0, "week_id"] = 99999
    assert _fails(validate.validate_prices(p, cal), "prices.week_in_calendar")


def test_source_schema_change_detected_and_invalid_identifier(good):
    cal, sales, prices = good
    assert _fails(validate.validate_calendar(cal.drop(columns=["week_id"])), "calendar.required_columns")
    assert _fails(
        validate.validate_prices(prices.rename(columns={"sell_price": "price"}), cal),
        "prices.required_columns",
    )
    s = sales.copy()
    s.loc[0, "state_id"] = "TX"
    assert _fails(validate.validate_sales(s, cal), "sales.state_prefix_of_store")


def test_ingest_rejects_invalid_source_and_stores_nothing(tmp_env):
    src = write_csvs(tmp_env / "src", 120, n_items=12)
    p = src / "sales_daily.csv"
    df = pd.read_csv(p)
    df.loc[0, "d_2"] = -5
    df.to_csv(p, index=False)
    with pytest.raises(raw.RawIngestError, match="validation failed"):
        raw.ingest_raw(src)
    assert raw.list_versions() == []


def test_manifest_json_is_valid(tmp_env):
    src = write_csvs(tmp_env / "src", 120, n_items=12)
    v = raw.ingest_raw(src)["dataset_version"]
    json.loads((raw.raw_root() / v / "manifest.json").read_text())


def test_publish_retries_transient_lock_then_succeeds(tmp_env, monkeypatch):
    """A sync client may briefly lock the fresh folder on Windows; the rename is retried, never copied."""
    src = write_csvs(tmp_env / "src", 120, n_items=12)
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "Access denied")
        return real(a, b)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(raw.time, "sleep", lambda s: None)
    mf = raw.ingest_raw(src)
    assert calls["n"] == 3 and mf["status"] == "ingested"
    raw.verify_version(mf["dataset_version"])


def test_persistent_lock_fails_clearly_and_leaves_no_partial_version(tmp_env, monkeypatch):
    src = write_csvs(tmp_env / "src", 120, n_items=12)

    def locked(a, b):
        raise PermissionError(5, "Access denied")

    monkeypatch.setattr(os, "replace", locked)
    monkeypatch.setattr(raw.time, "sleep", lambda s: None)
    with pytest.raises(raw.RawIngestError, match="stayed locked"):
        raw.ingest_raw(src)
    assert raw.list_versions() == []
    assert not any(
        p.name.startswith(".tmp-") for p in raw.raw_root().iterdir()
    )  # read-only temp files were cleaned up
