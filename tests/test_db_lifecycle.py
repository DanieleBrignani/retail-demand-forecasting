"""Integration tests against a REAL PostgreSQL (RETAILFC_TEST_DSN). Skipped when no database is configured."""

import numpy as np
import pandas as pd
import psycopg
import pytest

from retailfc.forecasting import store
from retailfc.warehouse import db
from tests.conftest import TEST_DSN, needs_db

pytestmark = [needs_db, pytest.mark.integration]


@pytest.fixture()
def conn():
    with psycopg.connect(TEST_DSN, autocommit=True) as c:
        for s in ("app", "raw", "staging"):
            c.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        c.execute("DROP TABLE IF EXISTS public.schema_migrations")
    db.migrate(TEST_DSN)
    with psycopg.connect(TEST_DSN) as c:
        yield c


def _rec(n=6, value=1.5):
    dates = pd.date_range("2012-01-02", periods=4).strftime("%Y-%m-%d")
    rows = [
        (f"I{i}", "UK_1", d, h + 1, value + i, np.nan, np.nan, "")
        for i in range(n // 2)
        for h, d in enumerate(dates)
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "item_id",
            "store_id",
            "target_date",
            "horizon",
            "forecast_value",
            "lower_bound",
            "upper_bound",
            "fallback_used",
        ],
    )


KW = dict(
    model_name="retailfc_lgbm",
    model_version="3",
    dataset_version="dv1",
    feature_version="fv1",
    origin_date="2012-01-01",
    origin_day=100,
    mode="historical_replay",
)


def test_migrations_idempotent(conn):
    assert db.migrate(TEST_DSN) == []  # fixture already applied everything; rerun applies nothing


def test_write_batch_is_idempotent_and_conflict_safe(conn):
    rec = _rec()
    r1 = store.write_batch(conn, rec, **KW)
    conn.commit()
    assert r1["status"] == "written" and r1["n_rows"] == len(rec)
    r2 = store.write_batch(conn, rec, **KW)  # rerun, same content
    assert r2["status"] == "noop_identical"
    n = conn.execute("SELECT count(*) FROM app.forecasts").fetchone()[0]
    assert n == len(rec)
    changed = rec.copy()
    changed["forecast_value"] += 1
    with pytest.raises(store.ForecastConflict):
        store.write_batch(conn, changed, **KW)
    conn.rollback()
    assert conn.execute("SELECT count(*) FROM app.forecasts").fetchone()[0] == len(rec)
    r3 = store.write_batch(conn, changed, **KW, run_label="rerun-2")  # explicit new label coexists
    assert r3["status"] == "written"
    assert conn.execute("SELECT count(*) FROM app.forecast_batches").fetchone()[0] == 2
    row = conn.execute(
        "SELECT model_version, dataset_version, feature_version FROM app.forecasts LIMIT 1"
    ).fetchone()
    assert row == ("3", "dv1", "fv1")


def test_failed_write_rolls_back_everything(conn):
    rec = _rec()
    bad = rec.copy()
    bad.loc[0, "horizon"] = 99  # violates CHECK mid-COPY
    with pytest.raises(psycopg.Error):
        store.write_batch(conn, bad, **KW)
    conn.rollback()
    assert conn.execute("SELECT count(*) FROM app.forecast_batches").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM app.forecasts").fetchone()[0] == 0


def test_negative_forecasts_rejected(conn):
    rec = _rec()
    rec.loc[1, "forecast_value"] = -0.1
    with pytest.raises(ValueError):
        store.write_batch(conn, rec, **KW)


def test_actuals_idempotent_conflict_detected_and_not_overwritten(conn):
    a = pd.DataFrame(
        {
            "item_id": ["I0", "I0"],
            "store_id": ["UK_1"] * 2,
            "date": ["2012-01-02", "2012-01-03"],
            "sales": [3, 0],
        }
    )
    assert store.ingest_actuals(conn, a, "historical_replay")["inserted"] == 2
    again = store.ingest_actuals(conn, a, "historical_replay")
    assert again["inserted"] == 0 and again["already_present"] == 2 and again["conflicts"] == 0
    b = a.copy()
    b.loc[0, "sales"] = 9
    r = store.ingest_actuals(conn, b, "historical_replay")
    assert r["conflicts"] == 1
    assert conn.execute("SELECT sales FROM app.actuals WHERE date='2012-01-02'").fetchone()[0] == 3
    with pytest.raises(ValueError):
        store.ingest_actuals(conn, a.assign(sales=[-1, 0]), "x")


def test_reconcile_scores_only_matched_actuals_and_keeps_forecasts(conn):
    rec = _rec(n=2, value=2.0)  # item I0, 4 horizons: 2012-01-02..05
    store.write_batch(conn, rec, **KW)
    a = pd.DataFrame(
        {
            "item_id": ["I0", "I0"],
            "store_id": ["UK_1"] * 2,
            "date": ["2012-01-02", "2012-01-03"],
            "sales": [3, 0],
        }
    )
    store.ingest_actuals(conn, a, "historical_replay")
    r = store.reconcile(conn)
    assert r["new_errors"] == 2 and r["unscored_forecasts"] == 2  # unmatched forecasts are not scored
    err = conn.execute(
        "SELECT horizon, error, abs_error FROM app.forecast_errors ORDER BY horizon"
    ).fetchall()
    assert err == [(1, -1.0, 1.0), (2, 2.0, 2.0)]  # error = forecast - actual
    assert store.reconcile(conn)["new_errors"] == 0  # idempotent
    assert conn.execute("SELECT count(*) FROM app.forecasts").fetchone()[0] == 4
    agg = conn.execute(
        "SELECT horizon, n, sum_abs_error FROM app.monitoring_aggregates ORDER BY horizon"
    ).fetchall()
    assert agg == [(1, 1, 1.0), (2, 1, 2.0)]
    # a new actual arrives later -> only the new pair is scored
    store.ingest_actuals(
        conn,
        pd.DataFrame({"item_id": ["I0"], "store_id": ["UK_1"], "date": ["2012-01-04"], "sales": [2]}),
        "x",
    )
    assert store.reconcile(conn)["new_errors"] == 1


def test_one_current_approved_model_enforced(conn):
    conn.execute(
        "INSERT INTO app.approved_models (model_name, model_version, dataset_version, approved_by) VALUES ('m','1','d','rule')"
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO app.approved_models (model_name, model_version, dataset_version, approved_by) VALUES ('m','2','d','rule')"
        )
