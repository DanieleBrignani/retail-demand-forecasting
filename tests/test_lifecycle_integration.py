"""Full lifecycle on SYNTHETIC data with real PostgreSQL + real MLflow (sqlite backend): mechanics, not benchmarks."""

from datetime import date

import numpy as np
import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from retailfc import config as C
from retailfc.api.app import create_app
from retailfc.forecasting import store
from retailfc.forecasting.generate import generate_forecast_batch
from retailfc.monitoring import core as MON
from retailfc.orchestration import retraining as RT
from retailfc.registry import mlflow_utils as ML
from retailfc.registry.pipeline import train_evaluate_promote
from retailfc.warehouse import db
from tests.conftest import TEST_DSN, needs_db
from tests.synth import make_panel

pytestmark = [needs_db, pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import os

    d = tmp_path_factory.mktemp("life")
    os.environ["RETAILFC_ARTIFACTS_DIR"] = str(d / "art")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(d / 'mlflow.db').as_posix()}"
    os.environ["RETAILFC_DSN"] = TEST_DSN
    with psycopg.connect(TEST_DSN, autocommit=True) as c:
        for s in ("app", "raw", "staging"):
            c.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        c.execute("DROP TABLE IF EXISTS public.schema_migrations")
    db.migrate(TEST_DSN)
    cfg = C.override(
        C.load_config(),
        **{
            "lightgbm.num_boost_round": 20,
            "lightgbm.early_stopping_rounds": 5,
            "lightgbm.params.min_data_in_leaf": 20,
            "lightgbm.params.num_threads": 2,
            "promotion.max_vs_snaive_ratio": 2.0,
            "promotion.max_abs_bias_pct": 80.0,
            "promotion.min_interval_coverage": 0.3,
        },
    )
    return cfg, make_panel(300)


@pytest.fixture(scope="module")
def trained(env):
    cfg, panel = env
    # trained through day 272 so a historical replay from origin 272 is not leaky; days 273..300 stay out-of-sample
    r = train_evaluate_promote(panel, cfg, feature_key="D", params=None, train_origin=272)
    return r


def test_first_challenger_is_registered_and_promoted_with_audit_evidence(env, trained):
    r = trained
    assert r["status"] == "promoted", r["decision"]
    assert r["model_version"] == "1"
    assert ML.champion_version(env[0]["serving"]["model_name"]) == "1"
    with psycopg.connect(TEST_DSN) as c:
        row = c.execute("SELECT model_version, evidence FROM app.approved_models WHERE is_current").fetchone()
        assert row[0] == "1" and row[1]["decision"]["promote"] is True
    ev = ML.read_evidence("retailfc_lgbm", "1")
    assert ev["challenger"]["wape"] > 0 and ev["challenger"]["coverage"] is not None


def test_model_must_be_selected_by_explicit_version(env, trained):
    with pytest.raises(ML.ModelUnavailable):
        ML.load_version("retailfc_lgbm", "latest")
    with pytest.raises(ML.ModelUnavailable):
        ML.load_version("retailfc_lgbm", "99")


def test_registry_round_trip_matches_in_memory_model(env, trained):
    cfg, panel = env
    point, bundle, info = ML.load_version("retailfc_lgbm", "1")
    assert (
        info["tags"]["feature_version"] == point.feature_version
        and info["tags"]["dataset_version"] == "synthetic-test"
    )
    assert bundle is not None and bundle.offsets  # calibration offsets persisted
    f = point.predict(panel, 272, np.arange(10), [1, 28])
    assert np.isfinite(f.forecast).all() and (f.forecast >= 0).all()


def test_forecast_batch_persisted_with_versions_idempotent_and_api_serves_it(env, trained):
    cfg, panel = env
    with psycopg.connect(TEST_DSN) as conn:
        r1 = generate_forecast_batch(conn, panel, "retailfc_lgbm", "1", origin=272, mode="historical_replay")
        assert r1["status"] == "written" and r1["n_rows"] == panel.n_series * 28
        r2 = generate_forecast_batch(conn, panel, "retailfc_lgbm", "1", origin=272, mode="historical_replay")
        assert r2["status"] == "noop_identical"
        row = conn.execute(
            "SELECT model_version, dataset_version, feature_version, forecast_origin, min(target_date), "
            "max(target_date), count(*) FROM app.forecasts GROUP BY 1,2,3,4"
        ).fetchall()
        assert len(row) == 1 and row[0][0] == "1" and row[0][6] == panel.n_series * 28
        assert str(row[0][3]) == str(pd.Timestamp(panel.date_of(272)).date())
        assert row[0][4] > row[0][3]  # every target after origin
        n_iv = conn.execute(
            "SELECT count(*) FROM app.forecasts WHERE lower_bound IS NOT NULL AND lower_bound <= forecast_value "
            "AND forecast_value <= upper_bound"
        ).fetchone()[0]
        assert n_iv > 0.9 * panel.n_series * 28
        assert conn.execute("SELECT count(*) FROM app.forecasts WHERE forecast_value < 0").fetchone()[0] == 0
    it, st = panel.ids.item_id.iloc[0], panel.ids.store_id.iloc[0]
    with TestClient(
        create_app(TEST_DSN, cfg, api_keys=["lifecycle-test-key-0123"]),
        headers={"x-api-key": "lifecycle-test-key-0123"},
    ) as c:
        j = c.post("/forecast", json={"item_id": it, "store_id": st, "horizon": 14}).json()
        assert j["model_version"] == "1" and len(j["forecasts"]) == 14
        assert j["forecast_origin"] == str(pd.Timestamp(panel.date_of(272)).date())


def test_all_zero_series_is_flagged_fallback_not_silent(env, trained):
    cfg, panel = env
    with psycopg.connect(TEST_DSN) as conn:
        zi = panel.ids.iloc[5]  # all-zero series in the fixture
        r = conn.execute(
            "SELECT fallback_used, max(forecast_value) FROM app.forecasts WHERE item_id=%s AND store_id=%s GROUP BY 1",
            (zi.item_id, zi.store_id),
        ).fetchall()
        assert r == [("all_zero_history", 0.0)]


def test_actuals_replay_reconcile_monitor_and_retrain_decision(env, trained):
    cfg, panel = env
    with psycopg.connect(TEST_DSN) as conn:
        # historical replay: release days 273..300 (the batch's target window, origin 272)
        ing = store.ingest_actuals(conn, store.replay_actuals_frame(panel, 273, 300), "historical_replay")
        assert ing["inserted"] == panel.n_series * 28
        out = MON.refresh(conn, date(2012, 1, 1), panel, 272)
        assert (
            out["reconcile"]["new_errors"] == panel.n_series * 28
            and out["reconcile"]["unscored_forecasts"] == 0
        )
        perf = out["performance"][0]
        assert perf["n"] == panel.n_series * 28 and perf["wape"] > 0
        # cross-check the DB aggregate against a direct recomputation from stored forecasts and actuals
        df = pd.read_sql_query(
            "SELECT f.forecast_value f, a.sales a FROM app.forecasts f JOIN app.actuals a ON "
            "a.item_id=f.item_id AND a.store_id=f.store_id AND a.date=f.target_date",
            conn,
        )
        assert perf["wape"] == pytest.approx((df.f - df.a).abs().sum() / df.a.sum(), rel=1e-9)
        assert perf["bias_pct"] == pytest.approx((df.f - df.a).sum() / df.a.sum(), rel=1e-9)
        # errors are keyed by vintage and horizon; forecasts untouched
        assert conn.execute("SELECT count(DISTINCT horizon) FROM app.forecast_errors").fetchone()[0] == 28
        assert conn.execute("SELECT count(*) FROM app.forecasts").fetchone()[0] == panel.n_series * 28
        assert out["actuals"]["missing_actuals_fraction"] == 0.0
        # decision flow: fresh model + cooldown => no retrain, audited
        inputs = RT.gather_inputs(conn, cfg, date(2012, 1, 1))
        dec = RT.decide(inputs, cfg["retraining"])
        assert dec.decision == "no_retrain"
        did = RT.record_decision(conn, dec, inputs, cfg["retraining"])
        assert (
            conn.execute(
                "SELECT decision FROM app.retraining_decisions WHERE decision_id=%s", (did,)
            ).fetchone()[0]
            == "no_retrain"
        )


def test_replay_with_model_trained_after_origin_is_refused(env, trained):
    cfg, panel = env
    with psycopg.connect(TEST_DSN) as conn:
        with pytest.raises(Exception, match="after the forecast origin"):
            generate_forecast_batch(conn, panel, "retailfc_lgbm", "1", origin=244, mode="historical_replay")
        conn.rollback()


def test_identical_challenger_is_registered_but_not_promoted_over_incumbent(env, trained):
    cfg, panel = env
    r = train_evaluate_promote(panel, cfg, feature_key="D", params=None, train_origin=panel.n_days)
    # same recipe, same comparison window (272, 300]: 0% improvement < 1% required => registered, NOT promoted
    assert r["status"] == "registered", r.get("decision")
    assert "improves_on_incumbent" in r["decision"]["failed"]
    assert ML.champion_version("retailfc_lgbm") == "1"
    assert r["model_version"] == "2"


def test_rollback_is_audited_lock_protected_and_does_not_touch_forecasts(env, trained):
    from retailfc.registry.pipeline import rollback

    with psycopg.connect(TEST_DSN) as conn:
        n_before = conn.execute("SELECT count(*) FROM app.forecasts").fetchone()[0]
        with pytest.raises(ValueError):
            rollback(conn, "retailfc_lgbm", "2", reason=" ")
        r = rollback(conn, "retailfc_lgbm", "2", reason="drill: challenger v2 re-approved")
        assert (
            r["status"] == "rolled_back" and r["from"] == "1" and ML.champion_version("retailfc_lgbm") == "2"
        )
        assert (
            conn.execute("SELECT model_version FROM app.approved_models WHERE is_current").fetchone()[0]
            == "2"
        )
        assert rollback(conn, "retailfc_lgbm", "2", reason="again")["status"] == "noop"
        back = rollback(conn, "retailfc_lgbm", "1", reason="drill: back to v1")
        assert back["to"] == "1" and ML.champion_version("retailfc_lgbm") == "1"
        rows = conn.execute(
            "SELECT model_version, approved_by FROM app.approved_models ORDER BY id"
        ).fetchall()
        assert [r[0] for r in rows][-3:] == ["1", "2", "1"] and rows[-1][1] == "manual:rollback"
        assert conn.execute("SELECT count(*) FROM app.forecasts").fetchone()[0] == n_before
        from mlflow.exceptions import MlflowException

        with pytest.raises(MlflowException):
            rollback(conn, "retailfc_lgbm", "99", reason="missing version")
        conn.rollback()


def test_concurrent_training_is_refused(env):
    cfg, panel = env
    with psycopg.connect(TEST_DSN) as hold:
        with db.advisory_lock(hold, db.LOCK_TRAINING) as got:
            assert got
            r = train_evaluate_promote(panel, cfg, feature_key="A", params=None, with_intervals=False)
            assert r["status"] == "skipped_concurrent"
