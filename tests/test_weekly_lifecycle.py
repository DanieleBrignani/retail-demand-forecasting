"""Weekly lifecycle on SYNTHETIC data with real PostgreSQL + real MLflow (sqlite backend): mechanics, not forecasting quality.

registration -> rule-based approval -> batch generation (weekly contract) -> idempotency/append-only -> actuals replay ->
weekly reconciliation (twice) -> monitoring -> API retrieval (success, rejection and error paths) -> revocation.
"""

import json
import os

import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from retailfc import config as C
from retailfc import weekly as W
from retailfc import weekly_lifecycle as WL
from retailfc.api.app import create_app
from retailfc.forecasting import store
from retailfc.monitoring import core as MON
from retailfc.registry import mlflow_utils as ML
from retailfc.warehouse import db
from tests.conftest import TEST_DSN, needs_db
from tests.synth import make_panel

pytestmark = [needs_db, pytest.mark.integration]
KEY = "unit-test-key-0123456789"  # throwaway test credential


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("wk")
    os.environ["RETAILFC_ARTIFACTS_DIR"] = str(d / "art")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(d / 'mlflow.db').as_posix()}"
    os.environ["RETAILFC_DSN"] = TEST_DSN
    with psycopg.connect(TEST_DSN, autocommit=True) as c:
        for s in ("app", "raw", "staging"):
            c.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        c.execute("DROP TABLE IF EXISTS public.schema_migrations")
    db.migrate(TEST_DSN)
    panel = make_panel(300)  # 42 complete 7-day blocks + 6 extra days
    exp = d / "exp"
    exp.mkdir()
    cfg = W.load_cfg()
    sel = {
        "protocol": cfg["protocol_version"],
        "selected": "mean4",
        "candidates": {},
        "qualified": ["mean4"],
        "strongest_baseline": "mean4",
    }
    man = {
        "protocol_version": cfg["protocol_version"],
        "config_path": str(W.DEFAULT_CONFIG),
        "config_sha256": W.cfg_hash(),
        "dataset_version": panel.dataset_version,
        "weekly_feature_version": W.WEEKLY_FEATURE_VERSION,
        "calendar": cfg["calendar"],
    }
    (exp / "selection.json").write_text(json.dumps(sel))
    (exp / "manifest.json").write_text(json.dumps(man))
    return panel, exp


@pytest.fixture(scope="module")
def registered(env):
    return WL.register_selected(env[1])


def conn():
    return psycopg.connect(TEST_DSN)


def test_registration_does_not_approve(env, registered):
    assert registered["model_name"] == "retailfc_weekly" and registered["model_version"] == "1"
    with conn() as c:
        assert c.execute("SELECT count(*) FROM app.approved_models").fetchone()[0] == 0
    assert WL.load_spec("retailfc_weekly", "1")["name"] == "mean4"
    with pytest.raises(ML.ModelUnavailable):
        WL.load_spec("retailfc_weekly", "latest")


def test_nothing_selected_means_nothing_registered(env, tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"selected": None}))
    (tmp_path / "manifest.json").write_text((env[1] / "manifest.json").read_text())
    with pytest.raises(ValueError, match="nothing to register"):
        WL.register_selected(tmp_path)


def test_unsupported_lgbm_serving_fails_loudly(env, tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"selected": "lgbm_w"}))
    (tmp_path / "manifest.json").write_text((env[1] / "manifest.json").read_text())
    with pytest.raises(NotImplementedError):
        WL.register_selected(tmp_path)


def test_approval_requires_the_selection_record_to_name_the_model(env, registered, tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"selected": "tsb"}))
    (tmp_path / "manifest.json").write_text((env[1] / "manifest.json").read_text())
    with conn() as c:
        assert WL.approve(c, "1", tmp_path)["status"] == "refused"
        assert c.execute("SELECT count(*) FROM app.approved_models").fetchone()[0] == 0
        assert WL.approve(c, "1", env[1])["status"] == "approved"
        row = c.execute("SELECT approved_by, is_current FROM app.approved_models").fetchone()
    assert row == ("rule:weekly_v2", True)


def _copy_exp(env, tmp_path, **changes):
    sel = json.loads((env[1] / "selection.json").read_text())
    man = json.loads((env[1] / "manifest.json").read_text())
    for k, v in changes.items():
        (man if k in man else sel)[k] = v
    (tmp_path / "selection.json").write_text(json.dumps(sel))
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return tmp_path


def test_approval_verifies_dataset_protocol_config_and_evidence_hashes(env, registered, tmp_path):
    cases = {
        "dataset": {"dataset_version": "retail-other"},
        "protocol": {"protocol_version": "weekly-v3"},
        "config": {"config_sha256": "0" * 64},
    }
    with conn() as c:
        n_before = c.execute("SELECT count(*) FROM app.approved_models WHERE revoked_at IS NULL").fetchone()[
            0
        ]
        for name, ch in cases.items():
            d = tmp_path / name
            d.mkdir()
            r = WL.approve(c, "1", _copy_exp(env, d, **ch))
            assert r["status"] == "refused", (name, r)
        d = tmp_path / "tampered"  # same names, but the evidence files are not the registered ones
        d.mkdir()
        sel = json.loads((env[1] / "selection.json").read_text())
        sel["note"] = "edited after registration"
        (d / "selection.json").write_text(json.dumps(sel))
        (d / "manifest.json").write_text((env[1] / "manifest.json").read_text())
        r = WL.approve(c, "1", d)
        assert r["status"] == "refused" and "selection.json differs" in r["reason"]
        assert (
            c.execute("SELECT count(*) FROM app.approved_models WHERE revoked_at IS NULL").fetchone()[0]
            == n_before
        )


def test_registration_refuses_a_config_that_does_not_match_the_manifest(env, tmp_path):
    bad = tmp_path / "other.yaml"
    bad.write_text(W.DEFAULT_CONFIG.read_text() + "# edited" + chr(10))
    with pytest.raises(WL.IncompatibleConfig):
        WL.register_selected(env[1], cfg_path=bad)


def test_generation_refuses_an_incompatible_config_or_dataset(env, registered, tmp_path):
    panel, _ = env
    other = tmp_path / "weekly_other.yaml"
    other.write_text(W.DEFAULT_CONFIG.read_text() + "# different configuration" + chr(10))
    with conn() as c:
        with pytest.raises(WL.IncompatibleConfig, match="configuration"):
            WL.generate_weekly_batch(c, panel, "1", origin_block=38, mode="historical_replay", cfg_path=other)
        shifted = panel.subset(range(panel.n_series))
        shifted.dataset_version = "retail-another-version"
        with pytest.raises(WL.IncompatibleConfig, match="dataset"):
            WL.generate_weekly_batch(c, shifted, "1", origin_block=38, mode="historical_replay")


def test_weekly_batch_has_explicit_units_periods_and_is_labelled_replay(env, registered):
    panel, _ = env
    with conn() as c:
        r = WL.generate_weekly_batch(c, panel, "1", origin_block=38, mode="historical_replay")
        assert r["status"] == "written" and r["granularity"] == "week" and r["origin_block"] == 38
        b = c.execute(
            "SELECT granularity, mode, protocol_version, feature_version, forecast_origin_day FROM app.forecast_batches"
        ).fetchone()
        assert (
            b[0] == "week"
            and b[1] == "historical_replay"
            and b[2] == "weekly-v2"
            and b[3].startswith("wfv1-")
        )
        assert b[4] == 38 * 7
        f = c.execute(
            "SELECT horizon, horizon_unit, target_period_start, target_period_end, forecast_origin FROM app.forecasts "
            "WHERE item_id=(SELECT min(item_id) FROM app.forecasts) AND store_id='UK_1' ORDER BY horizon"
        ).fetchall()
    assert [x[0] for x in f] == [1, 2, 3, 4] and {x[1] for x in f} == {"week"}
    for h, _u, s, e, o in f:
        assert (e - s).days == 6 and (s - o).days == 7 * (
            h - 1
        ) + 1  # the period starts the day after the origin block (h=1)


def test_production_mode_requires_the_latest_block_and_reruns_are_idempotent(env, registered):
    panel, _ = env
    with conn() as c:
        with pytest.raises(ValueError, match="latest complete block"):
            WL.generate_weekly_batch(c, panel, "1", origin_block=38, mode="production")
        again = WL.generate_weekly_batch(c, panel, "1", origin_block=38, mode="historical_replay")
        assert again["status"] == "noop_identical"
        n = c.execute("SELECT count(*) FROM app.forecasts").fetchone()[0]
        other = WL.generate_weekly_batch(
            c, panel, "1", origin_block=36, mode="historical_replay"
        )  # a new vintage
        assert other["status"] == "written"
        assert (
            c.execute("SELECT count(*) FROM app.forecasts").fetchone()[0] > n
        )  # the old vintage was not overwritten
        assert c.execute("SELECT count(DISTINCT forecast_batch_id) FROM app.forecasts").fetchone()[0] == 2


def test_weekly_reconciliation_scores_only_complete_periods_and_is_idempotent(env, registered):
    panel, _ = env
    with conn() as c:
        # release only part of the target blocks (days 267..290): blocks 39..41 = days 267..287 complete, block 42 incomplete
        store.ingest_actuals(c, store.replay_actuals_frame(panel, 267, 290), "historical_replay")
        c.commit()
        a = store.reconcile(c)
        b = store.reconcile(c)
        assert a["new_errors"] > 0 and b["new_errors"] == 0  # second run adds nothing
        hs = {
            r[0]
            for r in c.execute(
                "SELECT DISTINCT e.horizon FROM app.forecast_errors e JOIN app.forecast_batches b USING (forecast_batch_id) "
                "WHERE e.horizon_unit='week' AND b.forecast_origin_day = 266"
            )
        }
        assert hs == {
            1,
            2,
            3,
        }  # block 42 (days 288..294) lacks days beyond 290 -> unscored, never scored on partial data
        # actual = sum of the 7 daily actuals
        row = c.execute(
            "SELECT e.actual, (SELECT sum(sales) FROM app.actuals x WHERE x.item_id=f.item_id AND x.store_id=f.store_id "
            "AND x.date BETWEEN f.target_period_start AND f.target_period_end) FROM app.forecast_errors e "
            "JOIN app.forecasts f USING (forecast_id) WHERE e.horizon_unit='week' LIMIT 1"
        ).fetchone()
        assert row[0] == row[1]
        cov = MON.actuals_coverage(c, as_of=pd.Timestamp("2100-01-01").date())
        assert (
            cov["coverage"] is not None and cov["coverage"] < 1.0
        )  # unscored periods surface as missing coverage
        perf = MON.performance_by_batch(c)
    assert set(perf["granularity"]) == {"week"} and (perf["wape"] > 0).all()


def test_api_serves_weekly_with_explicit_contract_and_rejects_wrong_horizon(env, registered):
    cfg = C.override(C.load_config(), **{"serving.api_model_name": "retailfc_weekly"})
    with conn() as c:
        item = c.execute("SELECT item_id FROM app.forecasts LIMIT 1").fetchone()[0]
    with TestClient(create_app(TEST_DSN, cfg, api_keys=[KEY]), headers={"x-api-key": KEY}) as client:
        _api_checks(client, item, cfg)


def _api_checks(client, item, cfg):
    r = client.post("/forecast", json={"item_id": item, "store_id": "UK_1", "horizon": 4})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["granularity"] == "week" and j["horizon_unit"] == "week" and len(j["forecasts"]) == 4
    f = j["forecasts"][0]
    assert f["target_period_start"] == f["target_date"] and f["target_period_end"] > f["target_period_start"]
    assert j["mode"] == "historical_replay" and j["model_version"] == "1"
    bad = client.post("/forecast", json={"item_id": item, "store_id": "UK_1", "horizon": 5})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "horizon_out_of_range"
    assert (
        client.post("/forecast", json={"item_id": "nope", "store_id": "UK_1", "horizon": 1}).status_code
        == 404
    )
    with TestClient(create_app(TEST_DSN, cfg, api_keys=[KEY])) as anon:
        assert (
            anon.post("/forecast", json={"item_id": item, "store_id": "UK_1", "horizon": 1}).status_code
            == 401
        )


def test_revocation_makes_the_line_unavailable_and_keeps_the_audit_trail(env, registered):
    cfg = C.override(C.load_config(), **{"serving.api_model_name": "retailfc_weekly"})
    with conn() as c:
        with pytest.raises(ValueError):
            WL.revoke(c, "retailfc_weekly", "1", " ")
        assert WL.revoke(c, "retailfc_weekly", "1", "evidence found invalid in test")["status"] == "revoked"
        row = c.execute(
            "SELECT is_current, revoked_at IS NOT NULL, revoke_reason FROM app.approved_models"
        ).fetchone()
        assert row == (False, True, "evidence found invalid in test")
        item = c.execute("SELECT item_id FROM app.forecasts LIMIT 1").fetchone()[0]
        assert (
            c.execute("SELECT count(*) FROM app.forecasts").fetchone()[0] > 0
        )  # stored vintages are untouched
    with TestClient(create_app(TEST_DSN, cfg, api_keys=[KEY]), headers={"x-api-key": KEY}) as client:
        r = client.post("/forecast", json={"item_id": item, "store_id": "UK_1", "horizon": 1})
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_daily_and_weekly_lines_do_not_collide_on_equal_version_numbers():
    a = store.make_batch_id("1", "dv", "2011-01-01")
    b = store.make_batch_id("1", "dv", "2011-01-01", model_name="retailfc_weekly", granularity="week")
    assert a != b and a.startswith("fb-") and b.startswith("fbw-")


def test_explicit_dataset_change_is_allowed_only_when_asked_for(env, registered):
    panel, _ = env
    shifted = panel.subset(range(panel.n_series))
    shifted.dataset_version = "retail-another-version"
    with conn() as c:
        ok = WL.generate_weekly_batch(
            c,
            shifted,
            "1",
            origin_block=38,
            mode="historical_replay",
            run_label="explicit",
            allow_dataset_change=True,
        )
        assert ok["status"] == "written"
