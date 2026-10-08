import numpy as np
import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from retailfc.api.app import create_app
from retailfc.forecasting import store
from retailfc.warehouse import db
from tests.conftest import TEST_DSN, needs_db

pytestmark = [needs_db, pytest.mark.integration]

KEY = "unit-test-key-0123456789"  # throwaway test credential, not a real secret
OTHER_KEY = "second-test-key-9876543210"


def app_client(dsn=None, cfg=None, keys=None, **kw):
    """TestClient that sends a valid API key by default."""
    keys = [KEY] if keys is None else keys
    return TestClient(create_app(dsn or TEST_DSN, cfg, api_keys=keys, **kw), headers={"x-api-key": KEY})


@pytest.fixture()
def clean():
    with psycopg.connect(TEST_DSN, autocommit=True) as c:
        for s in ("app", "raw", "staging"):
            c.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        c.execute("DROP TABLE IF EXISTS public.schema_migrations")
    db.migrate(TEST_DSN)


def _seed(approve=True, with_batch=True):
    with psycopg.connect(TEST_DSN) as c:
        if approve:
            c.execute(
                "INSERT INTO app.approved_models (model_name, model_version, dataset_version, approved_by) "
                "VALUES ('retailfc_lgbm','7','dv','rule')"
            )
        if with_batch:
            dates = pd.date_range("2012-01-02", periods=28).strftime("%Y-%m-%d")
            rows = [
                ("RETAIL_1_001", "UK_1", d, h + 1, 2.0 + h, 1.0, 4.0 + h, "" if h < 26 else "unseen_series")
                for h, d in enumerate(dates)
            ]
            rec = pd.DataFrame(
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
            store.write_batch(
                c,
                rec,
                model_name="retailfc_lgbm",
                model_version="7",
                dataset_version="dv",
                feature_version="fv9",
                origin_date="2012-01-01",
                origin_day=100,
                mode="historical_replay",
            )
            c.commit()


@pytest.fixture()
def client(clean):
    _seed()
    with app_client() as c:
        yield c


def test_forecast_happy_path_records_versions_and_intervals(client):
    r = client.post(
        "/forecast",
        json={"item_id": "RETAIL_1_001", "store_id": "UK_1", "horizon": 7},
        headers={"x-request-id": "abc-123"},
    )
    assert r.status_code == 200 and r.headers["x-request-id"] == "abc-123"
    j = r.json()
    assert j["model_version"] == "7" and j["dataset_version"] == "dv" and j["feature_version"] == "fv9"
    assert j["forecast_origin"] == "2012-01-01" and len(j["forecasts"]) == 7
    assert j["forecasts"][0]["target_date"] == "2012-01-02" and j["forecasts"][0]["lower"] == 1.0
    assert j["fallback"]["used"] is False and j["generated_at"]


def test_fallback_information_is_surfaced(client):
    j = client.post("/forecast", json={"item_id": "RETAIL_1_001", "store_id": "UK_1", "horizon": 28}).json()
    assert (
        j["fallback"]["used"]
        and j["fallback"]["reasons"] == ["unseen_series"]
        and j["fallback"]["horizons"] == [27, 28]
    )


@pytest.mark.parametrize(
    "body",
    [
        {"item_id": "x", "store_id": "y", "horizon": 0},
        {"item_id": "x", "store_id": "y", "horizon": 29},
        {"item_id": "", "store_id": "y", "horizon": 3},
        {"store_id": "y", "horizon": 3},
    ],
)
def test_invalid_input_rejected_with_explicit_error(client, body):
    r = client.post("/forecast", json=body)
    assert (
        r.status_code == 422
        and r.json()["error"]["code"] == "invalid_request"
        and r.json()["error"]["request_id"]
    )


def test_unknown_series_is_404_not_a_fake_forecast(client):
    r = client.post("/forecast", json={"item_id": "NOPE_1_001", "store_id": "UK_1", "horizon": 3})
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_series"


def test_model_unavailable_when_nothing_approved(clean):
    _seed(approve=False, with_batch=False)
    with app_client() as c:
        r = c.post("/forecast", json={"item_id": "a", "store_id": "b", "horizon": 3})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
        assert c.get("/health/ready").status_code == 503
        assert c.get("/health/live").status_code == 200  # liveness independent of dependencies


def test_approved_model_without_forecast_batch_is_503(clean):
    _seed(approve=True, with_batch=False)
    with app_client() as c:
        r = c.post("/forecast", json={"item_id": "a", "store_id": "b", "horizon": 3})
        assert r.status_code == 503 and r.json()["error"]["code"] == "no_forecast_batch"


def test_database_unavailable_is_explicit():
    with app_client("postgresql://retailfc:retailfc@localhost:1/none") as c:
        r = c.post("/forecast", json={"item_id": "a", "store_id": "b", "horizon": 3})
        assert r.status_code == 503 and r.json()["error"]["code"] == "database_unavailable"
        assert c.get("/health/ready").status_code == 503 and c.get("/health/live").status_code == 200


def test_ready_and_metrics(client):
    assert client.get("/health/ready").json()["model_version"] == "7"
    client.post("/forecast", json={"item_id": "RETAIL_1_001", "store_id": "UK_1", "horizon": 2})
    m = client.get("/metrics").text
    assert 'retailfc_api_requests_total{endpoint="/forecast",status="200"}' in m
    assert "item_id" not in m and "RETAIL" not in m  # bounded label cardinality


def test_zero_capacity_rejects_immediately_instead_of_queueing(client):
    cfg = {"serving": {"model_name": "retailfc_lgbm", "max_concurrency": 0, "request_timeout_seconds": 2}}
    with app_client(TEST_DSN, cfg) as c:
        r = c.post("/forecast", json={"item_id": "RETAIL_1_001", "store_id": "UK_1", "horizon": 3})
        assert r.status_code == 503 and r.json()["error"]["code"] == "overloaded"


def test_parallel_requests_all_succeed_within_capacity(client):
    import threading

    res = []

    def go():
        res.append(
            client.post(
                "/forecast", json={"item_id": "RETAIL_1_001", "store_id": "UK_1", "horizon": 28}
            ).status_code
        )

    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert res == [200] * 8


def test_retrieval_is_deterministic_for_repeated_requests(client):
    a = client.post("/forecast", json={"item_id": "RETAIL_1_001", "store_id": "UK_1", "horizon": 5}).json()
    b = client.post("/forecast", json={"item_id": "RETAIL_1_001", "store_id": "UK_1", "horizon": 5}).json()
    assert a == b and np.isfinite([f["forecast"] for f in a["forecasts"]]).all()


# ------------------------------------------------------------------ authentication
BODY = {"item_id": "RETAIL_1_001", "store_id": "UK_1", "horizon": 3}


def test_missing_key_is_401_and_never_reaches_the_database(clean):
    _seed()
    with TestClient(create_app(TEST_DSN, api_keys=[KEY])) as c:  # no header
        r = c.post("/forecast", json=BODY)
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
        assert r.headers["www-authenticate"].startswith("ApiKey") and r.headers["x-request-id"]
        assert "forecasts" not in r.json()


def test_wrong_key_is_401_and_valid_key_is_200(clean):
    _seed()
    with TestClient(create_app(TEST_DSN, api_keys=[KEY])) as c:
        assert (
            c.post("/forecast", json=BODY, headers={"x-api-key": "wrong-key-wrong-key-1"}).status_code == 401
        )
        assert (
            c.post("/forecast", json=BODY, headers={"x-api-key": KEY[:-1]}).status_code == 401
        )  # prefix of a key
        assert c.post("/forecast", json=BODY, headers={"x-api-key": KEY}).status_code == 200


def test_authentication_precedes_validation_and_does_not_leak_series_existence(clean):
    _seed()
    with TestClient(create_app(TEST_DSN, api_keys=[KEY])) as c:
        bad = c.post("/forecast", json={"item_id": "", "horizon": 99})  # invalid body, no key
        unknown = c.post("/forecast", json={**BODY, "item_id": "NOPE_1_001"})  # unknown series, no key
        assert bad.status_code == 401 and unknown.status_code == 401


def test_two_live_keys_support_rotation(clean):
    _seed()
    with TestClient(create_app(TEST_DSN, api_keys=[KEY, OTHER_KEY])) as c:
        for k in (KEY, OTHER_KEY):
            assert c.post("/forecast", json=BODY, headers={"x-api-key": k}).status_code == 200


def test_fail_closed_when_no_key_is_configured(clean, monkeypatch):
    _seed()
    monkeypatch.delenv("RETAILFC_API_KEYS", raising=False)
    monkeypatch.delenv("RETAILFC_API_AUTH", raising=False)
    with TestClient(create_app(TEST_DSN)) as c:
        r = c.post("/forecast", json=BODY, headers={"x-api-key": KEY})
        assert r.status_code == 503 and r.json()["error"]["code"] == "auth_not_configured"
        assert c.get("/health/ready").status_code == 503
        assert c.get("/health/live").status_code == 200


def test_keys_are_read_from_environment(clean, monkeypatch):
    _seed()
    monkeypatch.setenv("RETAILFC_API_KEYS", f" {KEY} , {OTHER_KEY} ")
    with TestClient(create_app(TEST_DSN)) as c:
        assert c.post("/forecast", json=BODY, headers={"x-api-key": OTHER_KEY}).status_code == 200
        assert c.post("/forecast", json=BODY).status_code == 401
        assert c.get("/health/ready").json()["auth"] == "enabled"


def test_auth_can_only_be_disabled_explicitly(clean, monkeypatch):
    _seed()
    monkeypatch.delenv("RETAILFC_API_KEYS", raising=False)
    monkeypatch.setenv("RETAILFC_API_AUTH", "disabled")
    with TestClient(create_app(TEST_DSN)) as c:
        assert c.post("/forecast", json=BODY).status_code == 200
        assert c.get("/health/ready").json()["auth"] == "disabled"


def test_weak_keys_are_rejected_at_startup():
    with pytest.raises(ValueError, match="at least 16"):
        create_app(TEST_DSN, api_keys=["short"])


def test_key_never_appears_in_logs_or_metrics(clean, caplog):
    import logging

    _seed()
    caplog.set_level(logging.INFO, logger="retailfc.api")
    with TestClient(create_app(TEST_DSN, api_keys=[KEY])) as c:
        c.post("/forecast", json=BODY, headers={"x-api-key": "wrong-key-wrong-key-1"})
        c.post("/forecast", json=BODY, headers={"x-api-key": KEY})
        metrics = c.get("/metrics").text
    assert KEY not in caplog.text and "wrong-key-wrong-key-1" not in caplog.text and KEY not in metrics
    assert "auth_failure" in caplog.text  # failures are logged, without the credential


def test_health_probes_stay_open_without_a_key(clean):
    _seed()
    with TestClient(create_app(TEST_DSN, api_keys=[KEY])) as c:
        assert c.get("/health/live").status_code == 200 and c.get("/health/ready").status_code == 200
