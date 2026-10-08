import os

import pytest

from retailfc import config as C
from tests.synth import make_panel


@pytest.fixture(scope="session")
def cfg():
    return C.load_config()


@pytest.fixture(scope="session")
def panel():
    return make_panel(300)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAILFC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RETAILFC_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    return tmp_path


TEST_DSN = os.environ.get("RETAILFC_TEST_DSN")
needs_db = pytest.mark.skipif(not TEST_DSN, reason="RETAILFC_TEST_DSN not set (real PostgreSQL required)")
