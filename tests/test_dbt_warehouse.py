"""Warehouse + dbt integration (real PostgreSQL and dbt-postgres). Needs RETAILFC_TEST_DSN and DBT_BIN."""

import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import psycopg
import pytest

from retailfc.ingestion import raw
from retailfc.panel import load_panel_from_raw
from retailfc.warehouse import db, load
from tests.conftest import TEST_DSN, needs_db
from tests.synth import write_csvs

DBT = os.environ.get("DBT_BIN")
pytestmark = [
    needs_db,
    pytest.mark.integration,
    pytest.mark.skipif(not DBT or not Path(DBT).exists(), reason="DBT_BIN (dbt-postgres) not available"),
]
ROOT = Path(__file__).resolve().parents[1]


def dbt(cmd: str, version: str):
    u = urlparse(TEST_DSN)
    env = {
        **os.environ,
        "DBT_HOST": u.hostname,
        "DBT_PORT": str(u.port or 5432),
        "DBT_USER": u.username,
        "DBT_PASSWORD": u.password,
        "DBT_DBNAME": u.path.strip("/"),
    }
    return subprocess.run(
        [
            DBT,
            cmd,
            "--vars",
            f"{{dataset_version: {version}}}",
            "--profiles-dir",
            str(ROOT / "dbt"),
            "--project-dir",
            str(ROOT / "dbt"),
            "--target-path",
            str(ROOT / "dbt" / "target_test"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    os.environ["RETAILFC_DATA_DIR"] = str(d / "data")
    with psycopg.connect(TEST_DSN, autocommit=True) as c:
        for s in ("app", "raw", "staging"):
            c.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        c.execute("DROP TABLE IF EXISTS public.schema_migrations")
    db.migrate(TEST_DSN)
    src = write_csvs(d / "src", 150, n_items=18)
    v = raw.ingest_raw(src)["dataset_version"]
    assert load.load_raw_to_warehouse(v, TEST_DSN)["status"] == "loaded"
    assert load.load_raw_to_warehouse(v, TEST_DSN)["status"] == "already_loaded"  # idempotent
    return v


def test_dbt_build_passes_on_valid_data(loaded):
    r = dbt("build", loaded)
    assert r.returncode == 0, r.stdout[-3000:]
    assert "ERROR=0" in r.stdout


def test_staging_panel_equals_raw_file_panel(loaded):
    a = load_panel_from_raw(loaded)
    b = load.load_panel_from_staging(loaded, TEST_DSN)
    assert a.ids.equals(b.ids) and np.array_equal(a.sales, b.sales)
    np.testing.assert_allclose(a.price_weekly, b.price_weekly, equal_nan=True, rtol=1e-6)
    assert (a.calendar["date"].astype(str).to_numpy() == b.calendar["date"].astype(str).to_numpy()).all()


def test_dbt_tests_fail_on_injected_problems(loaded):
    """Negative controls: each injected defect must make `dbt test` fail."""
    assert dbt("build", loaded).returncode == 0
    with psycopg.connect(TEST_DSN, autocommit=True) as c:
        c.execute("UPDATE staging.stg_sales_daily SET sales = -3 WHERE series_ord=0 AND day_num=5")
        c.execute("INSERT INTO staging.stg_prices SELECT * FROM staging.stg_prices LIMIT 1")  # duplicate key
        c.execute("DELETE FROM staging.stg_calendar WHERE day_num = 20")  # missing date
    r = dbt("test", loaded)
    assert r.returncode != 0
    out = r.stdout
    assert "assert_valid_sales_and_prices" in out and "FAIL" in out
    assert "assert_key_uniqueness" in out and "assert_calendar_integrity" in out
    # rebuild restores a clean state
    assert dbt("build", loaded).returncode == 0


def test_warehouse_load_and_staging_panel_keep_large_quantities(tmp_path, monkeypatch):
    """Regression: an int16 read wrapped 74,215 to 8,679 in the raw load and in the staging read-back (equal but wrong)."""
    import numpy as np
    import pandas as pd

    from retailfc.panel import load_panel_from_raw

    monkeypatch.setenv("RETAILFC_DATA_DIR", str(tmp_path / "data"))
    with psycopg.connect(TEST_DSN, autocommit=True) as c:
        for s in ("app", "raw", "staging"):
            c.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        c.execute("DROP TABLE IF EXISTS public.schema_migrations")
    db.migrate(TEST_DSN)
    src = write_csvs(tmp_path / "src", 60, n_items=4)
    sales = pd.read_csv(src / "sales_daily.csv")
    sales.loc[0, "d_5"] = 74215
    sales.loc[1, "d_9"] = 80995
    sales.to_csv(src / "sales_daily.csv", index=False)
    v = raw.ingest_raw(src)["dataset_version"]
    load.load_raw_to_warehouse(v, TEST_DSN)
    expect = int(sales[[c for c in sales.columns if c.startswith("d_")]].to_numpy(np.int64).sum())
    with psycopg.connect(TEST_DSN) as c:
        assert (
            c.execute(
                "SELECT sum(sales)::bigint FROM raw.sales_long WHERE dataset_version=%s", (v,)
            ).fetchone()[0]
            == expect
        )
        assert (
            c.execute("SELECT max(sales) FROM raw.sales_long WHERE dataset_version=%s", (v,)).fetchone()[0]
            == 80995
        )
    with psycopg.connect(TEST_DSN, autocommit=True) as c:
        c.execute("CREATE SCHEMA IF NOT EXISTS staging")
    from_raw = load_panel_from_raw(v)
    assert int(from_raw.sales.sum(dtype=np.int64)) == expect and from_raw.sales.max() == 80995
