"""Load a verified raw dataset version into the PostgreSQL ``raw`` schema (idempotent per version)
and read a panel back from the dbt staging relations."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import psycopg

from ..ingestion import raw as rawmod
from ..panel import ID_COLS, SALES_DTYPE, Panel, read_sales_csv
from .db import connect


def _copy_df(cur: psycopg.Cursor, table: str, cols: list[str], df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, header=False, index=False, na_rep="\\N")
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
        cp.write(buf.getvalue())


def load_raw_to_warehouse(version: str, dsn: str | None = None, chunk_series: int = 250) -> dict:
    mf = rawmod.verify_version(version)  # checksums must match before anything is loaded
    d = rawmod.raw_root() / version
    sales_name = "sales_daily.csv"
    with connect(dsn) as conn:
        if conn.execute("SELECT 1 FROM raw.dataset_versions WHERE dataset_version=%s", (version,)).fetchone():
            return {"dataset_version": version, "status": "already_loaded"}
        o = mf["observed_sales"]
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO raw.dataset_versions VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb, now())",
                (
                    version,
                    mf["schema_version"],
                    1,
                    o["n_days"],
                    o["first_date"],
                    o["last_date"],
                    json.dumps(mf),
                ),
            )
            cal = pd.read_csv(d / "calendar.csv")
            cal.insert(0, "dataset_version", version)
            cal = cal[
                [
                    "dataset_version",
                    "date",
                    "week_id",
                    "weekday",
                    "wday",
                    "month",
                    "year",
                    "d",
                    "event_name_1",
                    "event_type_1",
                    "event_name_2",
                    "event_type_2",
                ]
            ]
            _copy_df(
                cur,
                "raw.calendar",
                [
                    "dataset_version",
                    "date",
                    "week_id",
                    "weekday",
                    "wday",
                    "month",
                    "year",
                    "d",
                    "event_name_1",
                    "event_type_1",
                    "event_name_2",
                    "event_type_2",
                ],
                cal,
            )
            sales = read_sales_csv(d / sales_name, o["n_days"])
            ser = sales[["id", *ID_COLS]].copy()
            ser.insert(0, "series_ord", np.arange(len(ser)))
            ser.insert(0, "dataset_version", version)
            _copy_df(
                cur,
                "raw.series",
                [
                    "dataset_version",
                    "series_ord",
                    "id",
                    "item_id",
                    "dept_id",
                    "cat_id",
                    "store_id",
                    "state_id",
                ],
                ser,
            )
            dcols = [f"d_{i}" for i in range(1, o["n_days"] + 1)]
            mat = sales[dcols].to_numpy()
            T = mat.shape[1]
            days = np.arange(1, T + 1)
            for i in range(0, len(mat), chunk_series):
                blk = mat[i : i + chunk_series]
                df = pd.DataFrame(
                    {
                        "dataset_version": version,
                        "series_ord": np.repeat(np.arange(i, i + len(blk)), T),
                        "day_num": np.tile(days, len(blk)),
                        "sales": blk.reshape(-1),
                    }
                )
                _copy_df(cur, "raw.sales_long", ["dataset_version", "series_ord", "day_num", "sales"], df)
            pr = pd.read_csv(d / "sell_prices.csv")
            pr.insert(0, "dataset_version", version)
            _copy_df(
                cur,
                "raw.prices",
                ["dataset_version", "store_id", "item_id", "week_id", "sell_price"],
                pr[["dataset_version", "store_id", "item_id", "week_id", "sell_price"]],
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS raw_sales_ix ON raw.sales_long (dataset_version, series_ord, day_num)"
            )
            cur.execute("ANALYZE raw.sales_long")
    return {
        "dataset_version": version,
        "status": "loaded",
        "series": int(len(ser)),
        "days": int(T),
        "price_rows": int(len(pr)),
    }


def load_panel_from_staging(version: str, dsn: str | None = None) -> Panel:
    """Panel built from dbt staging relations (``staging.stg_*``). Must equal the raw-file panel (tested)."""
    from ..panel import build_panel

    with connect(dsn) as conn:
        cal = pd.read_sql_query(
            "SELECT to_char(date,'YYYY-MM-DD') AS date, week_id, weekday, wday, month, year, d, event_name_1, "
            "event_type_1, event_name_2, event_type_2 "
            "FROM staging.stg_calendar WHERE dataset_version=%s ORDER BY day_num",
            conn,
            params=(version,),
        )
        ser = pd.read_sql_query(
            "SELECT series_ord, id, item_id, dept_id, cat_id, store_id, state_id FROM "
            "staging.stg_series WHERE dataset_version=%s ORDER BY series_ord",
            conn,
            params=(version,),
        )
        n, T = (
            len(ser),
            int(
                conn.execute(
                    "SELECT last_sales_day FROM raw.dataset_versions WHERE dataset_version=%s", (version,)
                ).fetchone()[0]
            ),
        )
        mat = np.zeros((n, T), dtype=SALES_DTYPE)
        got = 0
        with conn.cursor(name="sales_cur") as cur:
            cur.itersize = 500000
            cur.execute(
                "SELECT series_ord, day_num, sales FROM staging.stg_sales_daily WHERE dataset_version=%s",
                (version,),
            )
            for batch in iter(lambda: cur.fetchmany(500000), []):
                a = np.asarray(batch, dtype=np.int64)
                mat[a[:, 0], a[:, 1] - 1] = a[:, 2]
                got += len(a)
        if got != n * T:
            raise RuntimeError(f"staging sales rows {got} != series*days {n * T}: missing observations")
        prices = pd.read_sql_query(
            "SELECT store_id, item_id, week_id, sell_price::float4 AS sell_price FROM "
            "staging.stg_prices WHERE dataset_version=%s",
            conn,
            params=(version,),
        )
    sales_wide = pd.concat(
        [ser[["id", *ID_COLS]], pd.DataFrame(mat, columns=[f"d_{i}" for i in range(1, T + 1)])], axis=1
    )
    return build_panel(cal, sales_wide, prices, version)
