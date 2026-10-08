-- Raw layer: faithful copies of the three canonical sources, keyed by immutable dataset_version.
CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE raw.dataset_versions (
    dataset_version TEXT PRIMARY KEY,
    schema_version  TEXT NOT NULL,
    first_sales_day INT  NOT NULL,
    last_sales_day  INT  NOT NULL,
    first_date      DATE NOT NULL,
    last_date       DATE NOT NULL,
    manifest        JSONB NOT NULL,
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE raw.calendar (
    dataset_version TEXT NOT NULL REFERENCES raw.dataset_versions,
    date         DATE NOT NULL,
    week_id     INT  NOT NULL,
    weekday      TEXT NOT NULL,
    wday         SMALLINT NOT NULL,
    month        SMALLINT NOT NULL,
    year         SMALLINT NOT NULL,
    d            TEXT NOT NULL,
    event_name_1 TEXT, event_type_1 TEXT, event_name_2 TEXT, event_type_2 TEXT
);

CREATE TABLE raw.series (
    dataset_version TEXT NOT NULL REFERENCES raw.dataset_versions,
    series_ord INT  NOT NULL,           -- row order in the source sales file (stable panel index)
    id TEXT, item_id TEXT NOT NULL, dept_id TEXT NOT NULL, cat_id TEXT NOT NULL,
    store_id TEXT NOT NULL, state_id TEXT NOT NULL
);

-- Long sales: zero is a stored observation; an absent row would mean "not observed" (never produced by the loader).
CREATE TABLE raw.sales_long (
    dataset_version TEXT NOT NULL,
    series_ord INT NOT NULL,
    day_num    INT NOT NULL,
    sales      INT NOT NULL
);

CREATE TABLE raw.prices (
    dataset_version TEXT NOT NULL REFERENCES raw.dataset_versions,
    store_id TEXT NOT NULL, item_id TEXT NOT NULL, week_id INT NOT NULL, sell_price NUMERIC(10,2)
);
