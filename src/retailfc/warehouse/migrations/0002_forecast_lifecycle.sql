-- Forecast storage, actuals, errors. Forecasts are append-only per batch; reruns never overwrite.
CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE app.forecast_batches (
    forecast_batch_id TEXT PRIMARY KEY,
    model_name        TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    dataset_version   TEXT NOT NULL,
    feature_version   TEXT NOT NULL,
    forecast_origin   DATE NOT NULL,            -- last date with sales information used
    forecast_origin_day INT NOT NULL,
    generation_time   TIMESTAMPTZ NOT NULL,     -- when the forecast was produced
    data_available_at TIMESTAMPTZ,              -- when the origin's data became available (replay: = ingestion time)
    mode              TEXT NOT NULL CHECK (mode IN ('production', 'historical_replay')),
    price_scenario    TEXT NOT NULL DEFAULT 'observed',
    run_label         TEXT NOT NULL DEFAULT 'default',
    status            TEXT NOT NULL CHECK (status IN ('writing', 'complete', 'failed')),
    n_rows            INT,
    content_hash      TEXT,
    UNIQUE (model_version, dataset_version, forecast_origin, run_label)
);

CREATE TABLE app.forecasts (
    forecast_id       UUID PRIMARY KEY,
    forecast_batch_id TEXT NOT NULL REFERENCES app.forecast_batches ON DELETE RESTRICT,
    model_version     TEXT NOT NULL,
    dataset_version   TEXT NOT NULL,
    feature_version   TEXT NOT NULL,
    forecast_origin   DATE NOT NULL,
    generation_time   TIMESTAMPTZ NOT NULL,
    item_id           TEXT NOT NULL,
    store_id          TEXT NOT NULL,
    target_date       DATE NOT NULL,
    horizon           SMALLINT NOT NULL CHECK (horizon BETWEEN 1 AND 28),
    forecast_value    DOUBLE PRECISION NOT NULL CHECK (forecast_value >= 0),
    lower_bound       DOUBLE PRECISION,
    upper_bound       DOUBLE PRECISION,
    fallback_used     TEXT NOT NULL DEFAULT '',
    UNIQUE (forecast_batch_id, item_id, store_id, target_date),
    CHECK (target_date > forecast_origin),
    CHECK (lower_bound IS NULL OR upper_bound IS NULL OR lower_bound <= upper_bound)
);
CREATE INDEX forecasts_series_ix ON app.forecasts (item_id, store_id, forecast_origin DESC);
CREATE INDEX forecasts_target_ix ON app.forecasts (target_date);

CREATE TABLE app.actuals (
    item_id   TEXT NOT NULL,
    store_id  TEXT NOT NULL,
    date      DATE NOT NULL,
    sales     INT  NOT NULL CHECK (sales >= 0),
    source    TEXT NOT NULL,                     -- 'production' | 'historical_replay'
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (item_id, store_id, date)
);

CREATE TABLE app.actuals_conflicts (              -- same key, different value: never silently overwritten
    id BIGSERIAL PRIMARY KEY, item_id TEXT, store_id TEXT, date DATE,
    existing_sales INT, incoming_sales INT, detected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Errors are derived rows, one per (forecast, actual) pair; forecasts are untouched.
CREATE TABLE app.forecast_errors (
    forecast_id UUID PRIMARY KEY REFERENCES app.forecasts,
    forecast_batch_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon SMALLINT NOT NULL,
    target_date DATE NOT NULL,
    actual DOUBLE PRECISION NOT NULL,
    error DOUBLE PRECISION NOT NULL,             -- forecast - actual (positive = over-forecast)
    abs_error DOUBLE PRECISION NOT NULL,
    sq_error DOUBLE PRECISION NOT NULL,
    in_interval BOOLEAN,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX forecast_errors_batch_ix ON app.forecast_errors (forecast_batch_id, horizon);

CREATE TABLE app.approved_models (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL, model_version TEXT NOT NULL, dataset_version TEXT NOT NULL,
    approved_at TIMESTAMPTZ NOT NULL DEFAULT now(), approved_by TEXT NOT NULL,   -- rule id or 'manual'
    evidence JSONB NOT NULL DEFAULT '{}', is_current BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE UNIQUE INDEX approved_models_one_current ON app.approved_models (model_name) WHERE is_current;
