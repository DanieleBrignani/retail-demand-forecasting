CREATE TABLE app.monitoring_aggregates (        -- detailed, unbounded-cardinality aggregates live here (not in Prometheus)
    forecast_batch_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon SMALLINT NOT NULL,
    n BIGINT NOT NULL,
    sum_abs_error DOUBLE PRECISION NOT NULL,
    sum_error DOUBLE PRECISION NOT NULL,
    sum_sq_error DOUBLE PRECISION NOT NULL,
    sum_actual DOUBLE PRECISION NOT NULL,
    n_in_interval BIGINT, n_with_interval BIGINT,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (forecast_batch_id, horizon)
);

CREATE TABLE app.monitoring_snapshots (
    id BIGSERIAL PRIMARY KEY,
    taken_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind TEXT NOT NULL CHECK (kind IN ('drift', 'data_quality', 'performance', 'volume', 'freshness')),
    name TEXT NOT NULL,
    value DOUBLE PRECISION,
    detail JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX monitoring_snapshots_ix ON app.monitoring_snapshots (kind, name, taken_at DESC);

CREATE TABLE app.retraining_decisions (
    decision_id BIGSERIAL PRIMARY KEY,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    decision TEXT NOT NULL,                       -- no_retrain | retrain | investigate | blocked_*
    reasons JSONB NOT NULL,
    inputs JSONB NOT NULL,
    policy JSONB NOT NULL,
    challenger_version TEXT, promotion_decision JSONB
);

CREATE TABLE app.pipeline_runs (
    run_id BIGSERIAL PRIMARY KEY,
    flow TEXT NOT NULL, started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed', 'skipped')), detail JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE app.backtest_metrics (
    backtest_id TEXT NOT NULL, experiment_id TEXT NOT NULL, dataset_version TEXT NOT NULL, feature_version TEXT NOT NULL,
    forecast_origin INT, horizon TEXT NOT NULL, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
    evaluation_scope TEXT NOT NULL, segment_type TEXT NOT NULL, segment_value TEXT NOT NULL,
    aggregation TEXT NOT NULL, metric TEXT NOT NULL, value DOUBLE PRECISION, observation_count BIGINT NOT NULL,
    undefined BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX backtest_metrics_ix ON app.backtest_metrics (experiment_id, model_name, metric);
