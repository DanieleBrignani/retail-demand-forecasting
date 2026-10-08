-- Forecast granularity: daily and weekly forecasts share the lifecycle tables but are never confusable.
-- * horizon_unit says what `horizon` counts ('day' 1..28, 'week' 1..4); target_period_start/end give the covered period explicitly.
--   For a daily row the period is the single target date. For a weekly row `target_date` = period start (dataset 7-day block).
-- * A batch is unique per MODEL NAME + version (daily and weekly registry names both start at version 1).
-- Existing rows are backfilled as daily; no historical forecast vintage is touched or removed.
ALTER TABLE app.forecast_batches
    ADD COLUMN granularity TEXT NOT NULL DEFAULT 'day' CHECK (granularity IN ('day', 'week')),
    ADD COLUMN protocol_version TEXT NOT NULL DEFAULT '';
ALTER TABLE app.forecast_batches DROP CONSTRAINT forecast_batches_model_version_dataset_version_forecast_ori_key;
ALTER TABLE app.forecast_batches
    ADD CONSTRAINT forecast_batches_unique_run UNIQUE (model_name, model_version, dataset_version, forecast_origin, run_label);

ALTER TABLE app.forecasts
    ADD COLUMN horizon_unit TEXT NOT NULL DEFAULT 'day' CHECK (horizon_unit IN ('day', 'week')),
    ADD COLUMN target_period_start DATE,
    ADD COLUMN target_period_end DATE;
UPDATE app.forecasts SET target_period_start = target_date, target_period_end = target_date WHERE target_period_start IS NULL;
ALTER TABLE app.forecasts ALTER COLUMN target_period_start SET NOT NULL, ALTER COLUMN target_period_end SET NOT NULL;
ALTER TABLE app.forecasts DROP CONSTRAINT forecasts_horizon_check;
ALTER TABLE app.forecasts
    ADD CONSTRAINT forecasts_horizon_range_check
        CHECK ((horizon_unit = 'day' AND horizon BETWEEN 1 AND 28) OR (horizon_unit = 'week' AND horizon BETWEEN 1 AND 4)),
    ADD CONSTRAINT forecasts_period_check
        CHECK (target_period_end >= target_period_start AND target_period_start = target_date);
CREATE INDEX forecasts_period_ix ON app.forecasts (target_period_end);

ALTER TABLE app.forecast_errors
    ADD COLUMN horizon_unit TEXT NOT NULL DEFAULT 'day',
    ADD COLUMN target_period_end DATE;
UPDATE app.forecast_errors SET target_period_end = target_date WHERE target_period_end IS NULL;

-- Approval records are never deleted; an approval can be revoked with a reason (e.g. evidence found invalid).
ALTER TABLE app.approved_models
    ADD COLUMN revoked_at TIMESTAMPTZ,
    ADD COLUMN revoke_reason TEXT;
