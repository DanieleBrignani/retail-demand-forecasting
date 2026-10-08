#!/usr/bin/env bash
# Operational half of the real-data run (after scripts/run_retail_case_study.sh): train -> evidence -> promotion rules -> registry,
# batch forecast, HISTORICAL REPLAY of the following days as actuals, reconciliation/monitoring, retraining decision, inference timing.
#
# The model is trained with data up to ORIGIN and forecasts days ORIGIN+1..ORIGIN+28; those days are then "released" as actuals.
# (ORIGIN defaults to the holdout origin; the replay window therefore coincides with the already-consumed holdout window --
#  this is an operational demonstration, not a new evaluation or any kind of model selection.)
# Usage (MODEL_VERSION = registry version produced by the train step, default 1): WORK=/path/outside/synced/folders V=retail-xxxxxxxxxxxx bash scripts/run_retail_operations.sh
set -euo pipefail
WORK=${WORK:?working folder used by run_retail_case_study.sh}
V=${V:?dataset version, e.g. retail-369d316281ea}
ORIGIN=${ORIGIN:-711}
ASOF=${ASOF:-2011-12-09}          # last observed date of the dataset
LAST_DAY=${LAST_DAY:-739}
export RETAILFC_DSN=${RETAILFC_DSN:-postgresql://retailfc:retailfc@localhost:5442/retailfc_retail}
export RETAILFC_DATA_DIR=$WORK/store RETAILFC_ARTIFACTS_DIR=$WORK/artifacts
export MLFLOW_TRACKING_URI=sqlite:///$WORK/artifacts/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
CFG=configs/online_retail.yaml
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step train;     retailfc train --feature-set D --origin "$ORIGIN" --version "$V" --config $CFG
step forecast;  retailfc forecast generate --model-version "${MODEL_VERSION:-1}" --origin "$ORIGIN" --mode historical_replay --version "$V"
step replay;    retailfc actuals replay $((ORIGIN + 1)) "$LAST_DAY" --version "$V"
step monitor;   retailfc monitor refresh --as-of "$ASOF" --version "$V" --train-origin "$ORIGIN"
step retrain;   retailfc retrain evaluate --as-of "$ASOF" --dry-run --feature-set D --version "$V"
step inference; retailfc benchmark inference "${MODEL_VERSION:-1}" --version "$V" --out "$WORK/artifacts/benchmarks/inference.json"
step done
