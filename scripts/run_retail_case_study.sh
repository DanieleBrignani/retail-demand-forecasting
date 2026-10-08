#!/usr/bin/env bash
# Reproduces the Online Retail II analysis end to end. Needs: PostgreSQL, venv with retailfc, DBT_BIN, and the
# file online_retail_II.xlsx that YOU download from https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0).
# Resume: SKIP_DBT=1 SKIP_EDA=1 SKIP_VALIDATION=1 skip stages that already completed (validation outputs must exist).
# Usage: XLSX=/path/online_retail_II.xlsx WORK=/path/outside/synced/folders bash scripts/run_retail_case_study.sh
set -euo pipefail
XLSX=${XLSX:?path to online_retail_II.xlsx}
WORK=${WORK:?working folder outside OneDrive/Dropbox}
export RETAILFC_DSN=${RETAILFC_DSN:-postgresql://retailfc:retailfc@localhost:5442/retailfc_retail}
export RETAILFC_DATA_DIR=$WORK/store RETAILFC_ARTIFACTS_DIR=$WORK/artifacts
export MLFLOW_TRACKING_URI=sqlite:///$WORK/artifacts/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
CFG=configs/online_retail.yaml
mkdir -p "$WORK/artifacts"
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step adapt;    [ -f "$WORK/canonical/calendar.csv" ] || retailfc adapt-online-retail --xlsx "$XLSX" --out "$WORK/canonical" > "$WORK/adapt.json"
step ingest;   V=$(retailfc ingest --src "$WORK/canonical" --label retail \
                 --source-note "UCI Online Retail II (CC BY 4.0) adapted by retailfc.adapters.online_retail" \
                 | python -c "import sys,json; print(json.load(sys.stdin)['dataset_version'])")
echo "dataset_version=$V"
step migrate;  retailfc db migrate
step load;     retailfc db load --version "$V"
if [ -n "${DBT_BIN:-}" ] && [ -z "${SKIP_DBT:-}" ]; then step dbt; eval "$(python scripts/dbt_env.py)"; (cd dbt && "$DBT_BIN" build --vars "{dataset_version: $V}" --profiles-dir .); fi
[ -n "${SKIP_EDA:-}" ] || { step eda; retailfc eda --version "$V"; }
[ -n "${SKIP_VALIDATION:-}" ] || { step validation; retailfc backtest validation --version "$V" --config $CFG; }
EXP=$(ls "$RETAILFC_ARTIFACTS_DIR/experiments" | head -1); echo "experiment=$EXP"
step select;      retailfc backtest select "$EXP" --config $CFG
step uncertainty; retailfc backtest uncertainty --version "$V" --config $CFG
step interpret;   retailfc backtest interpret --version "$V" --config $CFG
step lock;        retailfc protocol-lock --version "$V" --config $CFG
step holdout;     retailfc backtest holdout --version "$V" --config $CFG
step report;      retailfc report build "$EXP" --out "$WORK/report"
step done
