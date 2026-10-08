#!/usr/bin/env bash
# Operational half of the weekly line (after `retailfc weekly backtest`): migrate, REGISTER the selected model, APPROVE it by rule,
# generate a HISTORICAL-REPLAY batch at the chosen origin block, release the following days as actuals, reconcile/monitor.
# Registration, approval and deployment are separate steps; approval is refused unless selection.json names the registered model.
# Usage: WORK=/path V=retail-xxxxxxxxxxxx [ORIGIN_BLOCK=101] [REVOKE_DAILY=1] bash scripts/run_weekly_operations.sh
set -euo pipefail
WORK=${WORK:?working folder used by the case study}
V=${V:?dataset version, e.g. retail-369d316281ea}
ORIGIN_BLOCK=${ORIGIN_BLOCK:-101}
ASOF=${ASOF:-2011-12-09}
export RETAILFC_DSN=${RETAILFC_DSN:-postgresql://retailfc:retailfc@localhost:5442/retailfc_retail}
export RETAILFC_DATA_DIR=$WORK/store RETAILFC_ARTIFACTS_DIR=$WORK/artifacts
export MLFLOW_TRACKING_URI=sqlite:///$WORK/artifacts/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
EXP=${EXP:-$WORK/artifacts/experiments/weekly-v2-$V}
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step migrate;  retailfc db migrate
step register; retailfc weekly register "$EXP"
MV=${MODEL_VERSION:-$(python -c "import mlflow;from mlflow.tracking import MlflowClient as C;print(max(int(v.version) for v in C('$MLFLOW_TRACKING_URI').search_model_versions(\"name='retailfc_weekly'\")))")}
step approve;  retailfc weekly approve --model-version "$MV" "$EXP"
if [ "${REVOKE_DAILY:-0}" = "1" ]; then
  step revoke-daily
  retailfc registry revoke retailfc_lgbm "${DAILY_VERSION:-2}" --reason "approval rests on feature set D with fv1 prices (look-ahead defect), see docs/audit_price_leakage.md"
fi
step forecast; retailfc weekly forecast --model-version "$MV" --origin-block "$ORIGIN_BLOCK" --mode historical_replay --version "$V"
step replay;   retailfc actuals replay $((ORIGIN_BLOCK * 7 + 1)) 739 --version "$V"
step monitor;  retailfc monitor refresh --as-of "$ASOF" --version "$V" --train-origin $((ORIGIN_BLOCK * 7))
step done
