#!/usr/bin/env bash
# Full local demonstration on the SYNTHETIC fixture (plumbing only -- numbers are meaningless as benchmarks).
# Requires: venv with retailfc installed, a PostgreSQL reachable at $RETAILFC_DSN, optional dbt (DBT_BIN).
set -euo pipefail
ROOT=$(pwd -W 2>/dev/null || pwd)   # native path on Windows Git Bash, plain pwd elsewhere
export RETAILFC_DSN=${RETAILFC_DSN:-postgresql://retailfc:retailfc@localhost:5442/retailfc_demo}
export RETAILFC_DATA_DIR=${RETAILFC_DATA_DIR:-$ROOT/data_demo}
export RETAILFC_ARTIFACTS_DIR=${RETAILFC_ARTIFACTS_DIR:-$ROOT/artifacts_demo}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:///$RETAILFC_ARTIFACTS_DIR/mlflow.db}
export MLFLOW_DISABLE_AGENT_HINT=1
CFG=configs/smoke.yaml
mkdir -p "$RETAILFC_ARTIFACTS_DIR"

python scripts/make_fixture.py --out "$RETAILFC_DATA_DIR/fixture_src" --days 300
VERSION=$(retailfc ingest --src "$RETAILFC_DATA_DIR/fixture_src" | python -c "import sys,json; print(json.load(sys.stdin)['dataset_version'])")
retailfc db migrate
retailfc db load --version "$VERSION"
if [ -n "${DBT_BIN:-}" ]; then eval "$(python scripts/dbt_env.py)"; (cd dbt && "$DBT_BIN" build --vars "{dataset_version: $VERSION}" --profiles-dir .); fi

retailfc eda --version "$VERSION"
retailfc backtest validation --version "$VERSION" --config $CFG
EXP=$(ls "$RETAILFC_ARTIFACTS_DIR/experiments" | head -1)
retailfc backtest select "$EXP" --config $CFG
retailfc backtest uncertainty --version "$VERSION" --config $CFG
retailfc backtest interpret --version "$VERSION" --config $CFG
retailfc protocol-lock --version "$VERSION" --config $CFG
retailfc backtest holdout --version "$VERSION" --config $CFG
retailfc report build "$EXP" --out "$RETAILFC_ARTIFACTS_DIR/report"

# model trained through day 272 so a historical replay from origin 272 is leak-free
retailfc train --feature-set D --origin 272 --version "$VERSION" --config $CFG
retailfc forecast generate --model-version 1 --origin 272 --mode historical_replay --version "$VERSION"
retailfc actuals replay 273 300 --version "$VERSION"
retailfc monitor refresh --as-of 2012-01-01 --version "$VERSION" --train-origin 272
retailfc retrain evaluate --as-of 2012-01-01 --dry-run --version "$VERSION"
# --- weekly line (granularity=week): frozen-protocol backtest -> mechanical selection -> registration -> rule approval -> batch -> replay
WCFG=configs/weekly_smoke.yaml
retailfc weekly backtest --version "$VERSION" --config $WCFG --tag weekly-smoke > "$RETAILFC_ARTIFACTS_DIR/weekly_smoke.log"
WEXP="$RETAILFC_ARTIFACTS_DIR/experiments/weekly-smoke-$VERSION"
python -c "import json,sys; s=json.load(open(sys.argv[1]))['selected']; print('weekly selection on the synthetic fixture:', s)" "$WEXP/selection.json"
if python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['selected'] else 1)" "$WEXP/selection.json"; then
  retailfc weekly register "$WEXP"
  retailfc weekly approve --model-version 1 "$WEXP"
  retailfc weekly forecast --model-version 1 --origin-block 38 --mode historical_replay --version "$VERSION"
  retailfc monitor refresh --as-of 2012-01-01 --version "$VERSION" --train-origin 266
else
  echo "no weekly model qualified on the synthetic fixture: nothing registered (the rejection stands)"
fi
echo "Done. Start the API with: retailfc api serve --port 8010  (then POST /forecast)"
