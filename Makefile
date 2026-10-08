# Convenience wrappers (also valid as copy-paste command lists). On Windows use `make` from Git Bash or run the commands directly.
PY ?= python
RFC ?= retailfc

setup:            ## create venv + install locked deps
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.lock && pip install --no-deps -e .[dev]
ingest:           ## SRC=/path/to/canonical/csvs (see `retailfc adapt-online-retail`)
	$(RFC) ingest --src $(SRC)
db:
	$(RFC) db migrate && $(RFC) db load
dbt-build:        ## VERSION=retail-xxxx ; needs dbt-postgres in an isolated env (DBT_BIN)
	cd dbt && $(or $(DBT_BIN),dbt) build --vars '{dataset_version: $(VERSION)}' --profiles-dir .
eda:
	$(RFC) eda
validation:
	$(RFC) backtest validation
lock:
	$(RFC) protocol-lock
holdout:           ## only after `lock`; single-shot
	$(RFC) backtest select $(EXP) && $(RFC) backtest holdout
uncertainty:
	$(RFC) backtest uncertainty && $(RFC) backtest interpret
train:
	$(RFC) train --feature-set D
forecast:          ## MODEL_VERSION=<explicit registry version>
	$(RFC) forecast generate --model-version $(MODEL_VERSION)
report:
	$(RFC) report build $(EXP)
test:
	pytest -m "not integration and not slow" -q
test-all:          ## needs RETAILFC_TEST_DSN (+ DBT_BIN)
	pytest -q
up:
	cp -n .env.example .env; docker compose up -d --build
down:
	docker compose down
lint:
	ruff check src tests scripts monitoring && ruff format --check src tests scripts monitoring
lock-deps:
	uv pip compile pyproject.toml --universal --python-version 3.13 -o requirements.lock
