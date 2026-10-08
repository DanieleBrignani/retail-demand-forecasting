"""Command line interface. Every command is a thin wrapper over shared library functions."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from . import config as C

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Retail demand forecasting platform")
db_app = typer.Typer(help="PostgreSQL schema and warehouse")
backtest_app = typer.Typer(help="Temporal backtesting and experiments")
forecast_app = typer.Typer(help="Forecast generation")
actuals_app = typer.Typer(help="Actuals ingestion (historical replay)")
monitor_app = typer.Typer(help="Monitoring")
retrain_app = typer.Typer(help="Retraining decision flow")
api_app = typer.Typer(help="Serving")
report_app = typer.Typer(help="Result tables and charts")
for name, sub in [
    ("db", db_app),
    ("backtest", backtest_app),
    ("forecast", forecast_app),
    ("actuals", actuals_app),
    ("monitor", monitor_app),
    ("retrain", retrain_app),
    ("api", api_app),
    ("report", report_app),
]:
    app.add_typer(sub, name=name)

CFG_OPT = typer.Option(None, "--config", help="experiment YAML (default: packaged experiment.yaml)")


def _echo(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


def _panel(version: str | None, max_series: int | None = None):
    from .panel import load_panel_from_raw

    return load_panel_from_raw(version, max_series=max_series)


@app.command()
def ingest(
    src: Path = typer.Option(..., help="Folder containing calendar.csv, sell_prices.csv and sales_daily.csv"),
    label: str = typer.Option("retail", help="dataset-version prefix"),
    source_note: str = typer.Option(
        None, help="provenance text for the manifest (default: the Online Retail II note)"
    ),
):
    """Validate the canonical CSVs and store an immutable, checksummed raw version."""
    from .ingestion.raw import SOURCE_NOTE, RawIngestError, ingest_raw

    try:
        mf = ingest_raw(src, source_note=source_note or SOURCE_NOTE, label=label)
        v = mf.pop("validation")
        mf["validation_summary"] = {
            "ok": v["ok"],
            "n_checks": len(v["checks"]),
            "warnings": [
                f"{c['name']}: {c['detail']}"
                for c in v["checks"]
                if not c["passed"] or c["severity"] == "warn" and "partial" in c["name"]
            ],
        }
        _echo(mf)
    except RawIngestError as e:
        typer.echo(f"INGEST REFUSED: {e}", err=True)
        raise typer.Exit(2) from e


@app.command("adapt-online-retail")
def adapt_online_retail(
    xlsx: Path = typer.Option(..., help="online_retail_II.xlsx from UCI (CC BY 4.0)"),
    out: Path = typer.Option(..., help="output folder for the canonical CSVs (outside synced folders)"),
    n_future: int = typer.Option(56, help="calendar days appended after the last observed day"),
):
    """Convert UCI Online Retail II transactions to the canonical CSVs + adapter_report.json."""
    from .adapters.online_retail import adapt

    _echo(adapt(xlsx, out, n_future))


@app.command()
def versions():
    """List stored raw dataset versions."""
    from .ingestion import raw

    _echo({"versions": raw.list_versions()})


@app.command("verify")
def verify(version: str):
    """Re-hash a stored raw version against its manifest."""
    from .ingestion import raw

    _echo(raw.verify_version(version)["files"])


@db_app.command("migrate")
def db_migrate():
    from .warehouse import db

    _echo({"applied": db.migrate()})


@db_app.command("load")
def db_load(version: str = typer.Option(None)):
    """Load a verified raw version into PostgreSQL (raw schema); then run `dbt build`."""
    from .ingestion import raw
    from .warehouse import load

    _echo(load.load_raw_to_warehouse(version or raw.latest_version()))


@app.command("protocol-lock")
def protocol_lock(version: str = typer.Option(None), config: Path = CFG_OPT):
    """Lock the experimental protocol (config hash + origins) BEFORE any holdout evaluation."""
    from . import protocol

    cfg = C.load_config(config)
    p = _panel(version)
    _echo(protocol.lock_protocol(cfg, p.dataset_version, p.n_days))


@app.command()
def eda(version: str = typer.Option(None), out: Path = typer.Option(None)):
    """Reproducible EDA tables and charts (training period only for decisions)."""
    from . import eda as E

    cfg = C.load_config()
    _echo(E.run_eda(_panel(version), cfg, out or C.artifacts_dir() / "eda"))


@backtest_app.command("validation")
def bt_validation(
    version: str = typer.Option(None),
    max_series: int = typer.Option(None),
    skip_sarima: bool = False,
    skip_tuning: bool = False,
    config: Path = CFG_OPT,
):
    """Baselines + SARIMA (restricted) + LightGBM A..D on validation origins. Never touches the holdout."""
    from . import experiments as E

    cfg = C.load_config(config)
    if max_series:
        cfg = C.override(cfg, **{"training.max_series": max_series})
    meta = E.run_validation(_panel(version), cfg, skip_sarima=skip_sarima, skip_tuning=skip_tuning)
    _echo({k: v for k, v in meta.items() if k not in ("tuning_trials", "environment")})


@backtest_app.command("variants")
def bt_variants(
    origins: str = typer.Option(..., help="comma-separated forecast origins (day numbers), e.g. 487,515,543"),
    version: str = typer.Option(None),
    tag: str = typer.Option("variants", help="output sub-folder name"),
    config: Path = CFG_OPT,
):
    """Compare model variants (subset vs all series, with/without level offset) on given origins. Baselines included."""
    from . import experiments as E

    cfg = C.load_config(config)
    panel = _panel(version)
    params = None
    mp = C.artifacts_dir() / "experiments"
    for f in sorted(mp.glob(f"*{panel.dataset_version}*/validation/run_meta.json")) if mp.exists() else []:
        params = json.loads(f.read_text()).get("tuned_params") or None
    res = E.run_variants(panel, cfg, [int(x) for x in origins.split(",")], params=params, tag=tag)
    _echo(res)


@backtest_app.command("select")
def bt_select(experiment_id: str, config: Path = CFG_OPT):
    from . import experiments as E

    _echo(E.select_feature_set(C.load_config(config), experiment_id))


@backtest_app.command("holdout")
def bt_holdout(version: str = typer.Option(None), config: Path = CFG_OPT):
    """Locked, single-shot final holdout evaluation."""
    from . import experiments as E

    _echo(E.run_holdout(_panel(version), C.load_config(config)))


@backtest_app.command("uncertainty")
def bt_uncertainty(version: str = typer.Option(None), config: Path = CFG_OPT):
    from . import analysis

    _echo(analysis.run_uncertainty_experiment(_panel(version), C.load_config(config)))


@backtest_app.command("interpret")
def bt_interpret(version: str = typer.Option(None), config: Path = CFG_OPT):
    from . import analysis

    _echo(analysis.run_interpretability(_panel(version), C.load_config(config)))


@app.command("train")
def train(
    feature_set: str = typer.Option("D"),
    version: str = typer.Option(None),
    origin: int = typer.Option(None, help="training origin day (default: last observed day)"),
    no_intervals: bool = False,
    config: Path = CFG_OPT,
):
    """Train a challenger, gather validation evidence, apply promotion rules, register in MLflow."""
    from .ingestion import raw
    from .registry.pipeline import train_evaluate_promote

    cfg = C.load_config(config)
    p = _panel(version)
    params = None
    mp = C.artifacts_dir() / "experiments"
    for f in sorted(mp.glob(f"*{p.dataset_version}*/validation/run_meta.json")) if mp.exists() else []:
        params = json.loads(f.read_text()).get("tuned_params") or None
    res = train_evaluate_promote(
        p,
        cfg,
        feature_key=feature_set,
        params=params,
        train_origin=origin,
        dataset_manifest=raw.read_manifest(p.dataset_version) if p.dataset_version != "unversioned" else None,
        with_intervals=not no_intervals,
    )
    _echo({k: v for k, v in res.items() if k != "evidence"})


@forecast_app.command("generate")
def forecast_generate(
    model_version: str = typer.Option(..., help="explicit registry version (integer)"),
    version: str = typer.Option(None),
    origin: int = typer.Option(None),
    mode: str = typer.Option("production", help="production | historical_replay"),
    run_label: str = "default",
):
    from .forecasting.generate import generate_forecast_batch
    from .warehouse import db

    cfg = C.load_config()
    with db.connect() as conn:
        _echo(
            generate_forecast_batch(
                conn,
                _panel(version),
                cfg["serving"]["model_name"],
                model_version,
                origin=origin,
                mode=mode,
                run_label=run_label,
            )
        )


@actuals_app.command("replay")
def actuals_replay(from_day: int, to_day: int, version: str = typer.Option(None)):
    """HISTORICAL REPLAY: release observed days [from_day, to_day] as newly arrived actuals."""
    from .forecasting import store
    from .warehouse import db

    with db.connect() as conn:
        _echo(
            store.ingest_actuals(
                conn, store.replay_actuals_frame(_panel(version), from_day, to_day), "historical_replay"
            )
        )


@monitor_app.command("refresh")
def monitor_refresh(
    as_of: str = typer.Option(None), version: str = typer.Option(None), train_origin: int = typer.Option(None)
):
    from .monitoring import core
    from .warehouse import db

    p = _panel(version) if train_origin else None
    with db.connect() as conn:
        out = core.refresh(conn, date.fromisoformat(as_of) if as_of else date.today(), p, train_origin)
        _echo(out)


@monitor_app.command("serve")
def monitor_serve(port: int = 9108, interval: int = 30):
    from .monitoring import exporter

    exporter.serve(port, interval)


@retrain_app.command("evaluate")
def retrain_evaluate(
    as_of: str = typer.Option(None),
    dry_run: bool = False,
    feature_set: str = "D",
    version: str = typer.Option(None),
):
    from .orchestration.flows import retraining_evaluation_flow

    _echo(retraining_evaluation_flow(as_of or date.today().isoformat(), version, feature_set, dry_run))


registry_app = typer.Typer(help="Model registry operations")
bench_app = typer.Typer(help="Reproducible latency benchmarks (record results with their environment)")
app.add_typer(registry_app, name="registry")
app.add_typer(bench_app, name="benchmark")


@registry_app.command("rollback")
def registry_rollback(
    to_version: str = typer.Option(..., help="registered version to re-approve"),
    reason: str = typer.Option(..., help="recorded in app.approved_models evidence"),
):
    """Roll back serving to an earlier registered version (audited; stored forecasts are never edited)."""
    from .registry.pipeline import rollback
    from .warehouse import db

    with db.connect() as conn:
        _echo(rollback(conn, C.load_config()["serving"]["model_name"], to_version, reason))


@bench_app.command("inference")
def bench_inference(
    model_version: str,
    version: str = typer.Option(None),
    chunk: int = 500,
    repeats: int = 15,
    out: Path = typer.Option(Path("artifacts/benchmarks/inference.json")),
):
    """p50/p95 of GlobalLGBM.predict (feature build + booster) per chunk of series x 28 horizons."""
    from . import benchmark
    from .registry import mlflow_utils as ML

    panel = _panel(version)
    point, _, _ = ML.load_version(C.load_config()["serving"]["model_name"], model_version)
    res = benchmark.model_inference(panel, point, panel.n_days, chunk=chunk, repeats=repeats)
    benchmark.write_benchmark(
        out,
        {"inference": res},
        {
            "chunk": chunk,
            "repeats": repeats,
            "warmup": 2,
            "dataset_version": panel.dataset_version,
            "model_version": model_version,
        },
    )
    _echo(res)


@bench_app.command("api")
def bench_api(
    base_url: str = "http://localhost:8010",
    requests: int = 500,
    concurrency: int = 1,
    out: Path = typer.Option(Path("artifacts/benchmarks/api.json")),
):
    """p50/p95 of POST /forecast against a running API (stored-forecast retrieval, not model inference)."""
    from . import benchmark
    from .warehouse import db

    with db.connect() as conn:
        keys = conn.execute("SELECT DISTINCT item_id, store_id FROM app.forecasts LIMIT 5000").fetchall()
    if not keys:
        typer.echo("no stored forecasts to benchmark", err=True)
        raise typer.Exit(2)
    res = benchmark.api_latency(base_url, keys, requests=requests, concurrency=concurrency)
    benchmark.write_benchmark(
        out,
        {"api": res},
        {
            "base_url": base_url,
            "requests": requests,
            "concurrency": concurrency,
            "n_distinct_series": len(keys),
        },
    )
    _echo(res)


@api_app.command("serve")
def api_serve(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run("retailfc.api.app:app_factory", factory=True, host=host, port=port, log_level="info")


@report_app.command("build")
def report_build(experiment_id: str, out: Path = typer.Option(None)):
    from . import report

    _echo(report.build_report(experiment_id, out))


@app.command("docs-feature-dictionary")
def docs_feature_dictionary(out: Path = Path("docs/feature_dictionary.md")):
    from .features.docgen import write_feature_dictionary

    write_feature_dictionary(out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()


weekly_app = typer.Typer(help="Weekly-granularity study (protocol: docs/weekly_protocol.md)")
app.add_typer(weekly_app, name="weekly")


@weekly_app.command("backtest")
def weekly_backtest(
    version: str = typer.Option(None),
    config: Path = typer.Option(
        None, "--config", help="weekly protocol YAML (default: configs/weekly_v2.yaml)"
    ),
    tag: str = typer.Option("weekly-v2", help="output sub-folder under artifacts/experiments"),
    no_lgbm: bool = typer.Option(False, help="baselines only (debugging)"),
):
    """Run the frozen weekly protocol: baselines + intermittent methods + global LightGBM, mechanical selection, manifest."""
    from . import weekly

    panel = _panel(version)
    out = C.artifacts_dir() / "experiments" / f"{tag}-{panel.dataset_version}"
    res = weekly.run(panel, out, config, with_lgbm=not no_lgbm)
    for name, txt in res["tables"].items():
        typer.echo(f"\n== {name}\n{txt}")
    typer.echo("\n== selection")
    _echo(res["selection"])
    _echo(res["manifest"])


@weekly_app.command("register")
def weekly_register(
    exp_dir: Path = typer.Argument(..., help="experiment folder written by `weekly backtest`"),
):
    """REGISTRATION of the model named by selection.json (fails if nobody was selected). Does not approve it."""
    from . import weekly_lifecycle as WL

    _echo(WL.register_selected(exp_dir))


@weekly_app.command("approve")
def weekly_approve(
    model_version: str = typer.Option(..., help="explicit registry version of retailfc_weekly"),
    exp_dir: Path = typer.Argument(..., help="experiment folder whose selection.json justifies the approval"),
):
    """APPROVAL by rule: refused unless selection.json names exactly this registered model."""
    from . import weekly_lifecycle as WL
    from .warehouse import db

    with db.connect() as conn:
        _echo(WL.approve(conn, model_version, exp_dir))


@weekly_app.command("forecast")
def weekly_forecast(
    model_version: str = typer.Option(..., help="explicit registry version (integer)"),
    version: str = typer.Option(None, help="dataset version"),
    origin_block: int = typer.Option(None, help="last observed 7-day block (default: latest complete block)"),
    mode: str = typer.Option("production", help="production | historical_replay"),
    run_label: str = "default",
):
    """Store the 4-block forecasts of a registered weekly model (granularity=week, explicit target periods)."""
    from . import weekly_lifecycle as WL
    from .warehouse import db

    with db.connect() as conn:
        _echo(
            WL.generate_weekly_batch(
                conn,
                _panel(version),
                model_version,
                origin_block=origin_block,
                mode=mode,
                run_label=run_label,
            )
        )


@registry_app.command("revoke")
def registry_revoke(
    name: str, version: str, reason: str = typer.Option(..., help="required for the audit trail")
):
    """Withdraw an approval (kept in the audit trail). The API then reports model_unavailable for that line."""
    from . import weekly_lifecycle as WL
    from .warehouse import db

    with db.connect() as conn:
        _echo(WL.revoke(conn, name, version, reason))
