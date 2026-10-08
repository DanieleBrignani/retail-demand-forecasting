"""Prefect flows (single orchestrator). Flow bodies call the same shared functions as the CLI.

Deployment: ``python -m retailfc.orchestration.serve`` registers scheduled deployments with a Prefect server.
Retries are applied only to idempotent steps (forecast writes, actuals ingestion, reconcile are idempotent).
"""

from __future__ import annotations

from datetime import date

from prefect import flow, get_run_logger, task

from .. import config as C
from ..forecasting import generate as G
from ..forecasting import store
from ..monitoring import core as MON
from ..panel import load_panel_from_raw
from ..warehouse import db
from . import retraining as RT


def _run(conn, flow_name: str) -> int:
    rid = conn.execute(
        "INSERT INTO app.pipeline_runs (flow, status) VALUES (%s,'running') RETURNING run_id", (flow_name,)
    ).fetchone()[0]
    conn.commit()
    return rid


def _finish(conn, rid: int, status: str, detail: dict) -> None:
    import json

    conn.execute(
        "UPDATE app.pipeline_runs SET status=%s, finished_at=now(), detail=%s::jsonb WHERE run_id=%s",
        (status, json.dumps(detail, default=str), rid),
    )
    conn.commit()


def _tracked(flow_name: str, fn):
    """Run fn(conn) while recording the pipeline run; failures are recorded (not swallowed) and re-raised."""
    with db.connect() as conn:
        rid = _run(conn, flow_name)
        try:
            out = fn(conn)
            _finish(
                conn,
                rid,
                "skipped"
                if isinstance(out, dict) and str(out.get("status", "")).startswith("skipped")
                else "succeeded",
                out if isinstance(out, dict) else {},
            )
            return out
        except Exception as e:
            conn.rollback()
            _finish(conn, rid, "failed", {"error": repr(e)})
            raise


@task(retries=2, retry_delay_seconds=10)
def t_replay_actuals(panel, from_day: int, to_day: int, conn_dsn=None) -> dict:
    with db.connect(conn_dsn) as conn:
        return store.ingest_actuals(
            conn, store.replay_actuals_frame(panel, from_day, to_day), "historical_replay"
        )


@flow(name="forecast-generation")
def forecast_generation_flow(
    version: str,
    origin: int | None = None,
    mode: str = "production",
    run_label: str = "default",
    dataset_version: str | None = None,
) -> dict:
    cfg = C.load_config()
    panel = load_panel_from_raw(dataset_version)
    return _tracked(
        "forecast-generation",
        lambda conn: G.generate_forecast_batch(
            conn, panel, cfg["serving"]["model_name"], version, origin=origin, mode=mode, run_label=run_label
        ),
    )


@flow(name="actuals-and-monitoring")
def actuals_monitoring_flow(
    from_day: int,
    to_day: int,
    as_of: str,
    dataset_version: str | None = None,
    train_origin: int | None = None,
) -> dict:
    panel = load_panel_from_raw(dataset_version)

    def body(conn):
        ing = store.ingest_actuals(
            conn, store.replay_actuals_frame(panel, from_day, to_day), "historical_replay"
        )
        mon = MON.refresh(conn, date.fromisoformat(as_of), panel, train_origin)
        return {"actuals": ing, "monitoring": {k: v for k, v in mon.items() if k != "performance"}}

    return _tracked("actuals-and-monitoring", body)


@flow(name="retraining-evaluation")
def retraining_evaluation_flow(
    as_of: str, dataset_version: str | None = None, feature_set: str = "D", dry_run: bool = False
) -> dict:
    """Scheduled decision flow. Trains/promotes only if the policy says so; always audits the decision."""
    log = get_run_logger()
    cfg = C.load_config()
    pol = cfg["retraining"]
    with db.connect() as conn:
        inputs = RT.gather_inputs(conn, cfg, date.fromisoformat(as_of))
        dec = RT.decide(inputs, pol)
        log.info("retraining decision: %s %s", dec.decision, dec.reasons)
        if dec.decision != "retrain" or dry_run:
            did = RT.record_decision(conn, dec, inputs, pol)
            return {"decision_id": did, **dec.to_dict()}
    from ..registry.pipeline import train_evaluate_promote

    panel = load_panel_from_raw(dataset_version)
    res = train_evaluate_promote(panel, cfg, feature_key=feature_set, params=None)
    with db.connect() as conn:
        did = RT.record_decision(conn, dec, inputs, pol, res.get("model_version"), res.get("decision"))
    return {
        "decision_id": did,
        **dec.to_dict(),
        "train_status": res.get("status"),
        "challenger_version": res.get("model_version"),
        "promoted": res.get("status") == "promoted",
    }
