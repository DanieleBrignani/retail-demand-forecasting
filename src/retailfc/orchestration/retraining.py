"""Retraining decision logic. A scheduled check may (and usually should) conclude "no retraining".

Gates, evaluated in order (first terminal outcome wins; every outcome is audited in app.retraining_decisions):
  1. concurrency      : another training/promotion run holds the lock       -> skipped_locked
  2. data quality     : too many missing actuals / conflicting actuals       -> blocked_data_quality
  3. cooldown         : trained less than cooldown_days ago                  -> no_retrain (cooldown)
  4. triggers         : performance degradation on comparable matured forecasts, governance age, persistent drift
  5. minimum new data : too few new actual days since the model's training origin -> no_retrain (insufficient_new_data)
Drift alone never triggers retraining: it yields ``investigate``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Decision:
    decision: str
    reasons: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"decision": self.decision, "reasons": self.reasons, "triggers": self.triggers}


def decide(inputs: dict, policy: dict) -> Decision:
    r: list[str] = []
    if inputs.get("lock_held"):
        return Decision("skipped_locked", ["training/promotion already running"])
    mf = inputs.get("missing_actuals_fraction")
    if mf is not None and mf > policy["max_missing_actuals_fraction"]:
        return Decision(
            "blocked_data_quality",
            [f"missing actuals fraction {mf:.3f} > {policy['max_missing_actuals_fraction']}"],
        )
    if inputs.get("actuals_conflicts", 0) > 0:
        return Decision(
            "blocked_data_quality", [f"{inputs['actuals_conflicts']} conflicting actuals need resolution"]
        )
    fm = inputs.get("feature_missing_max")
    if fm is not None and fm > policy["max_feature_missing_fraction"]:
        return Decision(
            "blocked_data_quality",
            [f"feature missingness {fm:.3f} > {policy['max_feature_missing_fraction']}"],
        )
    dsl = inputs.get("days_since_last_training")
    if dsl is not None and dsl < policy["cooldown_days"]:
        return Decision("no_retrain", [f"cooldown: trained {dsl}d ago < {policy['cooldown_days']}d"])

    triggers: list[str] = []
    n_m, cur, ref = inputs.get("matured_obs", 0), inputs.get("current_wape"), inputs.get("reference_wape")
    if n_m < policy["min_matured_forecast_obs"] or cur is None or ref is None or not ref:
        r.append(
            f"performance not assessable: {n_m} matured observations (min {policy['min_matured_forecast_obs']})"
        )
    else:
        deg = (cur - ref) / ref * 100
        if deg >= policy["wape_degradation_pct"]:
            triggers.append(f"performance_degradation:{deg:.1f}%")
        else:
            r.append(f"performance ok: WAPE change {deg:+.1f}% (< {policy['wape_degradation_pct']}%)")
    age = inputs.get("model_age_days")
    if age is not None and age >= policy["max_model_age_days"]:
        triggers.append(f"governance_age:{age}d")
    psis = [p for p in inputs.get("recent_drift_psi", []) if p is not None and p == p]
    k = policy["drift_persistence_checks"]
    drift_persistent = len(psis) >= k and all(p >= policy["drift_psi_threshold"] for p in psis[-k:])
    if drift_persistent:
        r.append(
            f"persistent drift (PSI>={policy['drift_psi_threshold']} for {k} checks): investigate; drift alone is not accuracy loss"
        )

    if not triggers:
        return Decision("investigate" if drift_persistent else "no_retrain", r)
    nd = inputs.get("new_actual_days", 0)
    if nd < policy["min_new_actual_days"]:
        return Decision(
            "no_retrain",
            [*r, f"insufficient new data: {nd} new actual days < {policy['min_new_actual_days']}"],
            triggers,
        )
    return Decision("retrain", r, triggers)


def gather_inputs(conn, cfg: dict, as_of: date, panel_dates=None) -> dict:
    from ..monitoring import core
    from ..warehouse import db

    pol = cfg["retraining"]
    name = cfg["serving"]["model_name"]
    cur = conn.execute(
        "SELECT model_version, approved_at, evidence FROM app.approved_models WHERE model_name=%s AND is_current",
        (name,),
    ).fetchone()
    inputs: dict = {}
    with db.advisory_lock(conn, db.LOCK_TRAINING) as got:
        inputs["lock_held"] = not got
    cov = core.actuals_coverage(conn, as_of)
    inputs["missing_actuals_fraction"] = cov["missing_actuals_fraction"]
    inputs["actuals_conflicts"] = int(
        conn.execute("SELECT count(*) FROM app.actuals_conflicts").fetchone()[0]
    )
    fm = conn.execute(
        "SELECT max(value) FROM app.monitoring_snapshots WHERE kind='data_quality' AND name LIKE 'feature_missing:%%' "
        "AND taken_at = (SELECT max(taken_at) FROM app.monitoring_snapshots WHERE name LIKE 'feature_missing:%%')"
    ).fetchone()[0]
    inputs["feature_missing_max"] = None if fm is None else float(fm)
    if cur:
        ver, approved_at, ev = cur
        inputs["model_version"] = ver
        inputs["days_since_last_training"] = (
            (datetime.now(approved_at.tzinfo) - approved_at).days if approved_at else None
        )
        inputs["model_age_days"] = inputs["days_since_last_training"]
        ch = (ev or {}).get("challenger") or {}
        inputs["reference_wape"] = ch.get("wape")
        perf = conn.execute(
            "SELECT sum(n), sum(sum_abs_error)/NULLIF(sum(sum_actual),0) FROM app.monitoring_aggregates "
            "WHERE model_version=%s",
            (ver,),
        ).fetchone()
        inputs["matured_obs"] = int(perf[0] or 0)
        inputs["current_wape"] = None if perf[1] is None else float(perf[1])
        last_train_date = conn.execute(
            "SELECT max(forecast_origin) FROM app.forecast_batches WHERE model_version=%s", (ver,)
        ).fetchone()[0]
        inputs["new_actual_days"] = int(
            conn.execute(
                "SELECT count(DISTINCT date) FROM app.actuals WHERE date > COALESCE(%s, date '1900-01-01')",
                (last_train_date,),
            ).fetchone()[0]
        )
    rows = conn.execute(
        "SELECT value FROM app.monitoring_snapshots WHERE kind='drift' AND name='demand_psi' "
        "ORDER BY taken_at DESC, id DESC LIMIT %s",
        (pol["drift_persistence_checks"],),
    ).fetchall()
    inputs["recent_drift_psi"] = [r[0] for r in reversed(rows)]
    return inputs


def record_decision(
    conn,
    decision: Decision,
    inputs: dict,
    policy: dict,
    challenger_version: str | None = None,
    promotion: dict | None = None,
) -> int:
    did = conn.execute(
        "INSERT INTO app.retraining_decisions (decision, reasons, inputs, policy, challenger_version, promotion_decision) "
        "VALUES (%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s::jsonb) RETURNING decision_id",
        (
            decision.decision,
            json.dumps(decision.to_dict()),
            json.dumps(inputs, default=str),
            json.dumps(policy),
            challenger_version,
            None if promotion is None else json.dumps(promotion, default=str),
        ),
    ).fetchone()[0]
    conn.commit()
    return did
