"""Train -> challenger evidence -> promotion decision -> registry/approval, guarded by an advisory lock.

Evidence protocol: the recipe (feature set + params) is fitted at an evaluation origin ``eo`` and scored on the
matured window (eo, eo+28]; the incumbent is scored on the same window only if its training origin <= eo
(otherwise the window is not out-of-sample for it and the comparison is refused). The final challenger
is then fitted at the latest origin and registered with that evidence attached.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..backtesting import runner as R
from ..features.build import build_examples
from ..features.spec import columns_for
from ..models.baselines import predict_baseline
from ..models.lgbm import GlobalLGBM, feature_groups_for
from ..panel import Panel
from ..uncertainty import QuantileBundle
from ..warehouse import db
from . import mlflow_utils as ML
from .promotion import Evidence, evaluate_promotion


def recipe_fit(
    panel: Panel,
    cfg: dict,
    origin: int,
    key: str,
    params: dict | None,
    sel: np.ndarray,
    with_intervals: bool = True,
    calibrate: bool = True,
) -> tuple[GlobalLGBM, QuantileBundle | None]:
    groups = feature_groups_for(cfg, key)
    point = GlobalLGBM.fit(
        panel, origin, cfg, groups, params=params, series_idx=sel, name="lgbm_point", measure_memory=False
    )
    if not with_intervals:
        return point, None
    outer = [min(cfg["uncertainty"]["quantiles"]), max(cfg["uncertainty"]["quantiles"])]
    bundle = QuantileBundle.fit(panel, origin, cfg, groups, sel, params, point, outer)
    if calibrate and origin - 28 >= cfg["protocol"]["min_history_days"] + 56 and origin <= panel.n_days:
        co = origin - 28  # calibration window (co, origin] strictly earlier
        cal = QuantileBundle.fit(panel, co, cfg, groups, sel, params, point, outer)
        raw = cal.predict_raw(panel, co, sel, list(range(1, 29)))
        y = panel.sales[raw.series_idx.to_numpy(), co + raw.h.to_numpy() - 1].astype(float)
        bundle.calibrate(raw, y, nominal=outer[1] - outer[0])
    return point, bundle


def _score(
    panel: Panel, cfg: dict, o: int, point: GlobalLGBM, bundle: QuantileBundle | None, edges: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Forecast every series at origin ``o`` (information <= o) and attach the realised actuals of (o, o+28]."""
    sel = panel.eligible_at(o)  # products observed by the origin (cold-start products are not scored here)
    horizons = list(range(1, 29))
    t0 = time.perf_counter()
    pf = point.predict(panel, o, sel, horizons)
    batch_seconds = time.perf_counter() - t0
    if bundle is not None:
        iv = bundle.apply(bundle.predict_raw(panel, o, sel, horizons))
        pf = pf.merge(iv[["series_idx", "h", "lower", "upper"]], on=["series_idx", "h"])
    df = R.attach_truth_and_segments(pf, panel, o, cfg, edges)
    sn = R.attach_truth_and_segments(
        predict_baseline("seasonal_naive", panel, o, sel, horizons, cfg), panel, o, cfg, edges
    )
    return df, sn, batch_seconds


def evidence_from_frames(
    dfs: list[pd.DataFrame],
    sns: list[pd.DataFrame],
    origins: list[int],
    batch_seconds: float,
    train_seconds: float | None,
    with_interval: bool,
) -> Evidence:
    """Pool one or several consecutive evaluation windows into promotion evidence.

    Pooled WAPE / bias / coverage are computed over all rows of all windows (bias is an aggregate-demand property whose single-window
    value is noisy: a 28-day window moves by +-20% even for the trailing-mean baseline); per-window biases are kept for transparency."""
    from ..backtesting import metrics as M

    df, sn = pd.concat(dfs, ignore_index=True), pd.concat(sns, ignore_index=True)
    m, sm = M.pooled(df).iloc[0], M.pooled(sn).iloc[0]
    segs = {}
    for st, col in R.SEGMENT_COLS.items():
        if st in ("volume", "intermittency", "store"):
            for v, r in M.pooled(df, [col]).iterrows():
                segs[f"{st}:{v}"] = None if pd.isna(r["wape"]) else float(r["wape"])
    cov = float(((df.actual >= df.lower) & (df.actual <= df.upper)).mean()) if with_interval else None
    return Evidence(
        wape=float(m["wape"]),
        snaive_wape=float(sm["wape"]),
        bias_pct=float(m["bias_pct"]),
        coverage=cov,
        segment_wape={k: v for k, v in segs.items() if v is not None},
        train_seconds=train_seconds,
        batch_seconds=batch_seconds,
        eval_origin=max(origins),
        n_obs=int(len(df)),
        eval_origins=list(origins),
        window_bias_pct=[float(M.pooled(d).iloc[0]["bias_pct"]) for d in dfs],
    )


def build_evidence(
    panel: Panel,
    cfg: dict,
    eo: int,
    point: GlobalLGBM,
    bundle: QuantileBundle | None,
    edges: list[float],
    train_seconds: float | None,
) -> Evidence:
    df, sn, bs = _score(panel, cfg, eo, point, bundle, edges)
    return evidence_from_frames([df], [sn], [eo], bs, train_seconds, bundle is not None)


def build_evidence_windows(
    panel: Panel,
    cfg: dict,
    origins: list[int],
    fit_fn,
    edges: list[float],
    train_seconds: float | None = None,
) -> Evidence:
    """Challenger evidence over several windows: ``fit_fn(o)`` must return a (point, bundle) pair fitted with data <= o."""
    dfs, sns, bs, with_iv = [], [], 0.0, False
    for o in origins:
        point, bundle = fit_fn(o)
        d, s, b = _score(panel, cfg, o, point, bundle, edges)
        dfs.append(d)
        sns.append(s)
        bs += b
        with_iv = with_iv or bundle is not None
    return evidence_from_frames(dfs, sns, origins, bs / len(origins), train_seconds, with_iv)


def incumbent_evidence(
    panel: Panel, cfg: dict, origins: list[int] | int, name: str, edges: list[float]
) -> tuple[Evidence | None, str | None]:
    """Score the champion on the same windows. Refused (None) unless it was trained with data <= the earliest window origin."""
    origins = [origins] if isinstance(origins, int) else list(origins)
    ver = ML.champion_version(name)
    if ver is None:
        return None, None
    point, bundle, info = ML.load_version(name, ver)
    tro = int(info["tags"].get("train_origin") or 10**9)
    if tro > min(origins):
        return None, ver  # not out-of-sample for the incumbent on these windows
    dfs, sns, bs = [], [], 0.0
    for o in origins:
        d, s, b = _score(panel, cfg, o, point, bundle, edges)
        dfs.append(d)
        sns.append(s)
        bs += b
    return evidence_from_frames(dfs, sns, origins, bs / len(origins), None, bundle is not None), ver


def train_evaluate_promote(
    panel: Panel,
    cfg: dict,
    *,
    feature_key: str,
    params: dict | None,
    dsn: str | None = None,
    train_origin: int | None = None,
    dataset_manifest: dict | None = None,
    with_intervals: bool = True,
) -> dict:
    """Full challenger lifecycle. Returns decision record. Mutually exclusive with other training/promotion runs."""
    name = cfg["serving"]["model_name"]
    origin = train_origin or panel.n_days
    sel = (
        np.arange(panel.n_series)
        if not cfg["training"]["max_series"]
        else np.sort(
            np.random.default_rng(cfg["seed"]).choice(
                panel.n_series, min(cfg["training"]["max_series"], panel.n_series), replace=False
            )
        )
    )
    with db.connect(dsn) as conn, db.advisory_lock(conn, db.LOCK_TRAINING) as got:
        if not got:
            return {"status": "skipped_concurrent", "detail": "another training/promotion run holds the lock"}
        inc_ver = ML.champion_version(name)
        inc_origin = None
        if inc_ver:
            inc_origin = (
                int(ML.setup().get_model_version(name, inc_ver).tags.get("train_origin") or 0) or None
            )
        eo = max(origin - 28, inc_origin or 0)
        k_windows = int(cfg["promotion"].get("evidence_windows", 1))
        ev_origins = [eo - 28 * k for k in range(k_windows - 1, -1, -1)]  # consecutive windows, ascending
        edges = R.edges_for(panel, eo, cfg)  # tertile edges from data <= eo
        if eo + 28 > panel.n_days or min(ev_origins) < cfg["protocol"]["min_history_days"] + 56:
            return {
                "status": "insufficient_evaluation_data",
                "eval_origins": ev_origins,
                "n_days": panel.n_days,
            }
        t0 = time.perf_counter()
        cev = build_evidence_windows(
            panel,
            cfg,
            ev_origins,
            lambda o: recipe_fit(panel, cfg, o, feature_key, params, sel, with_intervals),
            edges,
            None,
        )
        cev.train_seconds = (time.perf_counter() - t0) / k_windows  # mean fit+score time per evidence window
        iev, iv = incumbent_evidence(panel, cfg, ev_origins, name, edges)
        if inc_ver and iev is None:
            decision = {"promote": False, "failed": ["incumbent_not_comparable"], "checks": []}
        else:
            decision = evaluate_promotion(cev, iev, cfg["promotion"], require_coverage=with_intervals)
        t0 = time.perf_counter()
        point, bundle = recipe_fit(panel, cfg, origin, feature_key, params, sel, with_intervals)
        final_train = time.perf_counter() - t0
        cols = columns_for(point.groups, point.price_scenario)
        X = build_examples(
            panel,
            origin,
            sel[:50],
            [1, 2, 3],
            point.groups,
            point.enc,
            with_target=False,
            price_scenario=point.price_scenario,
        )
        tags = {
            "feature_version": point.feature_version,
            "dataset_version": panel.dataset_version,
            "train_origin": origin,
            "feature_set": feature_key,
            "forecast_strategy": "direct_global_multi_horizon",
            "evidence_eval_origin": eo,
            "evidence_windows": k_windows,
            "n_features": len(cols),
        }
        evidence = {
            "challenger": cev.to_dict(),
            "incumbent": None if iev is None else iev.to_dict(),
            "incumbent_version": iv,
            "decision": decision,
            "final_train_seconds": final_train,
        }
        reg = ML.log_and_register(
            point,
            bundle,
            cfg=cfg,
            name=name,
            tags=tags,
            params={**point.params, "feature_set": feature_key},
            evidence=evidence,
            metrics={
                "val_wape": cev.wape,
                "val_snaive_wape": cev.snaive_wape,
                "val_bias_pct": cev.bias_pct,
                "val_coverage": cev.coverage,
                "train_seconds": final_train,
            },
            dataset_manifest=dataset_manifest,
            sample_features=X,
            run_name=f"train@{origin}",
        )
        out = {"status": "registered", **reg, "decision": decision, "evidence": evidence}
        if decision["promote"]:
            promote(conn, name, reg["model_version"], panel.dataset_version, evidence)
            out["status"] = "promoted"
        return out


def rollback(conn, name: str, version: str, reason: str, dsn: str | None = None) -> dict:
    """Re-approve an earlier registered version (audited, lock-protected). Forecast batches are immutable, so rolling
    back changes which version's batch the API serves; it never edits stored forecasts."""
    if not reason.strip():
        raise ValueError("a rollback reason is required for the audit trail")
    with db.advisory_lock(conn, db.LOCK_TRAINING) as got:
        if not got:
            return {"status": "skipped_concurrent"}
        mv = ML.setup().get_model_version(name, str(version))  # raises if the version does not exist
        previous = ML.champion_version(name)
        if previous == str(version):
            return {"status": "noop", "version": str(version)}
        promote(
            conn,
            name,
            str(version),
            mv.tags.get("dataset_version", ""),
            {"rollback_from": previous, "reason": reason},
            approved_by="manual:rollback",
        )
        return {"status": "rolled_back", "from": previous, "to": str(version), "reason": reason}


def promote(
    conn,
    name: str,
    version: str,
    dataset_version: str,
    evidence: dict,
    approved_by: str = "rule:promotion_v1",
) -> None:
    ML.set_champion(name, version)
    with conn.transaction():
        conn.execute(
            "UPDATE app.approved_models SET is_current=FALSE WHERE model_name=%s AND is_current", (name,)
        )
        conn.execute(
            "INSERT INTO app.approved_models (model_name, model_version, dataset_version, approved_by, evidence) "
            "VALUES (%s,%s,%s,%s,%s::jsonb)",
            (name, version, dataset_version, approved_by, json.dumps(evidence, default=str)),
        )
