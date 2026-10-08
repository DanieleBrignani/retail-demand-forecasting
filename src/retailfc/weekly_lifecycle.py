"""Lifecycle of the weekly line: registration, rule-based approval, batch generation, revocation.

Separate concepts (docs/weekly_protocol_v2.md section 7):
* **registration**   -- a versioned artefact in the MLflow registry (``register_selected``);
* **approval**       -- ``promote`` with the evidence of ``selection.json`` (approved_by ``rule:weekly_v2``); never implied by registering;
* **historical evaluation** -- the backtest in ``retailfc.weekly`` (not repeated here);
* **operational deployment** -- ``generate_weekly_batch`` stores forecasts with exact versions, ``horizon_unit='week'`` and explicit periods.

A selected *baseline* is a first-class registry entry: the registered artefact is its specification (family, name, parameters, protocol and
config hashes). Inference for it runs through ``retailfc.weekly.baseline_forecasts``. Registering the LightGBM candidate for serving is not
implemented (it was not selected); asking for it fails loudly instead of pretending.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from . import weekly as W
from .forecasting import store
from .panel import Panel
from .registry import mlflow_utils as ML

WEEKLY_MODEL_NAME = "retailfc_weekly"
EXPERIMENT = "retailfc_weekly"
DAYS_PER_BLOCK = 7


class WeeklySpecPackage(mlflow.pyfunc.PythonModel):
    """Carrier of the specification. ``predict`` supports only the history-window baselines (columns lag1..lag13, lag1 = latest
    block); other baselines need the full history and run through ``retailfc.weekly``."""

    def load_context(self, context):
        self.spec = json.loads(Path(context.artifacts["spec"]).read_text())

    def predict(self, context, model_input, params=None):
        name = self.spec["name"]
        lags = model_input[[f"lag{k}" for k in range(1, 14)]].to_numpy(float)
        if name == "naive":
            f = lags[:, 0]
        elif name == "mean4":
            f = lags[:, :4].mean(axis=1)
        elif name == "mean13":
            f = lags.mean(axis=1)
        else:
            raise NotImplementedError(
                f"{name} needs the full history; use retailfc.weekly.baseline_forecasts"
            )
        return np.repeat(f[:, None], len(self.spec["horizons"]), axis=1)


def make_spec(selection: dict, manifest: dict, cfg: dict, evidence: dict | None = None) -> dict:
    name = selection.get("selected")
    if name is None:
        raise ValueError("selection.json has no selected model: nothing to register (the rejection stands)")
    if name == "lgbm_w":
        raise NotImplementedError(
            "serving the weekly LightGBM is not implemented; only baselines can be registered"
        )
    if name not in cfg["models"]["baselines"]:
        raise ValueError(f"{name!r} is not a registered-able model")
    return {
        "granularity": "week",
        "horizon_unit": "week",
        "horizons": W.horizons(cfg),
        "family": "baseline",
        "name": name,
        "parameters": {k: cfg["models"].get(k) for k in ("croston", "tsb") if name in (k, "croston_sba")},
        "trained": False,
        "train_origin": None,
        "protocol_version": manifest["protocol_version"],
        "config_sha256": manifest["config_sha256"],
        "config_file": manifest.get("config_path"),
        "dataset_version": manifest["dataset_version"],
        "weekly_feature_version": manifest["weekly_feature_version"],
        "calendar": manifest["calendar"],
        "evidence": evidence
        or {},  # SHA-256 of the selection.json / manifest.json that justified this registration
    }


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


class IncompatibleConfig(RuntimeError):
    """The configuration offered at generation time is not the one the registered model was evaluated with."""


def register_selected(exp_dir: str | Path, cfg_path: str | Path | None = None) -> dict:
    """REGISTRATION only: creates a registry version of the selected model; approval is a separate step."""
    exp = Path(exp_dir)
    selection = json.loads((exp / "selection.json").read_text())
    manifest = json.loads((exp / "manifest.json").read_text())
    cfg_file = cfg_path or manifest["config_path"]
    if W.cfg_hash(cfg_file) != manifest["config_sha256"]:
        raise IncompatibleConfig(
            f"{cfg_file} does not hash to the config recorded in the manifest; refusing to register"
        )
    cfg = W.load_cfg(cfg_file)
    spec = make_spec(
        selection,
        manifest,
        cfg,
        {"selection_sha256": _sha(exp / "selection.json"), "manifest_sha256": _sha(exp / "manifest.json")},
    )
    client = ML.setup()
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"register-{spec['name']}") as run, tempfile.TemporaryDirectory() as td:
        mlflow.set_tags(
            {"granularity": "week", "protocol_version": spec["protocol_version"], "family": "baseline"}
        )
        mlflow.log_params(
            {
                "model": spec["name"],
                "horizons": str(spec["horizons"]),
                "config_sha256": spec["config_sha256"][:16],
            }
        )
        mlflow.log_dict(spec, "spec.json")
        mlflow.log_dict(selection, "selection.json")
        mlflow.log_dict(manifest, "manifest.json")
        p = Path(td) / "spec.json"
        p.write_text(json.dumps(spec))
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=WeeklySpecPackage(),
            artifacts={"spec": str(p)},
            registered_model_name=WEEKLY_MODEL_NAME,
            pip_requirements=["numpy", "pandas"],
        )
        version = str(info.registered_model_version)
        for k, v in {
            "granularity": "week",
            "dataset_version": spec["dataset_version"],
            "feature_version": spec["weekly_feature_version"],
            "train_origin": "",
            "protocol_version": spec["protocol_version"],
        }.items():
            client.set_model_version_tag(WEEKLY_MODEL_NAME, version, k, v)
        return {
            "run_id": run.info.run_id,
            "model_name": WEEKLY_MODEL_NAME,
            "model_version": version,
            "spec": spec["name"],
        }


def load_spec(name: str, version: str) -> dict:
    if not str(version).isdigit():
        raise ML.ModelUnavailable(f"model version must be an explicit integer, got {version!r}")
    client = ML.setup()
    try:
        mv = client.get_model_version(name, str(version))
        p = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="spec.json")
        return json.loads(Path(p).read_text())
    except Exception as e:
        raise ML.ModelUnavailable(f"cannot load {name} v{version}: {e}") from e


def approve(conn, version: str, exp_dir: str | Path) -> dict:
    """APPROVAL by rule. Refused unless the evidence offered is exactly the evidence the registered version was created from:
    the selection must name this model, and the selection/manifest files must hash to the values recorded at registration,
    and the manifest's dataset version, protocol version and config hash must equal the registered specification's."""
    from .registry.pipeline import promote

    exp = Path(exp_dir)
    selection = json.loads((exp / "selection.json").read_text())
    manifest = json.loads((exp / "manifest.json").read_text())
    spec = load_spec(WEEKLY_MODEL_NAME, version)
    problems = []
    if selection.get("selected") != spec["name"]:
        problems.append(f"selection record names {selection.get('selected')!r}, not {spec['name']!r}")
    for k in ("dataset_version", "protocol_version", "config_sha256"):
        if manifest.get(k) != spec.get(k):
            problems.append(f"{k}: manifest {manifest.get(k)!r} != registered {spec.get(k)!r}")
    if selection.get("protocol") != spec["protocol_version"]:
        problems.append(
            f"selection protocol {selection.get('protocol')!r} != registered {spec['protocol_version']!r}"
        )
    rec = spec.get("evidence") or {}
    if not rec:
        problems.append("registered version carries no evidence hashes")
    for name, file in (("selection_sha256", "selection.json"), ("manifest_sha256", "manifest.json")):
        if rec.get(name) != _sha(exp / file):
            problems.append(f"{file} differs from the file registered as evidence")
    if problems:
        return {"status": "refused", "reason": "; ".join(problems)}
    promote(
        conn,
        WEEKLY_MODEL_NAME,
        str(version),
        spec["dataset_version"],
        {"selection": selection, "evidence_hashes": rec, "config_sha256": spec["config_sha256"]},
        approved_by="rule:weekly_v2",
    )
    return {"status": "approved", "model_version": str(version), "model": spec["name"]}


def revoke(conn, name: str, version: str, reason: str) -> dict:
    """Withdraw an approval (kept in the audit trail with a reason). The API then answers 503 model_unavailable for that line."""
    if not reason.strip():
        raise ValueError("a revocation reason is required")
    with conn.transaction():
        n = conn.execute(
            "UPDATE app.approved_models SET is_current=FALSE, revoked_at=now(), revoke_reason=%s "
            "WHERE model_name=%s AND model_version=%s AND is_current",
            (reason, name, str(version)),
        ).rowcount
    if n:
        try:
            ML.setup().delete_registered_model_alias(name, "champion")
        except Exception:  # alias may not exist in this registry
            pass
    return {"status": "revoked" if n else "noop", "model_name": name, "model_version": str(version)}


def block_dates(panel: Panel, t: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last calendar date of block t (1-based)."""
    cal = pd.to_datetime(panel.calendar["date"])
    return cal.iloc[(t - 1) * DAYS_PER_BLOCK], cal.iloc[t * DAYS_PER_BLOCK - 1]


def weekly_records(
    panel: Panel, spec: dict, Y: np.ndarray, pidx: np.ndarray, ids: pd.DataFrame, t: int, cfg: dict
) -> pd.DataFrame:
    """Forecast rows for every series eligible at block origin ``t`` (information: blocks <= t only)."""
    hz = W.horizons(cfg)
    rows = np.flatnonzero(W.eligible_mask(Y, pidx, t))
    f = W.baseline_forecasts(Y, t, cfg)[spec["name"]][rows]
    no_hist = ~(Y[rows, :t] > 0).any(axis=1)
    out = []
    for j, h in enumerate(hz):
        s, e = block_dates(panel, t + h)
        out.append(
            pd.DataFrame(
                {
                    "item_id": ids["item_id"].to_numpy()[rows],
                    "store_id": ids["store_id"].to_numpy()[rows],
                    "target_date": s.strftime("%Y-%m-%d"),
                    "horizon": np.int16(h),
                    "horizon_unit": "week",
                    "target_period_start": s.strftime("%Y-%m-%d"),
                    "target_period_end": e.strftime("%Y-%m-%d"),
                    "forecast_value": np.maximum(f[:, j].astype(np.float64), 0.0),
                    "lower_bound": np.nan,  # intervals stay experimental until the config thresholds hold (protocol section 6)
                    "upper_bound": np.nan,
                    "fallback_used": np.where(no_hist, "no_channel_history", ""),
                }
            )
        )
    return (
        pd.concat(out, ignore_index=True)
        .sort_values(["item_id", "store_id", "target_date"])
        .reset_index(drop=True)
    )


def generate_weekly_batch(
    conn,
    panel: Panel,
    model_version: str,
    *,
    origin_block: int | None = None,
    mode: str = "production",
    run_label: str = "default",
    cfg_path: str | Path | None = None,
    allow_dataset_change: bool = False,
) -> dict:
    """Store the 4-block forecasts of an explicit registry version. Production batches use the latest complete block;
    earlier origins are HISTORICAL REPLAY. Batches are append-only (see ``store.write_batch``)."""
    spec = load_spec(WEEKLY_MODEL_NAME, model_version)
    if cfg_path is None and spec.get("config_file") and Path(spec["config_file"]).exists():
        cfg_path = spec[
            "config_file"
        ]  # the file the version was registered with; the hash below is the real guard
    cfg = W.load_cfg(cfg_path)
    if W.cfg_hash(cfg_path) != spec["config_sha256"]:
        raise IncompatibleConfig(
            "the configuration file does not match the one this model version was registered with "
            f"(SHA-256 {W.cfg_hash(cfg_path)[:12]}… != {spec['config_sha256'][:12]}…); refusing to generate"
        )
    if panel.dataset_version != spec["dataset_version"] and not allow_dataset_change:
        raise IncompatibleConfig(
            f"model v{model_version} was evaluated on dataset {spec['dataset_version']}, not {panel.dataset_version}; "
            "re-run the protocol on the new data or pass allow_dataset_change=True explicitly"
        )
    Y, ids, _starts, pidx = W.to_weekly(panel)
    last = Y.shape[1]
    t = origin_block or last
    if t > last:
        raise ValueError(f"origin block {t} beyond the {last} complete blocks")
    if mode == "production" and t < last:
        raise ValueError(
            "production batches must use the latest complete block; use mode='historical_replay' for earlier origins"
        )
    if spec.get("train_origin") and int(spec["train_origin"]) > t:
        raise ValueError("model trained after the forecast origin")
    t0 = time.perf_counter()
    rec = weekly_records(panel, spec, Y, pidx, ids, t, cfg)
    infer_s = time.perf_counter() - t0
    _s, origin_date = block_dates(panel, t)
    origin_str = str(origin_date.date())
    res = store.write_batch(
        conn,
        rec,
        model_name=WEEKLY_MODEL_NAME,
        model_version=str(model_version),
        dataset_version=panel.dataset_version,
        feature_version=spec["weekly_feature_version"],
        origin_date=origin_str,
        origin_day=t * DAYS_PER_BLOCK,
        mode=mode,
        run_label=run_label,
        generation_time=datetime.now(UTC),
        data_available_at=datetime.now(UTC),
        granularity="week",
        protocol_version=spec["protocol_version"],
    )
    conn.commit()
    return {
        **res,
        "granularity": "week",
        "origin_block": t,
        "origin_date": origin_str,
        "model": spec["name"],
        "inference_seconds": infer_s,
        "fallback_share": float((rec["fallback_used"] != "").mean()),
        "n_series": int(rec[["item_id", "store_id"]].drop_duplicates().shape[0]),
    }
