"""MLflow tracking + Model Registry integration. Versions are always selected explicitly."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from .. import config as C
from ..models.lgbm import GlobalLGBM
from ..uncertainty import QuantileBundle


class ModelUnavailable(RuntimeError):
    pass


def tracking_uri() -> str:
    return (
        os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(C.artifacts_dir() / 'mlflow.db').as_posix()}"
    )


def setup() -> MlflowClient:
    uri = tracking_uri()
    if uri.startswith("sqlite"):
        C.artifacts_dir().mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(uri)


class ModelPackage(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper: input = encoded feature frame, output = point forecast. The full object (encoders, interval
    boosters) is reachable via ``unwrap_python_model()`` for the shared forecasting code."""

    def load_context(self, context):
        d = Path(context.artifacts["package"])
        self.has_intervals = (d / "offsets.json").exists()
        self.bundle = QuantileBundle.load(d) if self.has_intervals else None
        self.point = self.bundle.point if self.bundle else GlobalLGBM.load(d / "point")

    def predict(self, context, model_input, params=None):
        return self.point.predict_matrix(model_input)


def save_package(point: GlobalLGBM, bundle: QuantileBundle | None, d: Path) -> Path:
    if bundle is not None:
        bundle.save(d)
    else:
        point.save(d / "point")
    return d


def log_and_register(
    point: GlobalLGBM,
    bundle: QuantileBundle | None,
    *,
    cfg: dict,
    name: str,
    tags: dict,
    params: dict,
    metrics: dict,
    evidence: dict,
    dataset_manifest: dict | None,
    sample_features: pd.DataFrame,
    run_name: str,
) -> dict:
    client = setup()
    exp = mlflow.set_experiment(cfg["experiment_name"])
    with (
        mlflow.start_run(experiment_id=exp.experiment_id, run_name=run_name) as run,
        tempfile.TemporaryDirectory() as td,
    ):
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        for k, v in metrics.items():
            if v is not None and v == v:
                mlflow.log_metric(k, float(v))
        mlflow.log_dict(evidence, "evidence.json")
        mlflow.log_dict(cfg, "config.json")
        if dataset_manifest:
            mlflow.log_dict(dataset_manifest, "dataset_manifest.json")
        from ..experiments import environment_info

        mlflow.log_dict(environment_info(), "environment.json")
        pkg = save_package(point, bundle, Path(td) / "package")
        X = sample_features[point.feature_cols].head(200)
        sig = infer_signature(X, point.predict_matrix(X))
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ModelPackage(),
            artifacts={"package": str(pkg)},
            signature=sig,
            registered_model_name=name,
            pip_requirements=["lightgbm", "numpy", "pandas"],
        )
        version = info.registered_model_version
        client.set_model_version_tag(name, str(version), "feature_version", tags.get("feature_version", ""))
        client.set_model_version_tag(name, str(version), "dataset_version", tags.get("dataset_version", ""))
        client.set_model_version_tag(name, str(version), "train_origin", str(tags.get("train_origin", "")))
        return {"run_id": run.info.run_id, "model_name": name, "model_version": str(version)}


def load_version(name: str, version: str) -> tuple[GlobalLGBM, QuantileBundle | None, dict]:
    """Load an explicit registry version. Never resolves 'latest'."""
    if not str(version).isdigit():
        raise ModelUnavailable(f"model version must be an explicit integer, got {version!r}")
    client = setup()
    try:
        mv = client.get_model_version(name, str(version))
        m = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
        py = m.unwrap_python_model()
    except Exception as e:
        raise ModelUnavailable(f"cannot load {name} v{version}: {e}") from e
    return py.point, py.bundle, {"tags": dict(mv.tags), "run_id": mv.run_id, "version": str(version)}


def set_champion(name: str, version: str) -> None:
    setup().set_registered_model_alias(name, "champion", str(version))


def champion_version(name: str) -> str | None:
    try:
        return str(setup().get_model_version_by_alias(name, "champion").version)
    except Exception:
        return None


def read_evidence(name: str, version: str) -> dict | None:
    c = setup()
    mv = c.get_model_version(name, str(version))
    try:
        p = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="evidence.json")
        return json.loads(Path(p).read_text())
    except Exception:
        return None
