"""Configuration loading: packaged YAML + environment-driven paths."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import yaml

_DEFAULT = Path(__file__).with_name("experiment.yaml")

# Config sections whose content defines the scientific protocol (hashed for the lock).
PROTOCOL_SECTIONS = (
    "selection",
    "horizon",
    "protocol",
    "training",
    "features",
    "lightgbm",
    "statistical",
    "baselines",
    "segments",
    "uncertainty",
    "promotion",
    "seed",
)


def home() -> Path:
    return Path(os.environ.get("RETAILFC_HOME", Path.cwd()))


def data_dir() -> Path:
    return Path(os.environ.get("RETAILFC_DATA_DIR", home() / "data"))


def artifacts_dir() -> Path:
    return Path(os.environ.get("RETAILFC_ARTIFACTS_DIR", home() / "artifacts"))


def dsn() -> str:
    return os.environ.get("RETAILFC_DSN", "postgresql://retailfc:retailfc@localhost:5442/retailfc")


def _merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load YAML. A file may declare ``extends: default`` and then only list overrides (deep-merged)."""
    p = Path(path or os.environ.get("RETAILFC_CONFIG", _DEFAULT))
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("extends") == "default":
        with open(_DEFAULT, encoding="utf-8") as f:
            base = yaml.safe_load(f)
        cfg = _merge(base, {k: v for k, v in cfg.items() if k != "extends"})
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable hash over the scientific-protocol sections only."""
    sub = {k: cfg.get(k) for k in PROTOCOL_SECTIONS}
    blob = json.dumps(sub, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def override(cfg: dict, **kv) -> dict:
    """Return a deep copy with dotted-key overrides, e.g. override(cfg, **{'training.max_series': 50})."""
    out = deepcopy(cfg)
    for k, v in kv.items():
        cur = out
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = v
    return out
