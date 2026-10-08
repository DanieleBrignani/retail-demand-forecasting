"""Temporal protocol: forecast origins, training origins, holdout lock.

Conventions (1-based day numbers ``d_1 .. d_N``):
* ``origin`` o = last day whose sales are available to the forecasting process.
* targets are days o+1 .. o+H.
* ``sales[:, :o]`` is the entire information set about sales at origin o.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import config as C


@dataclass(frozen=True)
class Origins:
    n_days: int
    validation: tuple[int, ...]  # chronological, non-overlapping windows
    holdout: int  # origin of the final holdout window

    def as_dict(self) -> dict:
        return {"n_days": self.n_days, "validation": list(self.validation), "holdout": self.holdout}


def make_origins(n_days: int, cfg: dict) -> Origins:
    """Validation windows end exactly where the holdout window's *origin* is; holdout is the last window."""
    p = cfg["protocol"]
    w = p["window_days"]
    holdout = n_days - p["final_holdout_days"]
    k = p["n_validation_windows"]
    val = tuple(holdout - w * i for i in range(k, 0, -1))
    if val[0] < p["min_history_days"] + 2 * w:
        raise ValueError(f"not enough history ({n_days} days) for {k} validation windows + holdout")
    return Origins(n_days=n_days, validation=val, holdout=holdout)


def window_targets(origin: int, horizon: int) -> range:
    return range(origin + 1, origin + horizon + 1)


def check_non_overlap(origins: Origins, horizon: int) -> None:
    seq = (*origins.validation, origins.holdout)
    for a, b in zip(seq, seq[1:], strict=False):
        if b - a < horizon:
            raise AssertionError(f"windows with origins {a} and {b} overlap (horizon={horizon})")
    if origins.holdout + horizon > origins.n_days:
        raise AssertionError("holdout window exceeds observed data")


def training_origins(backtest_origin: int, cfg: dict, *, gap: int | None = None) -> list[int]:
    """Training origins o' such that o' + H <= backtest_origin - gap_offset (all targets observed).

    ``gap`` (default 0) additionally shifts the latest allowed origin back; used to carve an
    early-stopping validation origin (origin-28) that no training target window overlaps.
    """
    h = cfg["horizon"]
    t = cfg["training"]
    gap = 0 if gap is None else gap
    latest = backtest_origin - h - gap
    stride = t["train_origin_stride_days"]
    lowest = max(cfg["protocol"]["min_history_days"], latest - t["train_window_days"])
    out = []
    o = latest
    while o >= lowest:
        out.append(o)
        o -= stride
    return sorted(out)


# ------------------------------------------------------------------ protocol lock
def lock_path() -> Path:
    return C.artifacts_dir() / "protocol_lock.json"


def holdout_marker_path() -> Path:
    return C.artifacts_dir() / "holdout_evaluated.json"


def lock_protocol(cfg: dict, dataset_version: str, n_days: int) -> dict:
    origins = make_origins(n_days, cfg)
    rec = {
        "config_hash": C.config_hash(cfg),
        "dataset_version": dataset_version,
        "locked_at": datetime.now(UTC).isoformat(),
        "origins": origins.as_dict(),
    }
    p = lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = json.loads(p.read_text())
        if old["config_hash"] == rec["config_hash"] and old["dataset_version"] == dataset_version:
            return old
        raise RuntimeError(
            "Protocol already locked with a different config/dataset. Create a new artifacts "
            "dir (new experiment) -- redesign after seeing results requires an untouched period."
        )
    p.write_text(json.dumps(rec, indent=2))
    return rec


def assert_protocol_locked(cfg: dict, dataset_version: str) -> dict:
    p = lock_path()
    if not p.exists():
        raise RuntimeError("Protocol is not locked: run `retailfc protocol lock` before holdout evaluation.")
    rec = json.loads(p.read_text())
    if rec["config_hash"] != C.config_hash(cfg) or rec["dataset_version"] != dataset_version:
        raise RuntimeError("Config or dataset changed after the protocol lock; holdout evaluation refused.")
    return rec


def record_holdout_evaluation(info: dict, *, allow_repeat: bool = False) -> None:
    p = holdout_marker_path()
    if p.exists() and not allow_repeat:
        raise RuntimeError(
            "Final holdout already evaluated once. Re-evaluating after any redesign is not "
            "allowed on the same holdout; a new untouched period is required."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**info, "evaluated_at": datetime.now(UTC).isoformat()}, indent=2))
