"""Immutable, checksummed raw-file versions.

``ingest_raw(src_dir)`` validates the canonical CSVs (``calendar.csv``, ``sales_daily.csv``, ``sell_prices.csv``) and copies them to
``data/raw/<dataset_version>/`` together with ``manifest.json``.  The version id is derived from the
SHA-256 of the file contents, so:
* identical bytes -> identical version (re-ingest is a verified no-op),
* different bytes -> a *new* version directory (old versions are never touched),
* a stored version whose bytes no longer match its manifest is reported as corrupt (never overwritten).
The canonical CSVs are produced from the source data by an adapter (``retailfc adapt-online-retail`` for UCI Online Retail II);
nothing here downloads data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .. import config as C
from .validate import SCHEMA_VERSION, ValidationReport, price_coverage, validate_all

SOURCE_NOTE = (
    "UCI Online Retail II (https://archive.ics.uci.edu/dataset/502/online+retail+ii, CC BY 4.0), "
    "adapted by retailfc.adapters.online_retail."
)
FILES = {"calendar": "calendar.csv", "prices": "sell_prices.csv", "sales": "sales_daily.csv"}


class RawIngestError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def dataset_version_from(checksums: dict[str, str], label: str = "retail") -> str:
    blob = "\n".join(f"{k}:{v}" for k, v in sorted(checksums.items())).encode()
    return f"{label}-" + hashlib.sha256(blob).hexdigest()[:12]


def raw_root() -> Path:
    return C.data_dir() / "raw"


def _make_readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def _rmtree(path: Path) -> None:
    """Remove a (possibly read-only) directory tree; Windows refuses to delete read-only files."""

    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
            func(p)
        except OSError:
            pass

    if Path(path).exists():
        shutil.rmtree(path, onerror=_onerror)


def _publish(tmp: Path, dest: Path, attempts: int = 8, base_delay: float = 0.4) -> None:
    """Atomic directory rename with retry: on Windows a sync client or scanner can hold handles on a freshly
    written folder for a moment and make the rename fail with PermissionError. Never falls back to a copy,
    so a version directory either appears complete or not at all."""
    last: OSError | None = None
    for i in range(attempts):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError as e:
            last = e
            time.sleep(base_delay * (i + 1))
    raise RawIngestError(
        f"could not publish {dest.name}: the folder stayed locked after {attempts} attempts ({last}). "
        "Keep data/ outside synced folders (OneDrive/Dropbox) or pause the sync client, then re-run."
    )


def verify_version(version: str) -> dict:
    """Re-hash a stored version and compare with its manifest. Raises on any mismatch."""
    d = raw_root() / version
    mf = json.loads((d / "manifest.json").read_text())
    for name, meta in mf["files"].items():
        if sha256_file(d / name) != meta["sha256"]:
            raise RawIngestError(
                f"{version}/{name} bytes differ from manifest checksum (corrupt or tampered)"
            )
    return mf


def ingest_raw(
    src_dir: str | Path,
    *,
    source_note: str = SOURCE_NOTE,
    skip_validation: bool = False,
    label: str = "retail",
) -> dict:
    src = Path(src_dir)
    names = dict(FILES)
    missing = [n for n in names.values() if not (src / n).exists()]
    if missing:
        raise RawIngestError(
            f"Missing files in {src}: {missing}. Create them from the source data with "
            "`retailfc adapt-online-retail --xlsx <online_retail_II.xlsx> --out <folder>` and point --src at that folder."
        )

    checksums = {n: sha256_file(src / n) for n in names.values()}
    version = dataset_version_from(checksums, label)
    dest = raw_root() / version

    if dest.exists():  # same bytes already ingested -> verify, do not rewrite
        mf = verify_version(version)
        mf["status"] = "already_ingested"
        return mf

    cal = pd.read_csv(src / names["calendar"])
    sales = pd.read_csv(src / names["sales"])
    prices = pd.read_csv(src / names["prices"])
    rep: ValidationReport = validate_all(cal, sales, prices)
    if not rep.ok and not skip_validation:
        raise RawIngestError(
            "validation failed: " + "; ".join(f"{c.name} ({c.detail})" for c in rep.failures)
        )

    dcols = [c for c in sales.columns if str(c).startswith("d_")]
    cal_dates = pd.to_datetime(cal["date"])
    d2date = dict(zip(cal["d"], cal_dates, strict=False))
    manifest = {
        "dataset_version": version,
        "schema_version": SCHEMA_VERSION,
        "source": source_note,
        "ingested_at": datetime.now(UTC).isoformat(),
        "files": {n: {"sha256": checksums[n], "bytes": (src / n).stat().st_size} for n in names.values()},
        "observed_sales": {
            "first_d": dcols[0],
            "last_d": dcols[-1],
            "n_days": len(dcols),
            "first_date": str(d2date[dcols[0]].date()),
            "last_date": str(d2date[dcols[-1]].date()),
        },
        "calendar_range": {
            "first_date": str(cal_dates.min().date()),
            "last_date": str(cal_dates.max().date()),
            "n_days": int(len(cal)),
        },
        "n_series": int(len(sales)),
        "price_coverage": price_coverage(prices, cal, len(sales)),
        "validation": rep.to_dict(),
        "note": (
            "Days after the last observed sales day exist in the calendar only; their actual sales are "
            "unavailable and are never invented."
        ),
    }
    tmp = raw_root() / f".tmp-{version}"
    _rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for n in names.values():
            shutil.copyfile(src / n, tmp / n)
            _make_readonly(tmp / n)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _make_readonly(tmp / "manifest.json")
        _publish(tmp, dest)  # atomic publish of the version directory
    except Exception:
        _rmtree(tmp)
        raise
    manifest["status"] = "ingested"
    return manifest


def list_versions() -> list[str]:
    r = raw_root()
    return (
        sorted(p.name for p in r.iterdir() if p.is_dir() and not p.name.startswith(".")) if r.exists() else []
    )


def latest_version() -> str:
    v = list_versions()
    if not v:
        raise RawIngestError(
            "No raw dataset ingested. Run `retailfc ingest --src <folder with the canonical CSVs>`."
        )
    mfs = [(json.loads((raw_root() / x / "manifest.json").read_text())["ingested_at"], x) for x in v]
    return max(mfs)[1]


def read_manifest(version: str) -> dict:
    return json.loads((raw_root() / version / "manifest.json").read_text())
