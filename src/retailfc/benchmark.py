"""Latency / runtime benchmarks with an explicit, recorded procedure.

Model inference and API latency are measured SEPARATELY:
* model inference: ``GlobalLGBM.predict`` over a fixed number of series chunks (feature build + booster) -> p50/p95 per chunk,
* API: POST /forecast, which only retrieves stored forecasts from PostgreSQL (no model computation).
Warm-up runs are discarded; hardware/software versions and parameters are written next to the results.
"""

from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from .experiments import environment_info


def _pct(x: list[float]) -> dict:
    a = np.asarray(x) * 1000
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "n": len(a),
    }


def model_inference(
    panel, point, origin: int, *, chunk: int = 500, repeats: int = 15, warmup: int = 2
) -> dict:
    sel = np.arange(min(panel.n_series, chunk))
    for _ in range(warmup):
        point.predict(panel, origin, sel, list(range(1, 29)), chunk=chunk)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        point.predict(panel, origin, sel, list(range(1, 29)), chunk=chunk)
        times.append(time.perf_counter() - t)
    out = _pct(times)
    out["series_per_call"] = int(len(sel))
    out["rows_per_call"] = int(len(sel) * 28)
    out["ms_per_series"] = out["mean_ms"] / len(sel)
    return out


def api_latency(
    base_url: str,
    keys: list[tuple[str, str]],
    *,
    requests: int = 500,
    concurrency: int = 1,
    warmup: int = 20,
    horizon: int = 28,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    picks = [keys[i] for i in rng.integers(0, len(keys), requests + warmup)]
    lat: list[float] = []
    errors = 0
    with httpx.Client(
        base_url=base_url,
        timeout=10.0,
        headers={"x-api-key": os.environ.get("RETAILFC_API_KEY", "")},
        limits=httpx.Limits(max_connections=max(1, concurrency)),
    ) as c:

        def one(k):
            t = time.perf_counter()
            r = c.post("/forecast", json={"item_id": k[0], "store_id": k[1], "horizon": horizon})
            return time.perf_counter() - t, r.status_code

        for k in picks[:warmup]:
            one(k)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as ex:
            res = list(ex.map(one, picks[warmup:]))
        wall = time.perf_counter() - t0
    for d, s in res:
        lat.append(d)
        errors += s != 200
    out = _pct(lat)
    out.update(
        {
            "requests": requests,
            "concurrency": concurrency,
            "warmup": warmup,
            "errors": int(errors),
            "throughput_rps": requests / wall,
            "horizon": horizon,
        }
    )
    return out


def write_benchmark(path: Path, results: dict, procedure: dict) -> None:
    doc = {
        "results": results,
        "procedure": procedure,
        "environment": environment_info(),
        "machine": {"processor": platform.processor(), "system": platform.platform()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2, default=str))
