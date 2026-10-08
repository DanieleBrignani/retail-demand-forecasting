"""Demand-volume and intermittency segments derived only from data before each origin.

Volume: trailing-``volume_window_days`` mean sales; edges (tertiles of series with positive mean)
are fitted once on training-period data (a reference origin <= first validation origin) and reused.
Intermittency (Syntetos-Boylan): ADI = N / n_nonzero and CV^2 of non-zero sizes over the trailing window:
smooth (ADI<1.32, CV2<0.49), erratic (ADI<1.32, CV2>=0.49), intermittent (ADI>=1.32, CV2<0.49),
lumpy (ADI>=1.32, CV2>=0.49); ``no_sales`` if there is no positive sale in the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import Panel


def fit_volume_edges(panel: Panel, ref_origin: int, cfg: dict) -> list[float]:
    w = cfg["segments"]["volume_window_days"]
    x = panel.as_of(ref_origin).sales[:, -w:].astype(np.float32).mean(axis=1)
    pos = x[x > 0]
    if len(pos) < 3:
        return [float(np.inf), float(np.inf)]
    return [float(np.quantile(pos, 1 / 3)), float(np.quantile(pos, 2 / 3))]


def segment_frame(panel: Panel, origin: int, cfg: dict, volume_edges: list[float]) -> pd.DataFrame:
    sc = cfg["segments"]
    a = panel.as_of(origin)
    vw = min(sc["volume_window_days"], origin)
    mean_v = a.sales[:, -vw:].astype(np.float32).mean(axis=1)
    vol = np.where(
        mean_v == 0,
        "inactive",
        np.where(mean_v <= volume_edges[0], "low", np.where(mean_v <= volume_edges[1], "mid", "high")),
    )
    iw = min(sc["intermittency_window_days"], origin)
    x = a.sales[:, -iw:].astype(np.float32)
    nz = x > 0
    n_nz = nz.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        adi = iw / n_nz
        s1 = np.where(nz, x, 0).sum(axis=1)
        mu = s1 / n_nz
        var = (np.where(nz, x - mu[:, None], 0) ** 2).sum(axis=1) / n_nz
        cv2 = var / mu**2
    cls = np.where(
        adi < sc["adi_cutoff"],
        np.where(cv2 < sc["cv2_cutoff"], "smooth", "erratic"),
        np.where(cv2 < sc["cv2_cutoff"], "intermittent", "lumpy"),
    )
    cls = np.where(n_nz == 0, "no_sales", cls)
    return pd.DataFrame(
        {
            "series_idx": np.arange(panel.n_series, dtype=np.int32),
            "volume_segment": vol,
            "intermittency_segment": cls,
            "trailing_mean": mean_v,
            "adi": adi,
            "cv2": cv2,
        }
    )
