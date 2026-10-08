"""Weekly-granularity study, protocol ``weekly-v2`` (docs/weekly_protocol_v2.md, configs/weekly_v2.yaml; frozen before the run).

Daily panel -> weekly product x channel matrix (complete 7-day calendar blocks of the dataset) -> baselines, intermittent-demand
methods and a global direct LightGBM, evaluated on rolling origins. Every forecast made at origin ``t`` reads blocks ``<= t`` only
(``Y[:, :t]``); the universe at ``t`` is the set of products already observed by ``t`` (cold start is reported, not scored).
Selection / promotion is mechanical (:func:`select_model`); nothing in this module is tuned on results.
The earlier v1 code (all series, no universe policy) is superseded and not kept in this repository.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from .panel import Panel


def _default_config() -> Path:
    """configs/weekly_v2.yaml of a source checkout, or of the container (RETAILFC_HOME=/app)."""
    import os

    for root in (os.environ.get("RETAILFC_HOME"), str(Path(__file__).resolve().parents[2])):
        if root and (Path(root) / "configs" / "weekly_v2.yaml").exists():
            return Path(root) / "configs" / "weekly_v2.yaml"
    return Path("configs/weekly_v2.yaml")


DEFAULT_CONFIG = _default_config()
MIN_ORIGIN = 26  # first feature origin (needs 26 weeks of history)
EPS = 0.1
FEATURES = (
    [f"lag{k}" for k in range(1, 9)]
    + [f"mean{w}" for w in (4, 8, 13, 26)]
    + [f"nz{w}" for w in (4, 13, 26)]
    + ["since_sale", "age_weeks", "max13", "ly_mean3", "ly_lag", "woy_sin", "woy_cos", "chan", "h"]
)
WEEKLY_FEATURE_VERSION = (
    "wfv1-" + hashlib.sha256(json.dumps(FEATURES).encode()).hexdigest()[:8]
)  # no price / event inputs


# ------------------------------------------------------------------ config
def load_cfg(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def cfg_hash(path: str | Path | None = None) -> str:
    """SHA-256 of the file with line endings normalised to LF, so a Windows checkout and a Linux checkout hash alike."""
    raw = Path(path or DEFAULT_CONFIG).read_bytes()
    return hashlib.sha256(raw.replace(bytes([13, 10]), bytes([10]))).hexdigest()


def horizons(cfg: dict) -> list[int]:
    return list(cfg["horizons"])


def qs(cfg: dict) -> list[float]:
    return list(cfg["quantiles"])


# ------------------------------------------------------------------ data
def to_weekly(panel: Panel) -> tuple[np.ndarray, pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """(Y[n, W] units per block, ids, block start dates, product index per series) using complete 7-day blocks only."""
    cal = panel.calendar.iloc[: panel.n_days]
    size = cal.groupby("week_id").size()
    full = size[size == 7].index.to_numpy()
    keep = cal["week_id"].isin(full).to_numpy()
    wid = cal["week_id"].to_numpy()[keep]
    sales = panel.sales[:, keep].astype(np.float32)
    uniq = np.unique(wid)
    Y = sales.reshape(panel.n_series, len(uniq), 7).sum(axis=2)
    starts = pd.to_datetime(cal["date"].to_numpy()[keep][::7])
    pidx = pd.factorize(panel.ids["item_id"])[0].astype(np.int64)
    return Y, panel.ids[["item_id", "store_id"]].copy(), pd.DatetimeIndex(starts), pidx


def eligible_mask(Y: np.ndarray, pidx: np.ndarray, t: int) -> np.ndarray:
    """Series whose PRODUCT has >= 1 recorded sale (any channel) in blocks <= t. Reads Y[:, :t] only."""
    ever = (Y[:, :t] > 0).any(axis=1)
    prod = np.zeros(pidx.max() + 1, bool)
    np.logical_or.at(prod, pidx, ever)
    return prod[pidx]


# ------------------------------------------------------------------ baselines (history = Y[:, :t])
def baseline_forecasts(Y: np.ndarray, t: int, cfg: dict | None = None) -> dict[str, np.ndarray]:
    """Each value is (n, len(horizons)). ``all_zero`` is a reference, not a serving candidate."""
    cfg = cfg or load_cfg()
    hz = horizons(cfg)
    H = Y[:, :t]
    n, k = len(H), len(hz)
    out = {
        "all_zero": np.zeros((n, k), np.float32),
        "naive": np.repeat(H[:, -1:], k, axis=1),
        "mean4": np.repeat(H[:, -4:].mean(axis=1, keepdims=True), k, axis=1),
        "mean13": np.repeat(H[:, -13:].mean(axis=1, keepdims=True), k, axis=1),
    }
    cols = [t + h - 52 - 1 for h in hz]  # 0-based column of block t+h-52, always <= t-1
    out["seasonal_naive52"] = H[:, cols] if min(cols) >= 0 else out["naive"].copy()
    c, ts = cfg["models"]["croston"], cfg["models"]["tsb"]
    out["croston_sba"], out["tsb"] = _croston_tsb(
        H, c["alpha"], ts["alpha"], ts["beta"], ts["init_prob_weeks"], k
    )
    return out


def _croston_tsb(H, alpha=0.1, a_tsb=0.1, beta=0.1, init_weeks=13, k=4) -> tuple[np.ndarray, np.ndarray]:
    """Croston with the Syntetos-Boylan correction, and TSB. Series with no demand yet forecast 0.
    Initialisation: size at the first demand, interval = the first observed inter-demand gap (until a second demand: its
    position); TSB probability starts at the share of non-zero blocks in the first ``init_weeks`` blocks."""
    n, t = H.shape
    z = np.zeros(n)
    p = np.ones(n)
    started = np.zeros(n, bool)
    since = np.zeros(n)
    nd = np.zeros(n, int)
    pi = (H[:, : min(init_weeks, t)] > 0).mean(axis=1)
    zt = np.zeros(n)
    started_t = np.zeros(n, bool)
    for j in range(t):
        y = H[:, j].astype(np.float64)
        pos = y > 0
        since += 1
        first = pos & ~started
        z[first], p[first], since[first], started[first] = y[first], since[first], 0, True
        upd = pos & started & ~first
        second = upd & (nd == 1)
        z[upd] += alpha * (y[upd] - z[upd])
        p[upd & ~second] += alpha * (since[upd & ~second] - p[upd & ~second])
        p[second] = since[second]
        since[upd] = 0
        nd[first | upd] += 1
        firstt = pos & ~started_t
        zt[firstt], started_t[firstt] = y[firstt], True
        updt = pos & started_t & ~firstt
        zt[updt] += a_tsb * (y[updt] - zt[updt])
        pi += beta * (pos.astype(float) - pi)
    cro = np.where(started, (1 - alpha / 2) * z / np.maximum(p, 1e-9), 0.0)
    tsb = np.where(started_t, pi * zt, 0.0)
    return (
        np.repeat(cro[:, None], k, axis=1).astype(np.float32),
        np.repeat(tsb[:, None], k, axis=1).astype(np.float32),
    )


def empirical_quantiles(Y: np.ndarray, t: int, cfg: dict | None = None) -> np.ndarray:
    """(n, n_q): the series' own empirical quantiles of its last 26 blocks."""
    cfg = cfg or load_cfg()
    w = Y[:, :t][:, -26:]
    return np.quantile(w, qs(cfg), axis=1).T.astype(np.float32)


# ------------------------------------------------------------------ features
def features_at(Y: np.ndarray, t: int, week_start: pd.DatetimeIndex, chan: np.ndarray, h: int) -> np.ndarray:
    """Feature matrix (n, len(FEATURES)) for origin t and horizon h. Reads Y[:, :t] only (no price / event inputs)."""
    H = Y[:, :t]
    n = len(H)
    cols: list[np.ndarray] = [H[:, -k] for k in range(1, 9)]
    cols += [H[:, -w:].mean(axis=1) for w in (4, 8, 13, 26)]
    cols += [(H[:, -w:] > 0).mean(axis=1) for w in (4, 13, 26)]
    pos = H > 0
    ever = pos.any(axis=1)
    since = np.where(
        ever, np.argmax(pos[:, ::-1], axis=1), 104
    )  # blocks since the most recent positive block
    first = np.argmax(pos, axis=1)
    age = np.where(ever, np.minimum(t - first, 104), 0)
    cols += [since.astype(np.float32), age.astype(np.float32), H[:, -13:].max(axis=1)]
    c = t + h - 52  # 1-based block one year before the target block
    if c - 1 >= 1 and c + 1 <= t:
        ly3 = H[:, c - 2 : c + 1].mean(axis=1)
        lyl = H[:, c - 1]
    else:
        ly3 = lyl = np.full(n, np.nan, np.float32)
    cols += [ly3, lyl]
    tw = week_start[0] + pd.Timedelta(
        weeks=t + h - 1
    )  # start date of the target block (calendar known in advance)
    woy = tw.isocalendar().week
    cols += [
        np.full(n, np.sin(2 * np.pi * woy / 52), np.float32),
        np.full(n, np.cos(2 * np.pi * woy / 52), np.float32),
    ]
    cols += [chan.astype(np.float32), np.full(n, h, np.float32)]
    return np.column_stack(cols).astype(np.float32)


def level_of(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    return X[:, FEATURES.index("mean13")] + eps


def training_set(Y, week_start, chan, pidx, t: int, hz: list[int]):
    """Rows (t', h) with 26 <= t' and t'+h <= t, only for products observed by t'. Returns X, y, target_block (1-based)."""
    Xs, ys, tw = [], [], []
    for tp in range(MIN_ORIGIN, t):
        el = np.flatnonzero(eligible_mask(Y, pidx, tp))
        for h in hz:
            if tp + h > t:
                continue
            Xs.append(features_at(Y, tp, week_start, chan, h)[el])
            ys.append(Y[el, tp + h - 1])
            tw.append(np.full(len(el), tp + h))
    return np.concatenate(Xs), np.concatenate(ys).astype(np.float32), np.concatenate(tw)


class WeeklyLGBM:
    def __init__(self, point: lgb.Booster, quant: dict[float, lgb.Booster], eps: float):
        self.point, self.quant, self.eps = point, quant, eps

    @classmethod
    def fit(cls, Y, week_start, chan, pidx, t: int, cfg: dict, quantiles: bool = True) -> WeeklyLGBM:
        m = cfg["models"]["lgbm"]
        hz = horizons(cfg)
        eps = float(m["level_offset_eps"])
        X, y, tw = training_set(Y, week_start, chan, pidx, t, hz)
        lv = level_of(X, eps)
        va = tw > t - 4  # early stopping on the last 4 target blocks; training targets end at t-4
        tr = ~va
        if tr.sum() == 0 or va.sum() == 0:
            raise ValueError(
                f"origin {t}: not enough history to train (needs origins >= {MIN_ORIGIN + 8}); refusing to fit"
            )
        P = dict(m["params"], verbose=-1)
        d_tr = lgb.Dataset(X[tr], y[tr], init_score=np.log(lv[tr]), free_raw_data=False)
        d_va = lgb.Dataset(X[va], y[va], init_score=np.log(lv[va]), reference=d_tr)
        point = lgb.train(
            P,
            d_tr,
            m["num_boost_round"],
            valid_sets=[d_va],
            callbacks=[lgb.early_stopping(m["early_stopping_rounds"], verbose=False)],
        )
        quant = {}
        if quantiles:
            r = y / lv
            for q in qs(cfg):
                p = {**P, "objective": "quantile", "alpha": q, "num_leaves": 31, "min_data_in_leaf": 200}
                p.pop("tweedie_variance_power")
                quant[q] = lgb.train(p, lgb.Dataset(X[tr], r[tr]), m["quantile_rounds"])
        return cls(point, quant, eps)

    def predict(self, Y, week_start, chan, t: int, cfg: dict, rows: np.ndarray | None = None):
        """point (m, K) and RAW quantiles (m, K, n_q) (not yet monotone-corrected), for the rows given (default: all)."""
        hz, q = horizons(cfg), qs(cfg)
        rows = np.arange(len(Y)) if rows is None else rows
        pt = np.zeros((len(rows), len(hz)), np.float32)
        qt = np.zeros((len(rows), len(hz), len(q)), np.float32)
        for j, h in enumerate(hz):
            X = features_at(Y, t, week_start, chan, h)[rows]
            lv = level_of(X, self.eps)
            pt[:, j] = np.exp(self.point.predict(X, raw_score=True) + np.log(lv))
            for k, tau in enumerate(q):
                if tau in self.quant:
                    qt[:, j, k] = self.quant[tau].predict(X) * lv
        return np.maximum(pt, 0), qt


# ------------------------------------------------------------------ segments (past information only)
def segments_at(Y: np.ndarray, t: int, chan: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    """Segments for ``rows`` derived from blocks <= t (window = last 26 blocks); terciles use eligible series with sales only."""
    H = Y[rows, :t]
    w = H[:, -26:]
    units = w.sum(axis=1)
    nz = (w > 0).mean(axis=1)
    act = units > 0
    vol = np.full(len(rows), "inactive", dtype=object)
    if act.sum() >= 3:
        lo, hi = np.quantile(units[act], [1 / 3, 2 / 3])
        vol[act] = np.where(units[act] <= lo, "low", np.where(units[act] <= hi, "mid", "high"))
    inter = np.where(nz >= 0.75, "smooth", np.where(nz >= 0.25, "intermittent", "sparse")).astype(object)
    inter[~act] = "inactive"
    pos = H > 0
    ever = pos.any(axis=1)
    first = np.argmax(pos, axis=1)
    hist_w = np.where(ever, t - first, 0)
    hist = np.where(
        ~ever, "no_channel_history", np.where(hist_w < 13, "<13w", np.where(hist_w < 52, "13-51w", ">=52w"))
    )
    return pd.DataFrame(
        {
            "series_idx": rows.astype(np.int32),
            "volume": vol,
            "intermittency": inter,
            "history": hist.astype(object),
            "channel": np.where(chan[rows] == 1, "INT_1", "UK_1"),
            "fallback": (~ever).astype(
                np.int8
            ),  # no sale yet in THIS channel: forecast is a flagged fallback
            "active": act.astype(np.int8),
        }
    )


def one_step_scale(Y: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Denominators from the pre-origin history: mean |Δ1| and sqrt(mean Δ1²) over blocks <= t (0 for constant series)."""
    d = np.diff(Y[:, :t], axis=1)
    return np.abs(d).mean(axis=1), np.sqrt((d**2).mean(axis=1))


# ------------------------------------------------------------------ one origin
def run_origin(Y, week_start, chan, pidx, t: int, cfg: dict, with_lgbm: bool = True) -> dict:
    hz = horizons(cfg)
    el = eligible_mask(Y, pidx, t)
    rows, cold = np.flatnonzero(el), np.flatnonzero(~el)
    tgt = slice(t, t + max(hz))
    cov = {
        "origin": t,
        "n_series_universe": int(len(Y)),
        "n_eligible": int(len(rows)),
        "n_cold_start": int(len(cold)),
        "target_units_eligible": float(Y[rows, tgt].sum()),
        "target_units_cold_start": float(Y[cold, tgt].sum()),
    }
    cov["share_units_cold_start"] = cov["target_units_cold_start"] / max(
        cov["target_units_eligible"] + cov["target_units_cold_start"], 1e-12
    )
    t_b = time.time()
    bf = baseline_forecasts(Y, t, cfg)
    point = {k: v[rows] for k, v in bf.items()}
    baseline_seconds = time.time() - t_b  # all baselines together: an upper bound for each of them
    eq = empirical_quantiles(Y, t, cfg)[rows]
    quant = {
        "empirical_q": np.repeat(eq[:, None, :], len(hz), axis=1),
        "all_zero": np.zeros((len(rows), len(hz), len(qs(cfg))), np.float32),
    }
    meta = {"origin": t, "baseline_batch_seconds": baseline_seconds}
    if with_lgbm:
        t0 = time.time()
        m = WeeklyLGBM.fit(Y, week_start, chan, pidx, t, cfg)
        meta["train_seconds"] = time.time() - t0
        meta["best_iteration"] = int(m.point.best_iteration)
        t1 = time.time()
        point["lgbm_w"], quant["lgbm_w_raw"] = m.predict(Y, week_start, chan, t, cfg, rows)
        meta["batch_seconds"] = time.time() - t1
    s1, s2 = one_step_scale(Y, t)
    seg = segments_at(Y, t, chan, rows)
    actual = np.stack([Y[rows, t + h - 1] for h in hz], axis=1).astype(np.float64)
    return {
        "rows": rows,
        "point": point,
        "quant": quant,
        "actual": actual,
        "scale1": s1[rows],
        "scale2": s2[rows],
        "seg": seg,
        "meta": meta,
        "coverage": cov,
    }


# ------------------------------------------------------------------ interval post-processing
def fix_crossing(q: np.ndarray) -> np.ndarray:
    """Documented policy: non-negative and monotone via running maximum over the (ordered) quantiles."""
    return np.maximum.accumulate(np.maximum(q, 0), axis=2)


def crossing_rate(q: np.ndarray) -> float:
    return float((np.diff(q, axis=2) < -1e-9).any(axis=2).mean())


def calibrate_previous(res: dict[int, dict], cfg: dict) -> None:
    """Split-conformal widening of lgbm_w quantiles with the PRECEDING origin's realised errors only (no use of the current
    window's actuals). The first origin has no preceding origin and stays uncalibrated. Adds ``lgbm_w`` / ``lgbm_w_cal`` to quant."""
    origins = sorted(res)
    alpha = 1 - float(cfg["interval"]["nominal"])
    for i, t in enumerate(origins):
        r = res[t]
        if "lgbm_w_raw" not in r["quant"]:
            continue
        raw = r["quant"]["lgbm_w_raw"]
        r["quant"]["lgbm_w"] = fix_crossing(raw)
        cal = r["quant"]["lgbm_w"].copy()
        if i > 0:
            p = res[origins[i - 1]]
            qp = p["quant"]["lgbm_w"]
            y = p["actual"]
            for j in range(raw.shape[1]):
                s = np.maximum(qp[:, j, 0] - y[:, j], y[:, j] - qp[:, j, -1])
                n = len(s)
                lvl = min(1.0, (1 - alpha) * (1 + 1.0 / n))
                c = float(np.quantile(s, lvl))
                cal[:, j, 0] = np.maximum(cal[:, j, 0] - c, 0)
                cal[:, j, -1] = cal[:, j, -1] + c
        r["quant"]["lgbm_w_cal"] = cal


# ------------------------------------------------------------------ scoring
def long_frame(res: dict[int, dict], cfg: dict) -> pd.DataFrame:
    hz = horizons(cfg)
    parts = []
    for t, r in res.items():
        m = len(r["rows"])
        base = pd.DataFrame(
            {
                "origin": np.int16(t),
                "series_idx": np.repeat(r["rows"], len(hz)).astype(np.int32),
                "h": np.tile(np.array(hz, np.int8), m),
                "actual": r["actual"].reshape(-1),
                "scale1": np.repeat(r["scale1"], len(hz)),
                "scale2": np.repeat(r["scale2"], len(hz)),
            }
        )
        seg = (
            r["seg"].drop(columns=["series_idx"]).loc[np.repeat(np.arange(m), len(hz))].reset_index(drop=True)
        )
        base = pd.concat([base, seg], axis=1)
        for name, f in r["point"].items():
            parts.append(base.assign(model=name, forecast=f.reshape(-1).astype(np.float64)))
    return pd.concat(parts, ignore_index=True)


def metrics_by(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Pooled WAPE/MAE/RMSE/bias plus MASE/RMSSE (rows with a zero denominator excluded for EVERY model; counts reported)."""
    e = df["forecast"] - df["actual"]
    ok = df["scale1"] > 0
    t = pd.DataFrame(
        {
            **{c: df[c] for c in by},
            "n": 1,
            "abs_e": e.abs(),
            "sq_e": e * e,
            "e": e,
            "abs_a": df["actual"].abs(),
            "ok": ok.astype(int),
            "ase": np.where(ok, e.abs() / df["scale1"].where(ok, 1.0), 0.0),
            "sse": np.where(ok, (e * e) / (df["scale2"].where(ok, 1.0) ** 2), 0.0),
        }
    )
    g = t.groupby(by, observed=True).sum(numeric_only=True)
    out = pd.DataFrame(index=g.index)
    out["wape"] = g["abs_e"] / g["abs_a"].where(g["abs_a"] > 0)
    out["mae"] = g["abs_e"] / g["n"]
    out["rmse"] = np.sqrt(g["sq_e"] / g["n"])
    out["bias_pct"] = g["e"] / g["abs_a"].where(g["abs_a"] > 0)
    out["mase"] = g["ase"] / g["ok"].where(g["ok"] > 0)
    out["rmsse"] = np.sqrt(g["sse"] / g["ok"].where(g["ok"] > 0))
    out["n"] = g["n"].astype(int)
    out["n_mase_scored"] = g["ok"].astype(int)
    out["n_mase_excluded"] = (g["n"] - g["ok"]).astype(int)
    out["demand"] = g["abs_a"]
    return out


def aggregate_levels(df: pd.DataFrame) -> pd.DataFrame:
    """L2 channel and L3 total: forecasts and actuals summed over ELIGIBLE series per (origin, h, group)."""
    rows = []
    for lev, keys in (("L2_channel", ["channel"]), ("L3_total", [])):
        g = (
            df.groupby(["model", "origin", "h", *keys], observed=True)[["forecast", "actual"]]
            .sum()
            .reset_index()
        )
        g["level"] = lev
        rows.append(g)
    a = pd.concat(rows, ignore_index=True)
    e = a["forecast"] - a["actual"]
    a["abs_e"], a["abs_a"], a["e"] = e.abs(), a["actual"].abs(), e
    pooled = a.groupby(["model", "level"])[["abs_e", "abs_a", "e"]].sum()
    pooled["wape"], pooled["bias_pct"] = pooled["abs_e"] / pooled["abs_a"], pooled["e"] / pooled["abs_a"]
    return pooled[["wape", "bias_pct"]]


def interval_diagnostics(res: dict[int, dict], cfg: dict) -> pd.DataFrame:
    """Per quantile model (lgbm_w raw-fixed, lgbm_w_cal, empirical_q, all_zero): nominal vs empirical coverage on all rows and on
    ACTIVE series, width, pinball, and the RAW crossing rate before correction -- overall, by horizon and by volume segment."""
    hz, q = horizons(cfg), qs(cfg)
    rows = []
    for name in ("lgbm_w", "lgbm_w_cal", "empirical_q", "all_zero"):
        recs = []
        for t, r in res.items():
            if name not in r["quant"]:
                continue
            Q = r["quant"][name]
            a = r["actual"]
            inside = (a >= Q[:, :, 0]) & (a <= Q[:, :, -1])
            pin = sum(
                np.maximum(tau * (a - Q[:, :, k]), (tau - 1) * (a - Q[:, :, k])) for k, tau in enumerate(q)
            ) / len(q)
            raw_cross = crossing_rate(r["quant"]["lgbm_w_raw"]) if name.startswith("lgbm_w") else 0.0
            m = len(r["rows"])
            recs.append(
                pd.DataFrame(
                    {
                        "origin": t,
                        "h": np.tile(hz, m),
                        "volume": np.repeat(r["seg"]["volume"].to_numpy(), len(hz)),
                        "active": np.repeat(r["seg"]["active"].to_numpy(), len(hz)),
                        "inside": inside.reshape(-1),
                        "width": (Q[:, :, -1] - Q[:, :, 0]).reshape(-1),
                        "pinball": pin.reshape(-1),
                        "raw_cross": raw_cross,
                    }
                )
            )
        if not recs:
            continue
        d = pd.concat(recs, ignore_index=True)
        for scope, sub in (("all", d), ("active_only", d[d["active"] == 1])):
            rows.append(_diag_row(name, scope, "overall", "", sub))
            for h, g in sub.groupby("h"):
                rows.append(_diag_row(name, scope, "horizon", str(h), g))
            for v, g in sub.groupby("volume"):
                rows.append(_diag_row(name, scope, "volume", str(v), g))
    return pd.DataFrame(rows)


def _diag_row(model, scope, by, key, d) -> dict:
    return {
        "model": model,
        "scope": scope,
        "by": by,
        "key": key,
        "n": int(len(d)),
        "nominal": 0.8,
        "empirical_coverage": float(d["inside"].mean()),
        "mean_width": float(d["width"].mean()),
        "pinball": float(d["pinball"].mean()),
        "raw_crossing_rate": float(d["raw_cross"].mean()),
    }


# ------------------------------------------------------------------ selection (mechanical)
def select_model(tabs: dict[str, pd.DataFrame], cfg: dict, meta: dict) -> dict:
    """Apply docs/weekly_protocol_v2.md section 7. Returns every check for every candidate; ``selected`` is None if nobody qualifies."""
    R = cfg["selection"]
    L1, po, seg = tabs["L1"], tabs["per_origin"], tabs["segments"]
    base_names = [b for b in cfg["models"]["baselines"] if b in L1.index]
    cand = base_names + (["lgbm_w"] if "lgbm_w" in L1.index else [])
    zero_w = float(L1.loc["all_zero", "wape"])
    best_mase = float(L1.loc[base_names, "mase"].min())
    out: dict = {"protocol": cfg["protocol_version"], "candidates": {}}

    def chk(name, passed, detail):
        return {"check": name, "passed": bool(passed), "detail": detail}

    for m in cand:
        fc = tabs.get("forecast_coverage")
        if fc is None or m not in fc.index:
            c = [chk("forecast_coverage", False, "not measured (not verified)")]
        else:
            c = [
                chk(
                    "forecast_coverage",
                    fc.loc[m, "coverage"] >= R["min_forecast_coverage"],
                    f"{fc.loc[m, 'coverage']:.4f} of eligible series-rows have a finite forecast (min {R['min_forecast_coverage']})",
                )
            ]
        c.append(
            chk(
                "beats_all_zero",
                L1.loc[m, "wape"] <= R["max_wape_vs_all_zero"] * zero_w,
                f"wape {L1.loc[m, 'wape']:.4f} vs all-zero {zero_w:.4f} (max ratio {R['max_wape_vs_all_zero']})",
            )
        )
        c.append(
            chk(
                "mase_guard",
                L1.loc[m, "mase"] <= R["max_mase_vs_best_baseline_ratio"] * best_mase,
                f"mase {L1.loc[m, 'mase']:.4f} vs best baseline {best_mase:.4f} (max ratio {R['max_mase_vs_best_baseline_ratio']})",
            )
        )
        pb = float(L1.loc[m, "bias_pct"]) * 100
        ob = po[po["model"] == m]["bias_pct"].abs() * 100
        within = float((ob <= R["max_abs_origin_bias_pct"]).mean())
        c.append(
            chk(
                "bias_guard",
                abs(pb) <= R["max_abs_pooled_bias_pct"] and within >= R["min_origins_within_origin_bias"],
                f"pooled bias {pb:+.1f}% (max ±{R['max_abs_pooled_bias_pct']}); {within:.0%} of origins within "
                f"±{R['max_abs_origin_bias_pct']}% (min {R['min_origins_within_origin_bias']:.0%}); worst origin {ob.max():.1f}%",
            )
        )
        ts, bs = meta.get("train_seconds", {}).get(m), meta.get("batch_seconds", {}).get(m)
        if ts is None or bs is None:  # a missing measurement is NOT a pass
            c.append(chk("runtime", False, f"not measured (not verified): train {ts}, batch {bs}"))
        else:
            c.append(
                chk(
                    "runtime",
                    ts <= R["max_train_seconds_per_origin"] and bs <= R["max_batch_seconds"],
                    f"max train {ts:.2f}s (limit {R['max_train_seconds_per_origin']}), max batch {bs:.2f}s "
                    f"(limit {R['max_batch_seconds']}); baselines have no training (0 s) and their batch time is the "
                    f"time of all baselines together",
                )
            )
        out["candidates"][m] = {"checks": c}
    pre = {m for m in cand if all(x["passed"] for x in out["candidates"][m]["checks"])}
    strong = [b for b in base_names if b in pre]
    strongest = min(strong, key=lambda b: L1.loc[b, "wape"]) if strong else None
    out["strongest_baseline"] = strongest
    for m in cand:
        c = out["candidates"][m]["checks"]
        if strongest is None:
            c.append(chk("segment_degradation", False, "no qualified baseline to compare with"))
        else:
            mine = seg[seg["model"] == m].set_index(["segment_type", "segment"])
            ref = seg[seg["model"] == strongest].set_index(["segment_type", "segment"])
            tot = float(L1.loc[m, "demand"])
            bad = []
            for key in mine.index:
                if mine.loc[key, "demand"] / tot < R["min_segment_demand_share"] or key not in ref.index:
                    continue
                a, b = mine.loc[key, "wape"], ref.loc[key, "wape"]
                if m != strongest and a > b * (1 + R["max_segment_degradation_pct"] / 100):
                    bad.append(f"{key[0]}:{key[1]} {a:.3f} vs {b:.3f}")
            c.append(
                chk(
                    "segment_degradation",
                    not bad,
                    "; ".join(bad) if bad else "no segment >5% worse than the strongest baseline",
                )
            )
        if m == "lgbm_w":
            if strongest is None:
                c.append(chk("improves_on_strongest_baseline", False, "no qualified baseline"))
            else:
                gain = (1 - L1.loc[m, "wape"] / L1.loc[strongest, "wape"]) * 100
                c.append(
                    chk(
                        "improves_on_strongest_baseline",
                        gain >= R["min_improvement_over_strongest_baseline_pct"],
                        f"wape gain {gain:+.2f}% vs {strongest} (min {R['min_improvement_over_strongest_baseline_pct']}%)",
                    )
                )
        out["candidates"][m]["qualified"] = all(x["passed"] for x in c)
    qual = [m for m in cand if out["candidates"][m]["qualified"]]
    out["qualified"] = qual
    out["selected"] = min(qual, key=lambda m: L1.loc[m, "wape"]) if qual else None
    out["note"] = (
        "revised historical evaluation on already-inspected data; thresholds partly written after weekly v1 results "
        "(docs/weekly_protocol_v2.md section 0). Not an independent validation."
    )
    return out


# ------------------------------------------------------------------ run
def environment() -> dict:
    import os
    import sys

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
    }


def run(
    panel: Panel,
    out_dir: Path,
    cfg_path: str | Path | None = None,
    origins: list[int] | None = None,
    with_lgbm: bool = True,
) -> dict:
    cfg = load_cfg(cfg_path)
    hz = horizons(cfg)
    Y, ids, starts, pidx = to_weekly(panel)
    chan = (ids["store_id"] == "INT_1").to_numpy().astype(np.int8)
    W = Y.shape[1]
    origins = origins or list(cfg["origins"])
    assert max(origins) + max(hz) <= W, (max(origins), W)
    out_dir = Path(out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    t_run = time.time()
    res: dict[int, dict] = {}
    for t in origins:
        t0 = time.time()
        res[t] = run_origin(Y, starts, chan, pidx, t, cfg, with_lgbm)
        print(f"origin {t} done in {time.time() - t0:.0f}s", flush=True)
    calibrate_previous(res, cfg)
    df = long_frame(res, cfg)
    models = list(res[origins[0]]["point"])
    tabs: dict[str, pd.DataFrame] = {"L1": metrics_by(df, ["model"])}
    tabs["per_origin"] = metrics_by(df, ["model", "origin"]).reset_index()
    tabs["by_horizon"] = metrics_by(df, ["model", "h"])
    segs = [
        metrics_by(df, ["model", c]).reset_index().rename(columns={c: "segment"}).assign(segment_type=c)
        for c in ("channel", "volume", "intermittency", "history")
    ]
    tabs["segments"] = pd.concat(segs, ignore_index=True)
    tabs["levels"] = aggregate_levels(df)
    po = tabs["per_origin"]
    var = po.groupby("model")["wape"].agg(["mean", "std", "min", "max"])
    base = po[po["model"] == "mean4"].set_index("origin")["wape"]
    var["origins_better_than_mean4"] = [
        int((po[po["model"] == m].set_index("origin")["wape"] < base).sum()) for m in var.index
    ]
    tabs["origin_variability"] = var
    tabs["intervals"] = interval_diagnostics(res, cfg)
    tabs["universe_coverage"] = pd.DataFrame([r["coverage"] for r in res.values()])
    # measured, not assumed: share of eligible series-rows with a finite forecast, per model
    cov_m = df.groupby("model")["forecast"].apply(lambda x: float(np.isfinite(x).mean())).rename("coverage")
    tabs["forecast_coverage"] = cov_m.to_frame()
    fb = df[df["model"] == "naive"]
    tabs["fallback_rate"] = pd.DataFrame(
        {
            "fallback_no_channel_history_share": [float(fb["fallback"].mean())],
            "forecast_coverage_min_over_models": [float(cov_m.min())],
        }
    )
    base_sec = max(r["meta"]["baseline_batch_seconds"] for r in res.values())
    meta = {
        "train_seconds": {b: 0.0 for b in cfg["models"]["baselines"]},  # nothing is trained
        "batch_seconds": {b: base_sec for b in cfg["models"]["baselines"]},
    }
    if with_lgbm:
        meta["train_seconds"]["lgbm_w"] = max(r["meta"]["train_seconds"] for r in res.values())
        meta["batch_seconds"]["lgbm_w"] = max(r["meta"]["batch_seconds"] for r in res.values())
    sel = select_model(tabs, cfg, meta) if with_lgbm else {}
    for k, d in tabs.items():
        d.to_csv(
            out_dir / f"{k}.csv",
            index=k in ("L1", "levels", "origin_variability", "by_horizon", "forecast_coverage"),
        )
    for m in models:
        df[df["model"] == m].to_parquet(out_dir / "predictions" / f"{m}.parquet", index=False)
    (out_dir / "selection.json").write_text(json.dumps(sel, indent=2, default=str))
    manifest = {
        "protocol_version": cfg["protocol_version"],
        "config_path": str(cfg_path or DEFAULT_CONFIG),
        "config_sha256": cfg_hash(cfg_path),
        "dataset_version": panel.dataset_version,
        "weekly_feature_version": WEEKLY_FEATURE_VERSION,
        "calendar": cfg["calendar"],
        "origins": origins,
        "horizons": hz,
        "n_blocks": int(W),
        "n_series_universe": int(len(Y)),
        "seed": cfg["models"]["lgbm"]["params"]["seed"],
        "tuning_trials": cfg["models"]["lgbm"]["tuning_trials"],
        "runtime_seconds": time.time() - t_run,
        "per_origin_meta": {t: res[t]["meta"] for t in origins},
        "environment": environment(),
        "command": "retailfc weekly backtest --config configs/weekly_v2.yaml",
        "status": "revised historical evaluation on already-inspected data (not an independent holdout)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return {"manifest": manifest, "selection": sel, "tables": {k: d.to_string() for k, d in tabs.items()}}
