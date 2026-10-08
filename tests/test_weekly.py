"""Weekly study (protocol weekly-v2) mechanics on SYNTHETIC data: aggregation, information boundary, universe policy,
intermittent methods, scoring, interval diagnostics and the mechanical selection rule. Not evidence of forecasting quality."""

import copy

import numpy as np
import pandas as pd
import pytest

from retailfc import weekly as W

T = 70
BASE_NAMES = ["naive", "mean4", "mean13", "seasonal_naive52", "croston_sba", "tsb"]


@pytest.fixture(scope="module")
def cfg():
    return W.load_cfg()


def _Y(seed=0, n=60, weeks=80):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(1.0, 1.0, size=(n, 1))
    return rng.poisson(lam * (1 + 0.5 * np.sin(np.arange(weeks) / 8))).astype(np.float32)


def _starts(weeks=80):
    return pd.date_range("2009-12-01", periods=weeks, freq="7D")


def test_to_weekly_drops_the_partial_last_block():
    from retailfc.panel import Panel

    days = 7 * 3 + 4
    cal = pd.DataFrame({"date": pd.date_range("2009-12-01", periods=days + 10).astype(str)})
    cal["week_id"] = np.arange(len(cal)) // 7 + 1
    sales = np.ones((2, days), np.int16)
    ids = pd.DataFrame(
        {"item_id": ["a", "b"], "dept_id": "d", "cat_id": "c", "store_id": ["UK_1", "INT_1"], "state_id": "s"}
    )
    p = Panel(ids, sales, cal, np.full((2, cal.week_id.nunique()), np.nan, np.float32))
    Y, _, st, pidx = W.to_weekly(p)
    assert Y.shape == (2, 3) and (Y == 7).all() and len(st) == 3 and list(pidx) == [0, 1]


def test_forecasts_features_and_universe_never_read_the_future(cfg):
    Y = _Y()
    chan = np.zeros(len(Y), np.int8)
    pidx = np.arange(len(Y))
    Y2 = Y.copy()
    Y2[:, T:] = Y2[:, T:] * 7 + 3  # corrupt every block after the origin
    a, b = W.baseline_forecasts(Y, T, cfg), W.baseline_forecasts(Y2, T, cfg)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    np.testing.assert_array_equal(W.empirical_quantiles(Y, T, cfg), W.empirical_quantiles(Y2, T, cfg))
    np.testing.assert_array_equal(W.eligible_mask(Y, pidx, T), W.eligible_mask(Y2, pidx, T))
    for h in W.horizons(cfg):
        np.testing.assert_array_equal(
            W.features_at(Y, T, _starts(), chan, h), W.features_at(Y2, T, _starts(), chan, h)
        )
    rows = np.arange(len(Y))
    pd.testing.assert_frame_equal(W.segments_at(Y, T, chan, rows), W.segments_at(Y2, T, chan, rows))
    np.testing.assert_array_equal(W.one_step_scale(Y, T)[0], W.one_step_scale(Y2, T)[0])


def test_universe_is_product_level_and_excludes_products_launched_after_the_origin():
    Y = np.zeros((4, 40), np.float32)
    Y[0, 5] = 1  # product A, UK sells at block 6
    Y[2, 30] = 2  # product B, first sale at block 31
    pidx = np.array([0, 0, 1, 1])  # series 0/1 = product A (UK/INT), 2/3 = product B
    assert W.eligible_mask(Y, pidx, 20).tolist() == [
        True,
        True,
        False,
        False,
    ]  # both channels of A known; B is cold start
    assert W.eligible_mask(Y, pidx, 31).tolist() == [True, True, True, True]
    assert (
        W.eligible_mask(Y, pidx, 5).tolist() == [False] * 4
    )  # the sale at block 6 is not yet known at origin 5


def test_training_rows_exclude_products_that_have_not_launched(cfg):
    Y = _Y(n=20, weeks=60)
    Y[0, :40] = 0  # product 0 launches at block 41
    pidx = np.arange(20)
    chan = np.zeros(20, np.int8)
    _, y, _ = W.training_set(Y, _starts(60), chan, pidx, 50, W.horizons(cfg))
    full = sum(1 for tp in range(W.MIN_ORIGIN, 50) for h in W.horizons(cfg) if tp + h <= 50) * 20
    assert len(y) < full  # product 0's rows before its launch are absent


def test_seasonal_naive_picks_the_same_block_last_year(cfg):
    Y = np.arange(80, dtype=np.float32)[None, :].repeat(3, axis=0)
    f = W.baseline_forecasts(Y, 65, cfg)["seasonal_naive52"]
    np.testing.assert_array_equal(f[0], np.array([65 + h - 53 for h in W.horizons(cfg)], np.float32))


def test_croston_tsb_constant_and_dead_series():
    H = np.zeros((3, 40), np.float32)
    H[0, :] = 5.0
    H[1, ::4] = 8.0
    cro, tsb = W._croston_tsb(H)
    assert abs(cro[0, 0] - 5.0 * 0.95) < 0.3
    assert 8.0 * 0.95 / 4 - 0.6 < cro[1, 0] < 8.0 * 0.95 / 4 + 0.6
    assert cro[2, 0] == 0.0 and tsb[2, 0] == 0.0 and tsb[0, 0] > 4.0


def test_zero_denominators_are_excluded_not_replaced_by_epsilon():
    df = pd.DataFrame(
        {
            "model": "m",
            "origin": 1,
            "h": 1,
            "actual": [0.0, 2.0],
            "forecast": [1.0, 1.0],
            "scale1": [0.0, 2.0],  # constant series -> undefined MASE
            "scale2": [0.0, 2.0],
        }
    )
    t = W.metrics_by(df, ["model"]).iloc[0]
    assert t["n_mase_excluded"] == 1 and t["n_mase_scored"] == 1
    assert t["mase"] == pytest.approx(1 / 2) and t["rmsse"] == pytest.approx(np.sqrt(1 / 4))


def test_all_zero_wape_is_one_and_aggregates_are_reported(cfg):
    Y = _Y(n=40, weeks=60)
    chan = (np.arange(40) % 2).astype(np.int8)
    pidx = np.arange(40)
    res = {t: W.run_origin(Y, _starts(60), chan, pidx, t, cfg, with_lgbm=False) for t in (50, 54)}
    df = W.long_frame(res, cfg)
    t = W.metrics_by(df, ["model"])
    assert abs(t.loc["all_zero", "wape"] - 1.0) < 1e-9
    assert set(BASE_NAMES) <= set(t.index)
    lv = W.aggregate_levels(df)
    assert ("mean4", "L3_total") in lv.index and ("mean4", "L2_channel") in lv.index


def test_crossing_is_reported_before_the_correction_and_then_removed():
    q = np.array([[[3.0, 2.0, 5.0], [1.0, 2.0, 3.0]]], np.float32)  # the first row crosses
    assert W.crossing_rate(q) == 0.5
    f = W.fix_crossing(q)
    assert W.crossing_rate(f) == 0.0 and (f >= 0).all()


def test_lgbm_weekly_trains_predicts_and_is_scored(cfg):
    c = copy.deepcopy(cfg)
    c["models"]["lgbm"].update({"num_boost_round": 30, "quantile_rounds": 15})
    c["models"]["lgbm"]["params"].update({"min_data_in_leaf": 20, "num_threads": 2})
    Y = _Y(n=50, weeks=60)
    chan = np.zeros(50, np.int8)
    pidx = np.arange(50)
    res = {t: W.run_origin(Y, _starts(60), chan, pidx, t, c) for t in (46, 50)}
    W.calibrate_previous(res, c)
    r = res[50]
    assert np.isfinite(r["point"]["lgbm_w"]).all() and (r["point"]["lgbm_w"] >= 0).all()
    assert (np.diff(r["quant"]["lgbm_w"], axis=2) >= -1e-9).all()
    assert "lgbm_w_cal" in r["quant"] and (r["quant"]["lgbm_w_cal"][:, :, 0] >= 0).all()
    d = W.interval_diagnostics(res, c)
    assert {"lgbm_w", "lgbm_w_cal", "empirical_q"} <= set(d["model"])


def _tabs(wape, mase, bias):
    models = list(wape)
    L1 = pd.DataFrame({"wape": wape, "mase": mase, "bias_pct": bias, "demand": {m: 1000.0 for m in models}})
    po = pd.DataFrame([{"model": m, "origin": o, "bias_pct": bias[m]} for m in models for o in range(10)])
    segs = [
        {"model": m, "segment_type": "volume", "segment": s, "wape": wape[m], "demand": 500.0}
        for m in models
        for s in ("high", "low")
    ]
    cov = pd.DataFrame({"coverage": {m: 1.0 for m in models}})
    return {"L1": L1, "per_origin": po, "segments": pd.DataFrame(segs), "forecast_coverage": cov}


def _meta(models, train=1.0, batch=1.0):
    return {"train_seconds": {m: train for m in models}, "batch_seconds": {m: batch for m in models}}


META = _meta(["naive", "mean4", "mean13", "seasonal_naive52", "croston_sba", "tsb", "lgbm_w", "all_zero"])
BASE_WAPE = {
    "naive": 0.98,
    "mean4": 0.85,
    "mean13": 0.86,
    "seasonal_naive52": 1.3,
    "croston_sba": 1.3,
    "tsb": 0.9,
}


def test_selection_prefers_a_baseline_when_the_ml_model_fails_a_guardrail(cfg):
    wape = {**BASE_WAPE, "all_zero": 1.0, "lgbm_w": 0.80}
    mase = {m: 2.4 for m in wape} | {"lgbm_w": 3.5}  # scaled error far worse than the best baseline
    sel = W.select_model(_tabs(wape, mase, {m: 0.0 for m in wape}), cfg, META)
    assert sel["strongest_baseline"] == "mean4" and sel["selected"] == "mean4"
    failed = [c["check"] for c in sel["candidates"]["lgbm_w"]["checks"] if not c["passed"]]
    assert failed == ["mase_guard"]


def test_selection_accepts_ml_only_with_every_guardrail_and_enough_gain_and_no_hidden_opposing_bias(cfg):
    ok = {**BASE_WAPE, "all_zero": 1.0, "lgbm_w": 0.80}
    mase = {m: 2.4 for m in ok}
    bias = {m: 0.0 for m in ok}
    assert W.select_model(_tabs(ok, mase, bias), cfg, META)["selected"] == "lgbm_w"
    small = {**ok, "lgbm_w": 0.848}  # < 1% gain over mean4
    assert W.select_model(_tabs(small, mase, bias), cfg, META)["selected"] == "mean4"
    t = _tabs(ok, mase, bias)  # pooled bias 0, but origins at +-40%
    m = t["per_origin"]["model"] == "lgbm_w"
    t["per_origin"].loc[m & (t["per_origin"].origin < 5), "bias_pct"] = 0.4
    t["per_origin"].loc[m & (t["per_origin"].origin >= 5), "bias_pct"] = -0.4
    s = W.select_model(t, cfg, META)
    assert s["selected"] == "mean4" and not s["candidates"]["lgbm_w"]["qualified"]


def test_nobody_is_selected_if_no_candidate_qualifies(cfg):
    wape = {m: 1.2 for m in BASE_NAMES} | {"all_zero": 1.0, "lgbm_w": 1.1}
    s = W.select_model(_tabs(wape, {m: 2.0 for m in wape}, {m: 0.0 for m in wape}), cfg, META)
    assert s["selected"] is None


def test_weekly_features_do_not_depend_on_prices():
    assert not any(
        "price" in f for f in W.FEATURES
    )  # the daily price look-ahead defect cannot affect the weekly study


def test_training_refuses_when_history_is_too_short(cfg):
    Y = _Y(n=30, weeks=60)
    with pytest.raises(ValueError, match="not enough history"):
        W.WeeklyLGBM.fit(Y, _starts(60), np.zeros(30, np.int8), np.arange(30), 28, cfg)


def test_runtime_not_measured_is_not_a_pass(cfg):
    wape = {**BASE_WAPE, "all_zero": 1.0, "lgbm_w": 0.80}
    mase = {m: 2.4 for m in wape}
    s = W.select_model(_tabs(wape, mase, {m: 0.0 for m in wape}), cfg, {})  # no timings at all
    assert s["selected"] is None
    c = s["candidates"]["mean4"]["checks"]
    assert [x for x in c if x["check"] == "runtime"][0]["passed"] is False
    assert "not measured" in [x for x in c if x["check"] == "runtime"][0]["detail"]


def test_runtime_limits_are_enforced_when_measured(cfg):
    wape = {**BASE_WAPE, "all_zero": 1.0, "lgbm_w": 0.80}
    mase = {m: 2.4 for m in wape}
    slow = _meta(list(wape))
    slow["train_seconds"]["lgbm_w"] = cfg["selection"]["max_train_seconds_per_origin"] + 1
    s = W.select_model(_tabs(wape, mase, {m: 0.0 for m in wape}), cfg, slow)
    assert s["selected"] == "mean4" and not s["candidates"]["lgbm_w"]["qualified"]


def test_forecast_coverage_is_measured_not_assumed(cfg):
    wape = {**BASE_WAPE, "all_zero": 1.0, "lgbm_w": 0.80}
    mase = {m: 2.4 for m in wape}
    t = _tabs(wape, mase, {m: 0.0 for m in wape})
    t["forecast_coverage"].loc["lgbm_w", "coverage"] = 0.97  # 3% of rows without a finite forecast
    s = W.select_model(t, cfg, META)
    assert not s["candidates"]["lgbm_w"]["qualified"] and s["selected"] == "mean4"
    del t["forecast_coverage"]  # no measurement -> nobody can pass
    assert W.select_model(t, cfg, META)["selected"] is None


def test_run_measures_coverage_and_runtime_for_every_candidate(cfg):
    Y = _Y(n=40, weeks=60)
    chan = (np.arange(40) % 2).astype(np.int8)
    r = W.run_origin(Y, _starts(60), chan, np.arange(40), 50, cfg, with_lgbm=False)
    assert r["meta"]["baseline_batch_seconds"] >= 0
    f = r["point"]["naive"]
    assert np.isfinite(f).all() and f.shape == (len(r["rows"]), 4)
