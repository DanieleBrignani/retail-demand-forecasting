"""Explicit fallback / rejection behaviour for degraded inputs (see docs/limitations.md)."""

import numpy as np
import pytest

from retailfc import config as C
from retailfc.models import stats
from retailfc.models.lgbm import GlobalLGBM, feature_groups_for


@pytest.fixture(scope="module")
def small_cfg():
    return C.override(
        C.load_config(),
        **{
            "lightgbm.num_boost_round": 15,
            "lightgbm.early_stopping_rounds": 5,
            "lightgbm.params.min_data_in_leaf": 20,
            "lightgbm.params.num_threads": 2,
        },
    )


@pytest.fixture(scope="module")
def model(small_cfg, panel):
    return GlobalLGBM.fit(
        panel, 188, small_cfg, feature_groups_for(small_cfg, "D"), series_idx=np.arange(panel.n_series)
    )


def test_unseen_series_falls_back_to_seasonal_naive_and_is_flagged(model, panel):
    first_item = panel.ids.item_id.iloc[0]
    enc_before = dict(model.enc.ids["item_id"])
    try:
        model.enc.ids["item_id"].pop(first_item)  # simulate an item that did not exist at training time
        sel = np.arange(panel.n_series)
        out = model.predict(panel, 200, sel, [1, 7, 28])
        unseen_rows = out[panel.ids.item_id.iloc[out.series_idx].to_numpy() == first_item]
        assert len(unseen_rows) > 0 and (unseen_rows.fallback_used == "unseen_series").all()
        # fallback value is the seasonal-naive value, not a model output
        h7 = unseen_rows[unseen_rows.h == 7].iloc[0]
        assert h7.forecast == pytest.approx(panel.sales[h7.series_idx, 200 + 7 - 7 - 1])
        # other series are untouched by the fallback
        ok = out[panel.ids.item_id.iloc[out.series_idx].to_numpy() != first_item]
        assert not (ok.fallback_used == "unseen_series").any()
    finally:
        model.enc.ids["item_id"] = enc_before


def test_all_zero_history_series_forecasts_zero_with_flag(model, panel):
    out = model.predict(panel, 200, np.array([5]), [1, 14, 28])  # series 5 is all-zero in the fixture
    assert (out.forecast == 0).all() and (out.fallback_used == "all_zero_history").all()


def test_negative_raw_model_output_is_clipped_and_counted(model, panel, monkeypatch):
    real = model.booster.predict
    monkeypatch.setattr(model.booster, "predict", lambda X, *a, **k: np.full(len(X), -2.5))
    out = model.predict(panel, 200, np.array([0, 1]), [1, 2])
    assert (out.forecast >= 0).all() and (out.negative_raw == 1).all() and (out.raw < 0).all()
    monkeypatch.setattr(model.booster, "predict", real)


def test_nonfinite_model_output_uses_flagged_fallback(model, panel, monkeypatch):
    monkeypatch.setattr(model.booster, "predict", lambda X, *a, **k: np.full(len(X), np.nan))
    out = model.predict(panel, 200, np.array([0]), [1, 2, 3])
    assert (out.fallback_used == "nonfinite_model_output").all() and np.isfinite(out.forecast).all()


def test_sarima_fit_exception_falls_back_to_seasonal_naive_and_is_recorded(small_cfg, panel, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic convergence failure")

    monkeypatch.setattr(stats, "_fit_forecast", boom)
    sar = stats.SarimaForecaster(small_cfg, np.array([0, 1]))
    out = sar.predict(panel, 188, np.array([0, 1]), [1, 2, 3])
    assert (out.fallback_used == "sarima_fit_failed").all()
    assert all(r["failed"] for r in sar.records) and len(out) == 6
    from retailfc.models.baselines import predict_baseline

    sn = predict_baseline("seasonal_naive", panel, 188, np.array([0, 1]), [1, 2, 3], small_cfg)
    np.testing.assert_allclose(
        out.sort_values(["series_idx", "h"]).forecast, sn.sort_values(["series_idx", "h"]).forecast
    )


def test_baseline_fallback_flags(panel, small_cfg):
    from retailfc.models.baselines import predict_baseline

    out = predict_baseline(
        "mean", panel, 15, np.array([3, 5, 0]), [1], small_cfg
    )  # 3 launches late, 5 all-zero
    d = dict(zip(out.series_idx, out.fallback_used, strict=True))
    assert d[5] == "all_zero_history" and d[3] == "all_zero_history"  # series 3 has no sale before day 120
    assert d[0] in ("short_history", "")
