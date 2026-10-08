"""Level offset: forecasts scale with the series level instead of saturating at the training range (SYNTHETIC panel, mechanics only)."""

import numpy as np
import pytest

from retailfc import config as C
from retailfc.models.lgbm import GlobalLGBM, feature_groups_for
from retailfc.uncertainty import QuantileBundle

ORIGIN = 188


@pytest.fixture(scope="module")
def cfg_plain():
    return C.override(
        C.load_config(),
        **{
            "lightgbm.num_boost_round": 40,
            "lightgbm.early_stopping_rounds": 8,
            "lightgbm.params.min_data_in_leaf": 20,
            "lightgbm.params.num_threads": 2,
        },
    )


@pytest.fixture(scope="module")
def cfg_off(cfg_plain):
    return C.override(cfg_plain, **{"lightgbm.level_offset": True})


def _level(panel):
    return panel.sales[:, ORIGIN - 28 : ORIGIN].mean(axis=1)


def _split(panel):
    lv = _level(panel)
    low, high = np.where(lv < 3.0)[0], np.where(lv > 8.0)[0]
    assert len(low) > 10 and len(high) >= 4, (len(low), len(high))
    return low, high


def _bias(model, panel, series):
    f = model.predict(panel, ORIGIN, series, list(range(1, 29)))
    a = panel.sales[f.series_idx.to_numpy(), ORIGIN + f.h.to_numpy() - 1].astype(float)
    return (f.forecast.sum() - a.sum()) / a.sum()


def test_offset_extrapolates_to_levels_never_seen_in_training(panel, cfg_plain, cfg_off):
    """Train only on low-level series; forecast the high-level ones. Trees without an offset saturate near the training range."""
    low, high = _split(panel)
    g = feature_groups_for(cfg_plain, "A")
    plain = GlobalLGBM.fit(panel, ORIGIN, cfg_plain, g, series_idx=low)
    off = GlobalLGBM.fit(panel, ORIGIN, cfg_off, g, series_idx=low)
    b_plain, b_off = _bias(plain, panel, high), _bias(off, panel, high)
    assert b_plain < -0.4, b_plain  # the failure mode: severe under-forecast of unseen large series
    assert abs(b_off) < 0.4 and abs(b_off) < 0.5 * abs(b_plain), (b_off, b_plain)
    assert off.level_link == "log" and plain.level_link is None


def test_offset_forecasts_are_nonnegative_and_finite(panel, cfg_off):
    m = GlobalLGBM.fit(
        panel, ORIGIN, cfg_off, feature_groups_for(cfg_off, "D"), series_idx=np.arange(panel.n_series)
    )
    f = m.predict(panel, ORIGIN, np.arange(panel.n_series), [1, 14, 28])
    assert np.isfinite(f.forecast).all() and (f.forecast >= 0).all()


def test_offset_model_round_trips_through_serialization(panel, cfg_off, tmp_path):
    g = feature_groups_for(cfg_off, "C")
    m = GlobalLGBM.fit(panel, ORIGIN, cfg_off, g, series_idx=np.arange(panel.n_series))
    m.save(tmp_path / "m")
    m2 = GlobalLGBM.load(tmp_path / "m")
    assert m2.level_link == "log" and m2.level_eps == m.level_eps
    a = m.predict(panel, 200, np.arange(20), [1, 7, 28])
    b = m2.predict(panel, 200, np.arange(20), [1, 7, 28])
    np.testing.assert_allclose(a.forecast, b.forecast, rtol=1e-6)


def test_quantile_bundle_works_with_the_offset_and_is_ordered(panel, cfg_off, tmp_path):
    g = feature_groups_for(cfg_off, "A")
    sel = np.arange(panel.n_series)
    point = GlobalLGBM.fit(panel, ORIGIN, cfg_off, g, series_idx=sel)
    b = QuantileBundle.fit(panel, ORIGIN, cfg_off, g, sel, None, point, [0.1, 0.9])
    iv = b.apply(b.predict_raw(panel, ORIGIN, sel, [1, 7, 28]))
    assert (iv.lower <= iv.upper + 1e-9).all() and (iv.lower >= 0).all() and np.isfinite(iv.upper).all()
    b.save(tmp_path / "b")
    b2 = QuantileBundle.load(tmp_path / "b")
    iv2 = b2.apply(b2.predict_raw(panel, ORIGIN, sel, [1, 7, 28]))
    np.testing.assert_allclose(iv.upper, iv2.upper, rtol=1e-5)


def test_unsupported_objective_with_offset_is_refused(panel, cfg_off):
    bad = C.override(cfg_off, **{"lightgbm.params.objective": "regression"})
    with pytest.raises(ValueError, match="level_offset"):
        GlobalLGBM.fit(panel, ORIGIN, bad, feature_groups_for(bad, "A"), series_idx=np.arange(10))
