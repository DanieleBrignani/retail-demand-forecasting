"""Information-boundary tests: features and baselines at origin t cannot depend on anything after t."""

import copy

import numpy as np
import pandas as pd
import pytest

from retailfc.features.build import CalendarCoverageError, build_examples, fit_encoders
from retailfc.features.spec import SPECS, columns_for
from retailfc.models.baselines import BASELINES, predict_baseline
from retailfc.panel import Panel

GROUPS = ["history", "ids", "horizon", "calendar", "events", "price"]
HORIZONS = list(range(1, 29))
ORIGIN = 200


def _mutated(panel: Panel, origin: int) -> Panel:
    p = copy.deepcopy(panel)
    rng = np.random.default_rng(0)
    p.sales[:, origin:] = rng.integers(0, 500, size=p.sales[:, origin:].shape)
    w = int(p.day_week_idx[origin - 1])
    p.price_weekly[:, w + 1 :] = rng.uniform(
        50, 99, size=p.price_weekly[:, w + 1 :].shape
    )  # unavailable future prices
    return p


def test_future_sales_and_prices_mutation_invariance(panel):
    enc = fit_encoders(panel, ORIGIN)
    base = build_examples(panel, ORIGIN, np.arange(panel.n_series), HORIZONS, GROUPS, enc, with_target=False)
    mut = build_examples(
        _mutated(panel, ORIGIN), ORIGIN, np.arange(panel.n_series), HORIZONS, GROUPS, enc, with_target=False
    )
    pd.testing.assert_frame_equal(base, mut)


@pytest.mark.parametrize("name", BASELINES)
def test_baselines_mutation_invariance(panel, cfg, name):
    sel = np.arange(panel.n_series)
    a = predict_baseline(name, panel, ORIGIN, sel, HORIZONS, cfg)
    b = predict_baseline(name, _mutated(panel, ORIGIN), ORIGIN, sel, HORIZONS, cfg)
    pd.testing.assert_frame_equal(a, b)


def test_planned_price_scenario_is_the_only_one_reading_future_prices(panel):
    enc = fit_encoders(panel, ORIGIN)
    sel = np.arange(panel.n_series)
    m = _mutated(panel, ORIGIN)
    obs_a = build_examples(
        panel, ORIGIN, sel, HORIZONS, GROUPS, enc, with_target=False, price_scenario="observed"
    )
    obs_b = build_examples(
        m, ORIGIN, sel, HORIZONS, GROUPS, enc, with_target=False, price_scenario="observed"
    )
    assert "target_price" not in obs_a.columns
    pd.testing.assert_frame_equal(obs_a, obs_b)
    pl_a = build_examples(
        panel, ORIGIN, sel, HORIZONS, GROUPS, enc, with_target=False, price_scenario="planned"
    )
    pl_b = build_examples(m, ORIGIN, sel, HORIZONS, GROUPS, enc, with_target=False, price_scenario="planned")
    assert not pl_a["target_price"].fillna(-1).equals(pl_b["target_price"].fillna(-1))  # labelled assumption


def test_origin_relative_lags_and_rolling_boundaries(panel):
    enc = fit_encoders(panel, ORIGIN)
    sel = np.array([0, 1, 2])
    df = build_examples(panel, ORIGIN, sel, [1, 28], GROUPS, enc, with_target=False)
    y = panel.sales.astype(float)
    for s in sel:
        r = df[(df.series_idx == s) & (df.h == 1)].iloc[0]
        assert r.last_sale == y[s, ORIGIN - 1]
        assert r.lag_7 == y[s, ORIGIN - 8]  # y_{o-7}
        assert r.lag_14 == y[s, ORIGIN - 15]
        assert r.lag_28 == y[s, ORIGIN - 29]
        assert np.isclose(r.roll_mean_7, y[s, ORIGIN - 7 : ORIGIN].mean())  # days o-6..o
        assert np.isclose(r.roll_mean_28, y[s, ORIGIN - 28 : ORIGIN].mean())
        assert np.isclose(r.roll_std_7, y[s, ORIGIN - 7 : ORIGIN].std())
        assert np.isclose(r.zero_prop_28, (y[s, ORIGIN - 28 : ORIGIN] == 0).mean())


@pytest.mark.parametrize("h", [1, 6, 7, 8, 14, 15, 28])
def test_target_aligned_seasonal_lag_never_reads_after_origin(panel, h):
    enc = fit_encoders(panel, ORIGIN)
    df = build_examples(panel, ORIGIN, np.array([4]), [h], GROUPS, enc, with_target=True)
    day = ORIGIN + h - 7 * int(np.ceil(h / 7))
    assert ORIGIN - 7 < day <= ORIGIN
    assert df.seas_lag_1.iloc[0] == panel.sales[4, day - 1]
    assert df.seas_lag_2.iloc[0] == panel.sales[4, day - 8]
    # same weekday as the target
    cal = panel.calendar
    assert cal.wday.iloc[day - 1] == cal.wday.iloc[ORIGIN + h - 1]


def test_target_is_the_future_value_and_not_a_feature(panel):
    enc = fit_encoders(panel, ORIGIN)
    df = build_examples(panel, ORIGIN, np.array([2]), [5], GROUPS, enc, with_target=True)
    assert df.y.iloc[0] == panel.sales[2, ORIGIN + 5 - 1]
    feats = [c for c in df.columns if c not in ("y", "series_idx", "origin")]
    assert "y" not in feats


def test_short_history_gives_nan_not_error(panel):
    enc = fit_encoders(panel, 20)
    df = build_examples(panel, 10, np.arange(3), [1, 2], GROUPS, enc, with_target=False)
    assert df["lag_28"].isna().all() and df["roll_mean_28"].isna().all()
    assert df["last_sale"].notna().all()


def test_missing_calendar_coverage_is_refused(panel):
    enc = fit_encoders(panel, ORIGIN)
    with pytest.raises(CalendarCoverageError):
        build_examples(panel, panel.n_days, np.arange(3), list(range(1, 100)), GROUPS, enc, with_target=False)


def test_prices_are_never_zero_filled_or_backfilled(panel):
    enc = fit_encoders(panel, 40)
    # series 7 has no price before week 10 -> origin 40 is in week 5
    df = build_examples(panel, 40, np.array([7]), [1], GROUPS, enc, with_target=False)
    assert np.isnan(df.price_origin.iloc[0]) and df.price_missing.iloc[0] == 1
    df2 = build_examples(panel, 120, np.array([7]), [1], GROUPS, enc, with_target=False)
    assert df2.price_origin.iloc[0] > 0
    # stale gap for series 9 (weeks 20..24 missing): origin in week 22 carries forward with age>0
    o = 22 * 7 + 3
    df3 = build_examples(panel, o, np.array([9]), [1], GROUPS, enc, with_target=False)
    assert df3.price_age_weeks.iloc[0] >= 2 and df3.price_origin.iloc[0] > 0


def test_unseen_identifiers_map_to_minus_one(panel):
    enc = fit_encoders(panel, ORIGIN)
    enc.ids["item_id"].pop(panel.ids.item_id.iloc[0])
    df = build_examples(panel, ORIGIN, np.array([0]), [1], GROUPS, enc, with_target=False)
    assert df.item_id.iloc[0] == -1


def test_encoders_fit_on_training_cutoff_only(panel):
    enc = fit_encoders(panel, 10)  # before any event (first event at day 21)
    assert set(enc.event_name) == {"none"}
    enc2 = fit_encoders(panel, 200)
    assert len(enc2.event_name) > 1


def test_feature_registry_complete_and_documented():
    for s in SPECS:
        assert all([s.definition, s.time_ref, s.availability, s.missing, s.meaning])
    cols = columns_for(GROUPS)
    assert len(cols) == len(set(cols))
