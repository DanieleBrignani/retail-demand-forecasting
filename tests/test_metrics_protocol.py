import numpy as np
import pandas as pd
import pytest

from retailfc import protocol
from retailfc.backtesting import metrics as M


def _df(a, f, series=None, origin=1, td=None):
    n = len(a)
    return pd.DataFrame(
        {
            "actual": a,
            "forecast": f,
            "series_idx": series if series is not None else np.zeros(n, int),
            "origin": origin,
            "target_day": td if td is not None else np.arange(n),
        }
    )


def test_metric_formulas_hand_computed():
    df = _df([0, 2, 4, 0], [1, 1, 6, 0])
    m = M.pooled(df).iloc[0]
    e = np.array([1, -1, 2, 0])
    assert m.mae == pytest.approx(1.0)
    assert m.rmse == pytest.approx(np.sqrt((1 + 1 + 4 + 0) / 4))
    assert m.wape == pytest.approx(4 / 6)  # sum|e| / sum|a|
    assert m.bias == pytest.approx(e.mean())  # forecast - actual
    assert m.bias_pct == pytest.approx(e.sum() / 6)
    # smape: terms 200*1/1=200, 200*1/3, 200*2/10=40, both zero -> 0
    assert m.smape == pytest.approx((200 + 200 / 3 + 40 + 0) / 4)
    assert m.n_both_zero == 1


def test_zero_actuals_wape_is_undefined_not_zero_or_inf():
    m = M.pooled(_df([0, 0, 0], [1, 0, 2])).iloc[0]
    assert np.isnan(m.wape) and m.wape_undefined == 1
    assert np.isnan(m.bias_pct)
    assert m.mae == pytest.approx(1.0)


def test_all_zero_exact_forecast_gives_zero_errors():
    m = M.pooled(_df([0, 0], [0, 0])).iloc[0]
    assert m.mae == 0 and m.smape == 0 and np.isnan(m.wape)


def test_over_forecast_has_positive_bias():
    assert M.pooled(_df([1, 1], [3, 3])).iloc[0].bias > 0


def test_pooled_vs_per_series_vs_aggregated_differ():
    # series 0 high volume small relative error; series 1 low volume large relative error
    df = pd.concat(
        [
            _df([100, 100], [90, 110], series=np.array([0, 0]), td=np.array([1, 2])),
            _df([1, 1], [3, 3], series=np.array([1, 1]), td=np.array([1, 2])),
        ],
        ignore_index=True,
    )
    pooled = M.pooled(df).iloc[0].wape
    per = M.per_series_mean(df).iloc[0].wape
    agg = M.aggregated(df).iloc[0].wape
    assert pooled == pytest.approx((10 + 10 + 2 + 2) / 202)
    assert per == pytest.approx((0.1 + 2.0) / 2)
    assert agg == pytest.approx((8 + 12) / 202)  # day1: 93 vs 101, day2: 113 vs 101
    assert pooled != per


def test_per_series_excludes_undefined_and_counts_them():
    df = pd.concat(
        [_df([0, 0], [1, 1], series=np.array([0, 0])), _df([2, 2], [1, 1], series=np.array([1, 1]))]
    )
    r = M.per_series_mean(df).iloc[0]
    assert r.n_series == 2 and r.n_series_wape_undefined == 1 and r.wape == pytest.approx(0.5)


def test_origins_are_chronological_non_overlapping_and_holdout_last(cfg):
    o = protocol.make_origins(1913, cfg)
    assert o.holdout == 1885 and list(o.validation) == [1801, 1829, 1857]
    protocol.check_non_overlap(o, 28)
    assert o.holdout + 28 == 1913
    assert all(b - a == 28 for a, b in zip(o.validation, (*o.validation[1:], o.holdout), strict=False))


def test_training_origins_never_use_targets_after_backtest_origin(cfg):
    for bo in (1801, 1857, 1885):
        for gap in (0, 28):
            tr = protocol.training_origins(bo, cfg, gap=gap)
            assert max(tr) + 28 <= bo - gap
            assert min(tr) >= cfg["protocol"]["min_history_days"]
    # early-stopping origin (bo-28) targets end at bo and no training target window overlaps it
    tr = protocol.training_origins(1857, cfg, gap=28)
    assert max(tr) + 28 <= 1857 - 28


def test_not_enough_history_is_rejected(cfg):
    with pytest.raises(ValueError):
        protocol.make_origins(100, cfg)


def test_protocol_lock_and_single_holdout_evaluation(cfg, tmp_env):
    from retailfc import config as C

    with pytest.raises(RuntimeError):
        protocol.assert_protocol_locked(cfg, "v1")  # not locked yet
    protocol.lock_protocol(cfg, "v1", 1913)
    protocol.assert_protocol_locked(cfg, "v1")
    changed = C.override(cfg, **{"lightgbm.num_boost_round": 5})
    with pytest.raises(RuntimeError):
        protocol.assert_protocol_locked(changed, "v1")  # config changed after lock
    protocol.record_holdout_evaluation({"x": 1})
    with pytest.raises(RuntimeError):
        protocol.record_holdout_evaluation({"x": 2})  # holdout may be consumed once
