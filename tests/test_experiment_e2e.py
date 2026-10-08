"""End-to-end on SYNTHETIC fixture data (mechanics only; numbers are meaningless as benchmarks)."""

import numpy as np
import pandas as pd
import pytest

from retailfc import config as C
from retailfc import experiments as E
from retailfc import protocol as P


@pytest.fixture(scope="module")
def small_cfg():
    cfg = C.load_config()
    cfg = C.override(
        cfg,
        **{
            "lightgbm.num_boost_round": 25,
            "lightgbm.early_stopping_rounds": 5,
            "lightgbm.tuning.n_trials": 2,
            "statistical.subset_size": 6,
            "lightgbm.params.min_data_in_leaf": 20,
            "lightgbm.params.num_threads": 2,
        },
    )
    return cfg


@pytest.fixture(scope="module")
def exp(small_cfg, panel, tmp_path_factory):
    import os

    d = tmp_path_factory.mktemp("art")
    os.environ["RETAILFC_ARTIFACTS_DIR"] = str(d)
    meta = E.run_validation(panel, small_cfg)
    sel = E.select_feature_set(small_cfg, meta["experiment_id"])
    return meta, sel, d


@pytest.mark.slow
def test_validation_runs_all_models_on_same_origins_series_and_actuals(exp, small_cfg, panel):
    meta, sel, d = exp
    out = E.experiment_dir(meta["experiment_id"]) / "validation" / "predictions"
    names = sorted(p.stem for p in out.glob("*.parquet"))
    assert {
        "naive",
        "seasonal_naive",
        "mean",
        "seasonal_mean",
        "sarima",
        "lgbm_A",
        "lgbm_B",
        "lgbm_C",
        "lgbm_D",
    } <= set(names)
    ref = pd.read_parquet(out / "naive.parquet")
    for n in ("lgbm_D", "seasonal_naive"):
        p = pd.read_parquet(out / f"{n}.parquet")
        a = ref.set_index(["origin", "series_idx", "h"])["actual"]
        b = p.set_index(["origin", "series_idx", "h"])["actual"]
        assert a.index.equals(b.index) and np.allclose(a.to_numpy(), b.to_numpy())
    sar = pd.read_parquet(out / "sarima.parquet")
    assert sar.series_idx.nunique() <= 6  # restricted subset only
    assert set(pd.read_parquet(out / "lgbm_D.parquet").origin.unique()) == set(meta["origins"]["validation"])


@pytest.mark.slow
def test_metric_records_have_required_schema_and_reproduce_from_predictions(exp):
    meta, _, _ = exp
    v = E.experiment_dir(meta["experiment_id"]) / "validation"
    met = pd.read_parquet(v / "metrics.parquet")
    required = {
        "backtest_id",
        "experiment_id",
        "dataset_version",
        "feature_version",
        "forecast_origin",
        "horizon",
        "model_name",
        "model_version",
        "evaluation_scope",
        "segment_type",
        "segment_value",
        "metric",
        "value",
        "observation_count",
    }
    assert required <= set(met.columns)
    pr = pd.read_parquet(v / "predictions" / "naive.parquet")
    wape = (pr.forecast - pr.actual).abs().sum() / pr.actual.abs().sum()
    row = met[
        (met.model_name == "naive")
        & (met.aggregation == "pooled")
        & (met.metric == "wape")
        & (met.segment_type == "all")
        & (met.horizon == "1-28")
        & met.forecast_origin.isna()
        & (met.evaluation_scope == "full_panel")
    ]
    assert len(row) == 1 and row.value.iloc[0] == pytest.approx(wape, rel=1e-5)
    # restricted-subset scope exists for non-SARIMA models too
    assert ((met.evaluation_scope == "restricted_subset") & (met.model_name == "naive")).any()


@pytest.mark.slow
def test_training_never_sees_targets_after_origin(small_cfg, panel):
    """Mutating sales after a validation origin must not change that origin's LightGBM forecast."""
    import copy

    from retailfc.models.lgbm import GlobalLGBM, feature_groups_for

    o = 188
    g = feature_groups_for(small_cfg, "D")
    sel = np.arange(panel.n_series)
    m1 = GlobalLGBM.fit(panel, o, small_cfg, g, series_idx=sel)
    p2 = copy.deepcopy(panel)
    p2.sales[:, o:] = 999
    p2.price_weekly[:, int(p2.day_week_idx[o - 1]) + 1 :] = 77.0
    m2 = GlobalLGBM.fit(p2, o, small_cfg, g, series_idx=sel)
    f1 = m1.predict(panel, o, sel, [1, 14, 28])
    f2 = m2.predict(p2, o, sel, [1, 14, 28])
    np.testing.assert_allclose(f1.forecast.to_numpy(), f2.forecast.to_numpy(), rtol=1e-5)


@pytest.mark.slow
def test_serialization_round_trip(small_cfg, panel, tmp_path):
    from retailfc.models.lgbm import GlobalLGBM, feature_groups_for

    sel = np.arange(panel.n_series)
    m = GlobalLGBM.fit(panel, 188, small_cfg, feature_groups_for(small_cfg, "C"), series_idx=sel)
    m.save(tmp_path / "m")
    m2 = GlobalLGBM.load(tmp_path / "m")
    a = m.predict(panel, 200, sel, [1, 7, 28])
    b = m2.predict(panel, 200, sel, [1, 7, 28])
    np.testing.assert_allclose(a.forecast, b.forecast, rtol=1e-6)
    assert m2.feature_version == m.feature_version


@pytest.mark.slow
def test_holdout_requires_lock_and_runs_once(exp, small_cfg, panel):
    meta, sel, d = exp
    with pytest.raises(RuntimeError, match="not locked"):
        E.run_holdout(panel, small_cfg)
    P.lock_protocol(small_cfg, panel.dataset_version, panel.n_days)
    res = E.run_holdout(panel, small_cfg)
    assert res["holdout_origin"] == panel.n_days - 28
    with pytest.raises(RuntimeError, match="already evaluated"):
        E.run_holdout(panel, small_cfg)
    h = E.experiment_dir(meta["experiment_id"]) / "holdout" / "predictions"
    assert (h / "lgbm_selected.parquet").exists()
    # holdout origin is disjoint from every validation target window
    v_last_target_end = max(meta["origins"]["validation"]) + 28
    assert res["holdout_origin"] >= v_last_target_end


def test_sarima_failure_falls_back_and_is_recorded(small_cfg, panel):
    from retailfc.models.stats import SarimaForecaster

    sar = SarimaForecaster(small_cfg, np.array([3, 5]))  # 3: late launch, 5: all-zero series
    out = sar.predict(panel, 188, np.array([3, 5]), [1, 2, 3])
    assert (out[out.series_idx == 5].fallback_used == "sarima_all_zero_history").all()
    assert (out[out.series_idx == 5].forecast == 0).all()
    assert len(sar.records) == 2 and any(r["failed"] for r in sar.records)
    assert (out.forecast >= 0).all()
