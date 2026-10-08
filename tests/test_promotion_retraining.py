import pytest

from retailfc import config as C
from retailfc.orchestration.retraining import decide
from retailfc.registry.promotion import Evidence, evaluate_promotion

RULES = C.load_config()["promotion"]
POL = C.load_config()["retraining"]


def ev(**kw):
    base = dict(
        wape=0.60,
        snaive_wape=0.70,
        bias_pct=0.02,
        coverage=0.78,
        segment_wape={"volume:low": 0.9, "volume:high": 0.5},
        train_seconds=100,
        batch_seconds=10,
    )
    base.update(kw)
    return Evidence(**base)


def test_promote_when_all_rules_pass():
    r = evaluate_promotion(ev(wape=0.60), ev(wape=0.65), RULES)
    assert r["promote"] and not r["failed"]


def test_newer_but_not_better_is_not_promoted():
    r = evaluate_promotion(ev(wape=0.648), ev(wape=0.65), RULES)  # 0.3% better < 1% required
    assert not r["promote"] and "improves_on_incumbent" in r["failed"]


def test_must_beat_seasonal_naive():
    r = evaluate_promotion(ev(wape=0.69, snaive_wape=0.70), ev(wape=0.80), RULES)
    assert "beats_seasonal_naive" in r["failed"]


def test_segment_degradation_blocks_promotion():
    chal = ev(wape=0.60, segment_wape={"volume:low": 1.0, "volume:high": 0.4})
    inc = ev(wape=0.65, segment_wape={"volume:low": 0.9, "volume:high": 0.5})
    assert "segment_degradation_limit" in evaluate_promotion(chal, inc, RULES)["failed"]


@pytest.mark.parametrize(
    "kw,failed",
    [
        ({"bias_pct": 0.2}, "bias_limit"),
        ({"bias_pct": -0.2}, "bias_limit"),
        ({"coverage": 0.4}, "interval_coverage"),
        ({"coverage": None}, "interval_coverage"),
        ({"train_seconds": 1e9}, "train_runtime_limit"),
        ({"batch_seconds": 1e9}, "batch_runtime_limit"),
    ],
)
def test_other_gates(kw, failed):
    assert failed in evaluate_promotion(ev(**kw), ev(wape=0.70), RULES)["failed"]


def test_bootstrap_first_model_only_needs_baseline_and_gates():
    assert evaluate_promotion(ev(), None, RULES)["promote"]
    assert not evaluate_promotion(ev(wape=0.69), None, RULES)["promote"]


def test_undefined_wape_never_promotes():
    assert not evaluate_promotion(ev(wape=float("nan")), None, RULES)["promote"]


BASE = dict(
    lock_held=False,
    missing_actuals_fraction=0.0,
    actuals_conflicts=0,
    feature_missing_max=0.05,
    days_since_last_training=60,
    model_age_days=60,
    matured_obs=100000,
    current_wape=0.70,
    reference_wape=0.60,
    new_actual_days=28,
    recent_drift_psi=[0.0, 0.0, 0.0],
)


def d(**kw):
    return decide({**BASE, **kw}, POL)


def test_healthy_model_is_not_retrained_just_because_time_passed():
    r = d(current_wape=0.61, model_age_days=100)
    assert r.decision == "no_retrain"


def test_performance_degradation_triggers_retrain():
    r = d()
    assert r.decision == "retrain" and r.triggers[0].startswith("performance_degradation")


def test_cooldown_blocks():
    assert d(days_since_last_training=3).decision == "no_retrain"


def test_not_enough_new_data_blocks_even_with_degradation():
    r = d(new_actual_days=3)
    assert r.decision == "no_retrain" and "insufficient new data" in " ".join(r.reasons)


def test_missing_actuals_trigger_data_quality_block_not_retrain():
    assert d(missing_actuals_fraction=0.5).decision == "blocked_data_quality"
    assert d(actuals_conflicts=2).decision == "blocked_data_quality"
    assert d(feature_missing_max=0.9).decision == "blocked_data_quality"


def test_too_few_matured_forecasts_means_performance_unassessed():
    r = d(matured_obs=10)
    assert r.decision == "no_retrain" and "not assessable" in " ".join(r.reasons)


def test_drift_alone_investigates_but_never_retrains():
    r = d(current_wape=0.60, recent_drift_psi=[0.4, 0.5, 0.6])
    assert r.decision == "investigate"
    assert d(current_wape=0.60, recent_drift_psi=[0.4, 0.0, 0.6]).decision == "no_retrain"  # not persistent


def test_governance_age_triggers_reassessment():
    r = d(current_wape=0.60, model_age_days=400, days_since_last_training=400)
    assert r.decision == "retrain" and r.triggers[0].startswith("governance_age")


def test_concurrent_run_is_skipped():
    assert d(lock_held=True).decision == "skipped_locked"


# ---------------------------------------------------------------- drift indicator on zero-inflated demand
def _zero_inflated(rng, n, p_zero, scale):
    import numpy as np

    x = np.round(rng.lognormal(1.0, 1.0, n) * scale)
    x[rng.random(n) < p_zero] = 0
    return x


def test_psi_is_not_blind_on_zero_inflated_demand():
    """Regression: with ~90% zeros the quantile bins of the raw reference collapse into one bin and PSI was 0 for ANY shift."""
    import numpy as np

    from retailfc.monitoring.core import psi

    rng = np.random.default_rng(0)
    ref = _zero_inflated(rng, 100_000, 0.90, 1.0)
    same = _zero_inflated(rng, 100_000, 0.90, 1.0)
    fewer_zeros = _zero_inflated(rng, 100_000, 0.70, 1.0)
    bigger_orders = _zero_inflated(rng, 100_000, 0.90, 3.0)
    assert psi(ref, same) < 0.01
    assert psi(ref, fewer_zeros) > 0.1  # the zero share changed a lot
    assert psi(ref, bigger_orders) > 0.05  # same zero share, positives shifted
    assert psi(ref, fewer_zeros) > 10 * psi(ref, same)


def test_psi_dense_data_unchanged_and_small_samples_are_nan():
    import numpy as np

    from retailfc.monitoring.core import psi

    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.02
    assert psi(a, rng.normal(1, 1, 5000)) > 0.25
    assert np.isnan(psi(a[:10], a))


# ---------------------------------------------------------------- multi-window promotion evidence
def test_gate_judges_pooled_bias_but_shows_each_window():
    chal = ev(bias_pct=0.01, eval_origins=[1, 2, 3], window_bias_pct=[0.20, -0.18, 0.0])
    r = evaluate_promotion(chal, None, RULES)
    chk = next(c for c in r["checks"] if c["check"] == "bias_limit")
    assert (
        chk["passed"]
        and "3 window(s)" in chk["detail"]
        and "+20.0%" in chk["detail"]
        and "-18.0%" in chk["detail"]
    )


def test_a_model_biased_on_the_pooled_evidence_still_fails_even_if_one_window_is_fine():
    chal = ev(bias_pct=-0.45, eval_origins=[1, 2, 3], window_bias_pct=[-0.40, -0.55, -0.41])
    assert "bias_limit" in evaluate_promotion(chal, None, RULES)["failed"]


def test_build_evidence_windows_pools_rows_and_reports_each_window(panel, cfg):
    import numpy as np

    from retailfc.backtesting import runner as R
    from retailfc.models.baselines import predict_baseline
    from retailfc.registry.pipeline import build_evidence_windows

    class Stub:  # a "model" that forecasts like the naive baseline; no training needed
        def predict(self, pan, o, sel, horizons):
            return predict_baseline("naive", pan, o, sel, horizons, cfg)

    origins = [188, 216, 244]
    edges = R.edges_for(panel, origins[0], cfg)
    e = build_evidence_windows(panel, cfg, origins, lambda o: (Stub(), None), edges)
    assert e.eval_origin == 244 and e.eval_origins == origins and len(e.window_bias_pct) == 3
    assert e.n_obs == 3 * panel.n_series * 28 and e.coverage is None
    f_sum = a_sum = 0.0
    per = []
    for o in origins:
        f = predict_baseline("naive", panel, o, np.arange(panel.n_series), list(range(1, 29)), cfg)
        a = panel.sales[f.series_idx.to_numpy(), o + f.h.to_numpy() - 1].astype(float)
        per.append((f.forecast.sum() - a.sum()) / a.sum())
        f_sum, a_sum = f_sum + f.forecast.sum(), a_sum + a.sum()
    assert e.bias_pct == pytest.approx((f_sum - a_sum) / a_sum, rel=1e-5)
    assert e.window_bias_pct == pytest.approx(per, rel=1e-5)
    assert e.snaive_wape > 0
