"""Objective, configurable challenger-promotion rules (defined in config before any final evaluation).

Defaults and rationale (docs/experiment_plan.md):
* min_primary_improvement_pct = 1.0   : WAPE differences below ~1% are within origin-to-origin noise on panels of thousands of series.
* max_vs_snaive_ratio = 0.97          : must beat the free seasonal-naive baseline by a margin, else complexity is unjustified.
* max_segment_degradation_pct = 5.0   : do not buy aggregate gains with large losses on a volume/intermittency/store segment.
* min_interval_coverage = 0.70        : nominal 80% interval; allow slack for temporal dependence, but not collapse.
* max_abs_bias_pct = 10.0             : systematic over/under-forecast beyond 10% of demand is operationally harmful.
* runtime limits                       : bound operational cost.
A newer model is never promoted merely for being newer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Evidence:
    wape: float
    snaive_wape: float
    bias_pct: float
    coverage: float | None = None
    segment_wape: dict[str, float] = field(default_factory=dict)  # "volume:low" -> WAPE ...
    train_seconds: float | None = None
    batch_seconds: float | None = None
    eval_origin: int | None = None
    n_obs: int | None = None
    eval_origins: list[int] | None = None  # all evaluation windows pooled into this evidence
    window_bias_pct: list[float] | None = None  # bias of each window on its own (informational)

    def to_dict(self) -> dict:
        return asdict(self)


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def evaluate_promotion(
    chal: Evidence, inc: Evidence | None, rules: dict, *, require_coverage: bool = True
) -> dict:
    checks = []
    nan = lambda x: x is None or x != x  # noqa: E731
    if nan(chal.wape) or nan(chal.snaive_wape):
        checks.append(_check("evidence_defined", False, "WAPE undefined for challenger or seasonal naive"))
    else:
        checks.append(_check("evidence_defined", True, ""))
        ratio = chal.wape / chal.snaive_wape if chal.snaive_wape else float("inf")
        checks.append(
            _check(
                "beats_seasonal_naive",
                ratio <= rules["max_vs_snaive_ratio"],
                f"wape/snaive={ratio:.4f} (max {rules['max_vs_snaive_ratio']})",
            )
        )
        if inc is None or nan(inc.wape):
            checks.append(
                _check(
                    "improves_on_incumbent",
                    True,
                    "no incumbent: bootstrap promotion allowed on baseline check",
                )
            )
        else:
            imp = (inc.wape - chal.wape) / inc.wape * 100 if inc.wape else 0.0
            checks.append(
                _check(
                    "improves_on_incumbent",
                    imp >= rules["min_primary_improvement_pct"],
                    f"relative WAPE improvement {imp:.2f}% (min {rules['min_primary_improvement_pct']}%)",
                )
            )
            worst, worst_seg = 0.0, None
            for seg, w in chal.segment_wape.items():
                iw = inc.segment_wape.get(seg)
                if iw and iw > 0 and w == w:
                    d = (w - iw) / iw * 100
                    if d > worst:
                        worst, worst_seg = d, seg
            checks.append(
                _check(
                    "segment_degradation_limit",
                    worst <= rules["max_segment_degradation_pct"],
                    f"worst degradation {worst:.2f}% ({worst_seg}) (max {rules['max_segment_degradation_pct']}%)",
                )
            )
    checks.append(
        _check(
            "bias_limit",
            chal.bias_pct is not None and abs(chal.bias_pct) * 100 <= rules["max_abs_bias_pct"],
            f"pooled bias {None if chal.bias_pct is None else round(chal.bias_pct * 100, 2)}% over "
            f"{len(chal.eval_origins or [0])} window(s) (max +-{rules['max_abs_bias_pct']}%)"
            + (
                "; per-window (informational): " + ", ".join(f"{b * 100:+.1f}%" for b in chal.window_bias_pct)
                if chal.window_bias_pct and len(chal.window_bias_pct) > 1
                else ""
            ),
        )
    )
    if chal.coverage is None:
        checks.append(
            _check(
                "interval_coverage",
                not require_coverage,
                "coverage unavailable" + (" -> fail-closed" if require_coverage else " (not required)"),
            )
        )
    else:
        checks.append(
            _check(
                "interval_coverage",
                chal.coverage >= rules["min_interval_coverage"],
                f"coverage {chal.coverage:.3f} (min {rules['min_interval_coverage']})",
            )
        )
    if chal.train_seconds is not None:
        checks.append(
            _check(
                "train_runtime_limit",
                chal.train_seconds <= rules["max_train_seconds"],
                f"{chal.train_seconds:.0f}s (max {rules['max_train_seconds']})",
            )
        )
    if chal.batch_seconds is not None:
        checks.append(
            _check(
                "batch_runtime_limit",
                chal.batch_seconds <= rules["max_batch_forecast_seconds"],
                f"{chal.batch_seconds:.0f}s (max {rules['max_batch_forecast_seconds']})",
            )
        )
    ok = all(c["passed"] for c in checks)
    return {"promote": ok, "checks": checks, "failed": [c["check"] for c in checks if not c["passed"]]}
