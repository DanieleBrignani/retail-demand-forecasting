"""Generate docs/feature_dictionary.md from the feature registry (single source of truth)."""

from __future__ import annotations

from pathlib import Path

from .spec import FEATURE_CODE_VERSION, GROUP_ORDER, SPECS, columns_for, feature_version

HEADER = """# Feature dictionary

_Generated from `src/retailfc/features/spec.py` (feature code version {ver}). Do not edit by hand: run
`retailfc docs-feature-dictionary`._

Notation: **o** = forecast origin (last day with sales), **h** = horizon (1..28), **t = o + h** the target day,
**y_d** = sales on day d. Every sales-derived feature uses days **<= o**. Rows are built per (series, origin, horizon):
the direct global strategy never feeds a forecasted or future actual value back into a feature.

Feature groups used in the ablation (see `docs/experiment_plan.md`):

| Set | Groups | Feature version |
|---|---|---|
{sets}

Price scenarios: **observed** (primary, operational: prices listed by the origin's week, carried forward, never back-filled)
and **planned** (separate, labelled assumption: target-week prices known in advance; the source does not prove this).
"""


def write_feature_dictionary(out: Path, cfg: dict | None = None) -> None:
    from .. import config as C

    cfg = cfg or C.load_config()
    sets = []
    for k, g in cfg["features"]["feature_sets"].items():
        sets.append(f"| {k} | {', '.join(g)} | `{feature_version(columns_for(g), 'observed')}` |")
    lines = [HEADER.format(ver=FEATURE_CODE_VERSION, sets="\n".join(sets))]
    for g in GROUP_ORDER:
        specs = [s for s in SPECS if s.group == g]
        if not specs:
            continue
        lines.append(f"\n## Group: `{g}`\n")
        for s in specs:
            lines.append(
                f"### `{s.name}`{' (categorical)' if s.categorical else ''}\n"
                f"- **Definition:** {s.definition}\n- **Time reference:** {s.time_ref}\n"
                f"- **Availability assumption:** {s.availability}\n- **Missing-value policy:** {s.missing}\n"
                f"- **Business interpretation:** {s.meaning}\n"
            )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
