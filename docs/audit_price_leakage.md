# Audit — price look-ahead defect (feature version `fv1`), correction (`fv2`) and affected experiments

## The defect
The adapter computes the price of a 7-day block as the **median over all the block's transaction lines**. `Panel.as_of(origin)` exposed the price of the origin's *current* block. When the origin lies inside the block,
the exposed price therefore included transactions after the origin.

**Reproduction (real adapter and feature functions; `tests/test_price_availability.py`):** same product and channel; day 1 price 10 and day 4 price 10 → `price_origin` at origin day 1 = 10.
Change only the future day-4 transaction price to 100 → `price_origin` at origin day 1 = **55**. Before the fix the test failed with exactly this difference (10.0 vs 55.0).

## Why the existing tests missed it
`tests/test_features_leakage.py` mutates the **already aggregated price matrix** for blocks *after* the origin's block (`price_weekly[:, w+1:]`). The defect lives upstream, in how the block aggregate was built from transactions and in which block was
exposed; a test that starts from a precomputed matrix cannot observe it. The new tests start from transaction rows and go through the adapter.

## The fix (policy "completed-block prices")
A block price is visible only once the block has ended. See [`data_contract.md`](data_contract.md) §3. Price features were re-versioned (`fv2` for every feature set containing the price group; sets without price keep `fv1` because their semantics did not change).
Also fixed: the warehouse view `fct_sales_feature_ready` now exposes `block_price` and `block_price_known_from_day_num` instead of an unqualified `sell_price`.

## Which experiments are affected (dependency check, not assumption)
| experiment | reads price? | status |
|---|---|---|
| Daily baselines (naive, seasonal naive, mean, seasonal mean) | no | not affected by price; affected by the **universe change** (scored on all 9,414 series incl. not-yet-launched products) |
| Daily LightGBM feature sets A, B, C (no price group) | no | not affected by price; affected by the universe change (encoders, training rows) |
| Daily LightGBM **feature set D** (validation, holdout, feature ablation D vs C), model registry `retailfc_lgbm` v1 and **v2**, variants study, the "price: no demonstrable value" conclusion | **yes (fv1)** | **tainted**: the price features could see future transactions of the origin's block. A leak can only inflate apparent accuracy; the ablation showed no value anyway, but the numbers are not valid as evidence. v2 was approved by the (post-hoc) gate using feature set D → **its approval is not valid evidence** and the model is superseded. |
| EDA price coverage / drift PSI (`eda.py`, `monitoring/core.py`) | yes, descriptive | not features; they describe realised data. Not used for training. |
| Weekly v1 and weekly v2 | **no** (`retailfc.weekly.FEATURES` has no price input; asserted by a test) | not affected by price; v1 affected by the universe change, v2 uses the corrected universe |
| `planned` price scenario | scenario by construction reads target-week prices | never used for reported results |

Superseded historical reports stay in the repository with a "superseded" banner (see `docs/results_*`). Corrected daily results (feature set D under `fv2`, universe policy) are listed as **pending** unless a corrected run is reported in the results documents.
