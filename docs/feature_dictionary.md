# Feature dictionary

_Generated from `src/retailfc/features/spec.py` (feature code version 1). Do not edit by hand: run
`retailfc docs-feature-dictionary`._

Notation: **o** = forecast origin (last day with sales), **h** = horizon (1..28), **t = o + h** the target day,
**y_d** = sales on day d. Every sales-derived feature uses days **<= o**. Rows are built per (series, origin, horizon):
the direct global strategy never feeds a forecasted or future actual value back into a feature.

Feature groups used in the ablation (see `docs/experiment_plan.md`):

| Set | Groups | Feature version |
|---|---|---|
| A | history, ids, horizon | `fv1-694c98f5` |
| B | history, ids, horizon, calendar | `fv1-d15453d9` |
| C | history, ids, horizon, calendar, events | `fv1-f6d7857e` |
| D | history, ids, horizon, calendar, events, price | `fv1-edce4063` |

Price scenarios: **observed** (primary, operational: prices listed by the origin's week, carried forward, never back-filled)
and **planned** (separate, labelled assumption: target-week prices known in advance; the source does not prove this).


## Group: `history`

### `last_sale`
- **Definition:** y_o
- **Time reference:** day o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** never missing (o >= 1)
- **Business interpretation:** Most recent demand signal.

### `lag_7`
- **Definition:** y_{o-7}
- **Time reference:** day o-7
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if o <= 7
- **Business interpretation:** Same weekday as the origin one week earlier.

### `lag_14`
- **Definition:** y_{o-14}
- **Time reference:** day o-14
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if o <= 14
- **Business interpretation:** Two weeks back.

### `lag_28`
- **Definition:** y_{o-28}
- **Time reference:** day o-28
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if o <= 28
- **Business interpretation:** Four weeks back.

### `roll_mean_7`
- **Definition:** mean(y_{o-6..o})
- **Time reference:** days o-6..o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if o < 7
- **Business interpretation:** Short-term level.

### `roll_mean_14`
- **Definition:** mean(y_{o-13..o})
- **Time reference:** days o-13..o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if o < 14
- **Business interpretation:** Medium-term level.

### `roll_mean_28`
- **Definition:** mean(y_{o-27..o})
- **Time reference:** days o-27..o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if o < 28
- **Business interpretation:** Monthly level.

### `roll_std_7`
- **Definition:** std(y_{o-6..o}), ddof=0
- **Time reference:** days o-6..o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if o < 7
- **Business interpretation:** Short-term volatility.

### `roll_std_28`
- **Definition:** std(y_{o-27..o}), ddof=0
- **Time reference:** days o-27..o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if o < 28
- **Business interpretation:** Monthly volatility.

### `zero_prop_28`
- **Definition:** share of zero-sales days in o-27..o
- **Time reference:** days o-27..o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if o < 28
- **Business interpretation:** Intermittency; zero sales do not prove zero demand.

### `days_since_sale`
- **Definition:** o - (last day <= o with y>0)
- **Time reference:** days <= o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if no positive sale yet
- **Business interpretation:** Recency of demand; large values may mean inactivity or stock-out (not observable).

### `age_since_first_sale`
- **Definition:** o - (first day with y>0) + 1; 0 if none
- **Time reference:** days <= o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** 0 if none
- **Business interpretation:** Apparent time on sale; first positive sale is not necessarily product launch.

### `seas_lag_1`
- **Definition:** y_{t-7*ceil(h/7)}
- **Time reference:** last same-weekday-as-target day <= o
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if day < 1
- **Business interpretation:** Most recent observation of the target's weekday; anchored to the origin, never to t.

### `seas_lag_2`
- **Definition:** y_{t-7*ceil(h/7)-7}
- **Time reference:** same weekday, one week earlier
- **Availability assumption:** Sales up to and including the origin day o.
- **Missing-value policy:** NaN if day < 1
- **Business interpretation:** Weekly pattern persistence.


## Group: `ids`

### `item_id` (categorical)
- **Definition:** integer code of item_id
- **Time reference:** static
- **Availability assumption:** Static catalogue attributes; encoders fitted on training data only, unseen id -> -1 (treated as missing).
- **Missing-value policy:** unseen -> -1
- **Business interpretation:** Item identity.

### `dept_id` (categorical)
- **Definition:** integer code of dept_id
- **Time reference:** static
- **Availability assumption:** Static catalogue attributes; encoders fitted on training data only, unseen id -> -1 (treated as missing).
- **Missing-value policy:** unseen -> -1
- **Business interpretation:** Department.

### `cat_id` (categorical)
- **Definition:** integer code of cat_id
- **Time reference:** static
- **Availability assumption:** Static catalogue attributes; encoders fitted on training data only, unseen id -> -1 (treated as missing).
- **Missing-value policy:** unseen -> -1
- **Business interpretation:** Category.

### `store_id` (categorical)
- **Definition:** integer code of store_id
- **Time reference:** static
- **Availability assumption:** Static catalogue attributes; encoders fitted on training data only, unseen id -> -1 (treated as missing).
- **Missing-value policy:** unseen -> -1
- **Business interpretation:** Store.

### `state_id` (categorical)
- **Definition:** integer code of state_id
- **Time reference:** static
- **Availability assumption:** Static catalogue attributes; encoders fitted on training data only, unseen id -> -1 (treated as missing).
- **Missing-value policy:** unseen -> -1
- **Business interpretation:** State.


## Group: `horizon`

### `h`
- **Definition:** forecast horizon in days (1..28)
- **Time reference:** t - o
- **Availability assumption:** Known by construction.
- **Missing-value policy:** never missing
- **Business interpretation:** Lets the direct global model learn horizon-specific behaviour.


## Group: `calendar`

### `t_dow` (categorical)
- **Definition:** weekday of target day (wday 1..7)
- **Time reference:** day t
- **Availability assumption:** Calendar is published in advance; assumed known >= 28 days ahead of the origin (the calendar table covers the target dates).
- **Missing-value policy:** never missing
- **Business interpretation:** Weekly seasonality.

### `t_month` (categorical)
- **Definition:** month of target day
- **Time reference:** day t
- **Availability assumption:** Calendar is published in advance; assumed known >= 28 days ahead of the origin (the calendar table covers the target dates).
- **Missing-value policy:** never missing
- **Business interpretation:** Yearly seasonality.

### `t_dom`
- **Definition:** day of month of target day
- **Time reference:** day t
- **Availability assumption:** Calendar is published in advance; assumed known >= 28 days ahead of the origin (the calendar table covers the target dates).
- **Missing-value policy:** never missing
- **Business interpretation:** Pay-cycle effects.


## Group: `events`

### `t_event_type_1` (categorical)
- **Definition:** event_type_1 code of target day (vocabulary fit on training calendar)
- **Time reference:** day t
- **Availability assumption:** Calendar is published in advance; assumed known >= 28 days ahead of the origin (the calendar table covers the target dates).
- **Missing-value policy:** no event -> code of 'none'; unseen -> -1
- **Business interpretation:** Holiday/sporting/cultural/religious type.

### `t_event_name_1` (categorical)
- **Definition:** event_name_1 code of target day
- **Time reference:** day t
- **Availability assumption:** Calendar is published in advance; assumed known >= 28 days ahead of the origin (the calendar table covers the target dates).
- **Missing-value policy:** no event -> code of 'none'; unseen -> -1
- **Business interpretation:** Named event effects.


## Group: `price`

### `price_origin`
- **Definition:** last listed price observed in weeks <= week(o), carried forward
- **Time reference:** weeks <= week(o)
- **Availability assumption:** Prices listed up to the origin's week are assumed known at o.
- **Missing-value policy:** NaN if never listed by o (never filled with 0, never back-filled)
- **Business interpretation:** Current price level.

### `price_age_weeks`
- **Definition:** weeks since the carried-forward price was last listed
- **Time reference:** weeks <= week(o)
- **Availability assumption:** as above
- **Missing-value policy:** NaN if never listed
- **Business interpretation:** Price staleness.

### `price_missing`
- **Definition:** 1 if no price listed by o else 0
- **Time reference:** weeks <= week(o)
- **Availability assumption:** as above
- **Missing-value policy:** never missing
- **Business interpretation:** Availability indicator (not an imputation).

### `price_chg_4w`
- **Definition:** price_origin / price_origin_{4 weeks earlier} - 1
- **Time reference:** weeks <= week(o)
- **Availability assumption:** as above
- **Missing-value policy:** NaN if either price unavailable
- **Business interpretation:** Recent promotion / price-change signal.

### `price_rel_max`
- **Definition:** price_origin / max listed price up to o
- **Time reference:** weeks <= week(o)
- **Availability assumption:** as above
- **Missing-value policy:** NaN if never listed
- **Business interpretation:** Discount depth relative to history.


## Group: `price_planned`

### `target_price`
- **Definition:** listed price of the target week (SCENARIO: planned prices known)
- **Time reference:** week(t)
- **Availability assumption:** Only in the labelled 'planned' scenario; assumes the retailer's price plan is known 28 days ahead -- NOT proven for this dataset.
- **Missing-value policy:** NaN if not listed
- **Business interpretation:** Planned price.

### `target_price_chg`
- **Definition:** target_price / price_origin - 1 (SCENARIO: planned)
- **Time reference:** week(t)
- **Availability assumption:** as above
- **Missing-value policy:** NaN if either missing
- **Business interpretation:** Planned price change.
