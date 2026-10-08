{{ config(materialized='view') }}
-- Feature-ready relation: sales + calendar attributes + the BLOCK price with its availability day.
-- The block price is the median of ALL transaction lines of the 7-day block, so it is only knowable once the block has ended:
-- it must not be used as a feature for any origin day < block_price_known_from_day_num (the panel loader enforces this; see
-- docs/data_contract.md, "price availability"). NULL when no line was recorded in the block (never zero-filled).
select
    d.dataset_version, d.series_ord, s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id,
    d.day_num, d.date, d.sales,
    c.week_id, c.wday, c.month, c.year, c.event_name_1, c.event_type_1,
    p.sell_price as block_price,
    max(d.day_num) over (partition by d.dataset_version, d.series_ord, c.week_id) as block_price_known_from_day_num,
    (p.sell_price is not null) as block_price_recorded
from {{ ref('stg_sales_daily') }} d
join {{ ref('stg_series') }} s on s.dataset_version = d.dataset_version and s.series_ord = d.series_ord
join {{ ref('stg_calendar') }} c on c.dataset_version = d.dataset_version and c.day_num = d.day_num
left join {{ ref('stg_prices') }} p
  on p.dataset_version = d.dataset_version and p.store_id = s.store_id and p.item_id = s.item_id and p.week_id = c.week_id
