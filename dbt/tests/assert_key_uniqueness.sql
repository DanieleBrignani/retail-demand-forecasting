-- Duplicate keys: sales (series, day), series (item, store), prices (store, item, week)
select 'sales' as src, series_ord::text as k1, day_num::text as k2, null as k3 from {{ ref('stg_sales_daily') }}
  group by series_ord, day_num having count(*) > 1
union all
select 'series', item_id, store_id, null from {{ ref('stg_series') }} group by item_id, store_id having count(*) > 1
union all
select 'prices', store_id, item_id, week_id::text from {{ ref('stg_prices') }}
  group by store_id, item_id, week_id having count(*) > 1
