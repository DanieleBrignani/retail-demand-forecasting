-- Referential integrity and join safety, EXHAUSTIVE (no sampling).
-- A join cannot fan out or drop sales rows when (a) every right-hand join key is unique and (b) every sales row finds its partner.
-- Proving (a) and (b) over the full panel is a cheap aggregate/anti-join, replacing the former full 3-way count that took 61 minutes
-- on 7M rows and the interim sampled count. (Key uniqueness of prices is also covered by assert_key_uniqueness.)
select 'orphan_price' as kind, p.store_id || '/' || p.item_id as k from {{ ref('stg_prices') }} p
left join (select distinct store_id, item_id from {{ ref('stg_series') }}) s using (store_id, item_id)
where s.item_id is null
union all
select 'duplicate_series_ord', series_ord::text from {{ ref('stg_series') }} group by series_ord having count(*) > 1
union all
select 'duplicate_calendar_day', day_num::text from {{ ref('stg_calendar') }} group by day_num having count(*) > 1
union all
select 'sales_without_series', d.series_ord::text from {{ ref('stg_sales_daily') }} d
left join {{ ref('stg_series') }} s on s.series_ord = d.series_ord where s.series_ord is null group by d.series_ord
union all
select 'sales_without_calendar_day', d.day_num::text from {{ ref('stg_sales_daily') }} d
left join {{ ref('stg_calendar') }} c on c.day_num = d.day_num where c.day_num is null group by d.day_num
