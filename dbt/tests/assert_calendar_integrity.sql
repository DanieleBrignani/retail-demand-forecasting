-- Calendar coverage: contiguous dates/day numbers, 7 days per price week, covers every sales day.
select 'gap' as kind, day_num::text as k from (
  select day_num, date, lag(date) over (order by day_num) as prev_date, lag(day_num) over (order by day_num) as prev_day
  from {{ ref('stg_calendar') }}) t
where prev_date is not null and (date - prev_date <> 1 or day_num - prev_day <> 1)
union all
select 'week_over_7_days', week_id::text from {{ ref('stg_calendar') }} group by week_id having count(*) > 7
union all
select 'sales_beyond_calendar', 'x' where (select max(day_num) from {{ ref('stg_sales_daily') }}) > (select max(day_num) from {{ ref('stg_calendar') }})
