{{ config(post_hook=[
    "create index if not exists stg_sales_daily_ix on {{ this }} (dataset_version, series_ord, day_num)"]) }}
-- item x store x day. Zero is an observation; no row is fabricated for unobserved days.
select s.dataset_version, s.series_ord, s.day_num, c.date, s.sales
from {{ source('raw', 'sales_long') }} s
join {{ ref('stg_calendar') }} c
  on c.dataset_version = s.dataset_version and c.day_num = s.day_num
where s.dataset_version = '{{ var("dataset_version") }}'
