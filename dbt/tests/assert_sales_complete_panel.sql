-- No missing observations: rows == series x observed days (zero sales are rows; absent rows are failures).
with exp as (
  select (select count(*) from {{ ref('stg_series') }}) * (select last_sales_day from {{ source('raw','dataset_versions') }}
          where dataset_version = '{{ var("dataset_version") }}') as expected
), got as (select count(*) as n from {{ ref('stg_sales_daily') }})
select * from exp, got where expected <> n
