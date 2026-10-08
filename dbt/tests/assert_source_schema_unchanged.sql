-- Unexpected schema changes in raw tables (added/removed columns) fail the build.
with expected(tbl, col) as (values
 ('calendar','date'),('calendar','week_id'),('calendar','weekday'),('calendar','wday'),('calendar','month'),('calendar','year'),
 ('calendar','d'),('calendar','event_name_1'),('calendar','event_type_1'),('calendar','event_name_2'),('calendar','event_type_2'),
 ('calendar','dataset_version'),
 ('prices','store_id'),('prices','item_id'),('prices','week_id'),('prices','sell_price'),('prices','dataset_version'),
 ('series','series_ord'),('series','id'),('series','item_id'),('series','dept_id'),('series','cat_id'),('series','store_id'),
 ('series','state_id'),('series','dataset_version'),
 ('sales_long','dataset_version'),('sales_long','series_ord'),('sales_long','day_num'),('sales_long','sales')),
actual as (select table_name as tbl, column_name as col from information_schema.columns
           where table_schema = 'raw' and table_name in ('calendar','prices','series','sales_long'))
select 'missing' as kind, e.tbl, e.col from expected e left join actual a using (tbl, col) where a.col is null
union all
select 'unexpected', a.tbl, a.col from actual a left join expected e using (tbl, col) where e.col is null
