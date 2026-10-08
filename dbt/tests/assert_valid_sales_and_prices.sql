-- Negative / null sales and non-positive / null prices are invalid (zero sales are valid).
select 'sales' as src, series_ord::text as k from {{ ref('stg_sales_daily') }} where sales < 0 or sales is null
union all
select 'prices', store_id || '/' || item_id from {{ ref('stg_prices') }} where sell_price is null or sell_price <= 0
