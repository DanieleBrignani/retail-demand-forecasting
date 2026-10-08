-- Price per store x item x 7-day BLOCK (transaction-derived median over the whole block; not an externally observed list price).
-- Absent blocks stay absent (never zero-filled). Known only after the block has ended (see fct_sales_feature_ready).
select dataset_version, store_id, item_id, week_id, sell_price
from {{ source('raw', 'prices') }}
where dataset_version = '{{ var("dataset_version") }}'
