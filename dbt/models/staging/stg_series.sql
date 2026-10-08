select dataset_version, series_ord, id, item_id, dept_id, cat_id, store_id, state_id
from {{ source('raw', 'series') }}
where dataset_version = '{{ var("dataset_version") }}'
