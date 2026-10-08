select
    dataset_version, date, week_id, weekday, wday, month, year, d,
    cast(substr(d, 3) as int) as day_num,
    event_name_1, event_type_1, event_name_2, event_type_2
from {{ source('raw', 'calendar') }}
where dataset_version = '{{ var("dataset_version") }}'
