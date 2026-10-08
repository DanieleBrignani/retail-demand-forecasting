-- state_id / cat_id must belong to the dataset's declared dimension values (vars valid_states / valid_categories).
select 'state_id' as dim, state_id as value from {{ ref('stg_series') }}
where state_id not in ({% for s in var('valid_states') %}'{{ s }}'{{ "," if not loop.last }}{% endfor %})
union all
select 'cat_id', cat_id from {{ ref('stg_series') }}
where cat_id not in ({% for s in var('valid_categories') %}'{{ s }}'{{ "," if not loop.last }}{% endfor %})
