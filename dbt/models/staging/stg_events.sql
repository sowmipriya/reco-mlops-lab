-- Typed, cleaned and deduplicated events. Kafka delivers at-least-once,
-- so the same event_id can arrive twice: keep the latest load.
with source as (
    select * from {{ source('raw', 'events') }}
)

select
    event_id,
    customer_id,
    session_id,
    product_id,
    lower(event_type)            as event_type,
    timestamp(event_ts)          as event_ts,
    cast(quantity as int64)      as quantity,
    lower(channel)               as channel,
    _loaded_at
from source
where event_id is not null
  and customer_id is not null
  and product_id is not null
qualify row_number() over (partition by event_id order by _loaded_at desc) = 1
