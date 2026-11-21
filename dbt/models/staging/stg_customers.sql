select
    customer_id,
    timestamp(first_seen_ts) as first_seen_ts
from {{ source('raw', 'customers') }}
